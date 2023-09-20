import os

import torch
import ipdb
import open3d as o3d
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from utils_elias.metrics import meanAveragePrecision, PanopticEval


def compute_mAP(pred_leaf_ids, pred_leaf_id_probs, leaf_labels, cfg):
    n_samples = len(pred_leaf_ids)
    # compute mAP
    mAP_calculator = meanAveragePrecision(min_leaf_points=len(pred_leaf_ids[0]) * 0.001)
    mAP = 0
    for sample in range(n_samples):
        rec, prec, curr_mAP = mAP_calculator.compute_class_AP(
            pred_leaf_ids[sample],
            pred_leaf_id_probs[sample].float(),
            leaf_labels[sample],
            iou_threshold=0.5,
        )
        mAP += curr_mAP
    mAP /= n_samples
    return mAP


def compute_pq(pred_leaf_ids, leaf_labels, cfg):
    n_samples = len(pred_leaf_ids)
    panQual_calculator = PanopticEval(
        n_classes=3,
        min_points=len(pred_leaf_ids[0]) * 0.001,
        ignore=[
            2,
        ],
    )
    for sample in range(n_samples):
        valid_points = leaf_labels[sample] >= 0
        # use semantic mask to compute panoptic quality only on valid points by ignoring sem class 2
        valid_mask = torch.ones_like(pred_leaf_ids[sample]).long().numpy()
        valid_mask[~valid_points] = 2
        panQual_calculator.addBatch(
            pred_leaf_ids[sample].long().numpy() > 0,
            pred_leaf_ids[sample].long().numpy(),
            valid_mask,
            leaf_labels[sample].long().numpy(),
        )
    (
        class_all_PQ,
        class_all_SQ,
        class_all_RQ,
        class_all_PR,
        class_all_RC,
    ) = panQual_calculator.getPQ(return_pr_rc=True)
    return class_all_PQ, class_all_SQ, class_all_RQ, class_all_PR, class_all_RC


def eval_only(batch, cfg):
    """Evaluates a batch

    Args:
        batch (dict): Contains:
                        points: Nx3 tensor with point positions
                        leaf_labels: Nx1 tensor containing leaf labels
                        sampled_indices: Nx1 binary mask for postprocessed points (during validation steps postprocessing is performed on downsample cloud to speed up)
                        pred_leaf_ids: Nx1 tensor with predicted leaf ids
                        pred_leaf_id_probs: Nx1 tensor with predicted leaf id probabilities
        cfg (dict): Parameters from config file
    """
    pred_leaf_ids = batch["pred_leaf_ids"]
    pred_leaf_id_probs = batch["pred_leaf_id_probs"]
    leaf_labels = batch["leaf_labels"]
    metrics = {}
    mAP = compute_mAP(pred_leaf_ids, pred_leaf_id_probs, leaf_labels, cfg)
    metrics["mAP"] = mAP
    PQ, SQ, RQ, PR, RC = compute_pq(pred_leaf_ids, leaf_labels, cfg)
    metrics["PQ"] = PQ
    metrics["SQ"] = SQ
    metrics["RQ"] = RQ
    metrics["PR"] = PR
    metrics["RC"] = RC

    return metrics


def compute_metrics(
    cfg,
    preds,
    confid_mode,
    confid_inlier_ratio=None,
    global_confid_thres=False,
    gt_thres=False,
):
    mean_metrics = {}
    mean_metrics["mAP"] = 0
    mean_metrics["PQ"] = 0
    mean_metrics["SQ"] = 0
    mean_metrics["RQ"] = 0
    mean_metrics["PR"] = 0
    mean_metrics["RC"] = 0
    batch = {}
    batch["points"] = []
    batch["leaf_labels"] = []
    batch["pred_leaf_ids"] = []
    batch["pred_leaf_id_probs"] = []
    # batch["leaf_labels"] = []
    metrics = []
    leaf_sizes = np.empty(0)

    if confid_inlier_ratio != None:
        if confid_mode == "network":
            confid_field = "pred_leaf_confid"
        elif confid_mode == "hdbscan":
            confid_field = "hdbscan_probs"
        elif confid_mode == "cluster_var":
            confid_field = "cluster_var"
        elif confid_mode == "gt":
            confid_field = "gt_iou"
        else:
            raise ValueError("confid mode not supported")
        # find confid_thres based on all patches
        if global_confid_thres:
            all_confids = torch.empty(0)

            for pred in os.listdir(preds):
                if pred.split(".")[-1] != "ply":
                    continue
                pcd = o3d.t.io.read_point_cloud(os.path.join(preds, pred))
                pred_leaf_confid = torch.tensor(
                    pcd.point[confid_field].numpy()
                ).unsqueeze(0)
                all_confids = torch.cat((all_confids, pred_leaf_confid.unique()[1:]))
            confid_thres = stats.scoreatpercentile(
                all_confids.numpy() * (-1), confid_inlier_ratio * 100
            ) * (-1)

    for pred in os.listdir(preds):
        if pred.split(".")[-1] != "ply":
            continue
        # if pred != "patch_22_31.ply":
        #     continue

        pcd = o3d.t.io.read_point_cloud(os.path.join(preds, pred))
        batch["points"] = torch.tensor(pcd.point["positions"].numpy()).unsqueeze(0)
        batch["leaf_labels"] = torch.tensor(pcd.point["gt_leaf_ids"].numpy()).unsqueeze(
            0
        )
        # import ipdb;ipdb.set_trace()  # fmt: skip
        ############### DEBUG
        if batch["leaf_labels"].unique()[0] != 1:
            batch["leaf_labels"][batch["leaf_labels"] == 0] = -1
        ######################
        batch["pred_leaf_ids"] = torch.tensor(
            pcd.point["pred_leaf_ids"].numpy()
        ).unsqueeze(0)

        ###############3
        min_points = len(batch["pred_leaf_ids"][0]) * 0.05
        unique_lab, counts = batch["leaf_labels"].unique(return_counts=True)
        # subtract one for invalid (-1)
        n_gt_leaves = len(unique_lab[counts > min_points]) - 1
        #################
        # filter predictions with low confidence
        if confid_inlier_ratio != None:
            batch["pred_leaf_id_probs"] = torch.tensor(
                pcd.point[confid_field].numpy()
            ).unsqueeze(0)
            batch["pred_leaf_id_probs"][batch["leaf_labels"] == -1] = -1
            if confid_mode == "cluster_var":
                valid_mask = batch["pred_leaf_id_probs"] >= 0
                batch["pred_leaf_id_probs"][valid_mask] *= -1
                batch["pred_leaf_id_probs"][valid_mask] -= batch["pred_leaf_id_probs"][
                    valid_mask
                ].min()
                pcd.point[confid_field] = o3d.core.Tensor(
                    batch["pred_leaf_id_probs"][0].numpy()
                )
            # average point probs inside the same leaf
            leaf_list = batch["pred_leaf_ids"].unique()
            for leaf in leaf_list[1:]:
                leaf_mask = batch["pred_leaf_ids"] == leaf
                batch["pred_leaf_id_probs"][leaf_mask] = batch["pred_leaf_id_probs"][
                    leaf_mask
                ].mean()

            if not global_confid_thres:
                min_points = (
                    len(batch["pred_leaf_ids"][0])
                    * cfg["data"]["min_leaf_point_ratio_inference"]
                )
                if gt_thres:
                    unique_lab, counts = batch["leaf_labels"].unique(return_counts=True)
                    # subtract one for invalid (-1)
                    n_gt_leaves = len(unique_lab[counts > min_points]) - 1
                    target_n_leaves = int(n_gt_leaves * confid_inlier_ratio)
                    relevant_probs = batch["pred_leaf_id_probs"][
                        batch["leaf_labels"] >= 0
                    ]
                    unique_probs, counts = relevant_probs.unique(return_counts=True)
                    confid_thres = unique_probs[-target_n_leaves].item()
                else:
                    relevant_probs = batch["pred_leaf_id_probs"][
                        batch["leaf_labels"] >= 0
                    ]
                    unique_probs, counts = relevant_probs.unique(return_counts=True)
                    # for prob in unique_probs:
                    #     leaf_mask = relevant_probs == prob
                    #     relevant_probs[leaf_mask] = -1
                    # confid_thres = stats.scoreatpercentile(
                    #     unique_probs[1:][counts[1:] > min_points].numpy() * (-1),
                    #     confid_inlier_ratio * 100,
                    # ) * (-1)
                    confid_thres = stats.scoreatpercentile(
                        unique_probs[1:].numpy() * (-1),
                        confid_inlier_ratio * 100,
                    ) * (-1)
                    # print(confid_inlier_ratio, confid_thres)
            batch["pred_leaf_ids"][batch["pred_leaf_id_probs"] < confid_thres] = -1
            # remaining_confids = batch["pred_leaf_id_probs"][~(batch["pred_leaf_id_probs"]<confid_thres)].unique()
            # print("remaining confids", remaining_confids)
            batch[confid_field] = torch.tensor(
                pcd.point[confid_field].numpy()
            ).unsqueeze(0)
            # ############## debug

            # label_mask = (batch['leaf_labels']>1).squeeze()
            # batch['leaf_labels']= batch['leaf_labels'][:, label_mask, :]
            # batch["pred_leaf_ids"] = batch["pred_leaf_ids"][:, label_mask, :]
            # batch['pred_leaf_id_probs'] = batch['pred_leaf_id_probs'][:, label_mask, :]
            # batch["points"] = batch["points"][:, label_mask, :]
            # batch["pred_leaf_confid"] = batch["pred_leaf_confid"][:, label_mask, :]
            # #############
            # compute leaf area by voxel downsampling and counting points
            pcd_smpl = pcd.voxel_down_sample(0.01)
            pcd_smpl.point["pred_leaf_ids"].numpy()[
                pcd_smpl.point[confid_field].numpy() < confid_thres
            ] = -1
            leaf_sizes = np.append(
                leaf_sizes,
                np.unique(pcd_smpl.point["pred_leaf_ids"].numpy(), return_counts=True)[
                    1
                ][1:],
            )
        else:
            batch["pred_leaf_id_probs"] = torch.ones_like(batch["pred_leaf_ids"])
            # compute leaf area by voxel downsampling and counting points
            pcd_smpl = pcd.voxel_down_sample(0.01)
            leaf_sizes = np.append(
                leaf_sizes,
                np.unique(pcd_smpl.point["pred_leaf_ids"].numpy(), return_counts=True)[
                    1
                ][1:],
            )

        batch["sampled_indices"] = (
            torch.ones_like(batch["pred_leaf_ids"][0]).squeeze().bool()
        )
        # visualize_ious(batch["points"],
        #                     batch["pred_leaf_ids"],
        #                     batch['leaf_labels'],
        #                     instance_list[0], instance_max_ious[0], instance_max_ious[0])
        res = eval_only(batch, cfg)
        metrics.append(res)
        # _, instance_list, instance_max_ious = compute_gt_ious(
        #         batch["pred_leaf_ids"],
        #         batch["leaf_labels"],
        #         1,
        #     )
        # remapped_preds = map_instances(batch["pred_leaf_ids"], batch["leaf_labels"])
        # wrong_pred_mask = torch.logical_and(batch["leaf_labels"] != -1, batch["leaf_labels"] != remapped_preds)
        # colored_pcd = compute_leaf_colors(batch["points"][0], batch["pred_leaf_ids"][0])
        # # colored_pcd.colors = np.asarray(colored_pcd.colors)
        # np.asarray(colored_pcd.colors)[wrong_pred_mask.squeeze()] *= 0.5
        # # np.asarray(colored_pcd.colors)[(batch["leaf_labels"] == -1).squeeze()] = np.array((0.5,0.5,0.5))
        # colored_pcd = colored_pcd.select_by_index(np.where((batch["leaf_labels"] != -1).squeeze())[0])
        # o3d.visualization.draw_geometries([colored_pcd])

        print(
            "plant:",
            pred,
            "PQ",
            res["PQ"],
            "SQ",
            res["SQ"],
            "RQ",
            res["RQ"],
            "RC",
            res["RC"],
            "PR",
            res["PR"],
            "PQd",
            res["SQ"] * res["PR"],
            "mAP",
            res["mAP"],
            "n",
            n_gt_leaves,
        )

    for i in range(len(metrics)):
        mean_metrics["mAP"] += metrics[i]["mAP"] / len(metrics)
        mean_metrics["PQ"] += metrics[i]["PQ"] / len(metrics)
        mean_metrics["SQ"] += metrics[i]["SQ"] / len(metrics)
        mean_metrics["RQ"] += metrics[i]["RQ"] / len(metrics)
        mean_metrics["PR"] += metrics[i]["PR"] / len(metrics)
        mean_metrics["RC"] += metrics[i]["RC"] / len(metrics)

    print(mean_metrics)

    # if res["PQ"] < 0.5:
    #     visualize_errors(batch)
