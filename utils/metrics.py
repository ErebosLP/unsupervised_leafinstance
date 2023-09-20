import numpy as np
import torch
import ipdb


class meanAveragePrecision:
    def __init__(self, min_leaf_points=0) -> None:
        self.min_leaf_points = min_leaf_points

    def points_iou(self, pred_cluster, gt_cluster):
        intersection = float(len(np.intersect1d(pred_cluster, gt_cluster)))
        return intersection / float(len(pred_cluster) + len(gt_cluster) - intersection)

    def find_best_match(self, pred_cluster, gt_clusters):
        iou_max = -np.inf
        best_match = -1
        for i, gt_cluster in enumerate(gt_clusters):
            iou = self.points_iou(pred_cluster, gt_cluster)
            if iou > iou_max:
                iou_max = iou
                best_match = i
        return iou_max, best_match

    def voc_ap(self, recall, precision):
        """ap = voc_ap(recall, precision)
        Compute PASCAL VOC AP given precision and recall.
        recall and precision contain one element per detected instance,
        ordered by certainty (most certain element first)
        (see here for an explanation https://github.com/rafaelpadilla/Object-Detection-Metrics)
        """
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], [recall], [1.0]))
        mpre = np.concatenate(([0.0], [precision], [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def compute_class_AP(
        self, pred_instance_ids, pred_instance_probs, gt_instance_ids, iou_threshold
    ):
        # Compute predicted clusters
        pred_instances, pred_counts = torch.unique(
            pred_instance_ids, return_counts=True
        )
        pred_instances = pred_instances[pred_counts > self.min_leaf_points]
        pred_clusters = []
        pred_cluster_probs = []
        for instance in pred_instances:
            cluster = torch.where(pred_instance_ids == instance)[0]
            pred_clusters.append(cluster)
            cluster_probs = pred_instance_probs[pred_instance_ids == instance]
            pred_cluster_probs.append(cluster_probs.mean())
        # Compute cluster probabilities

        # Compute gt clusters
        gt_instances, gt_counts = torch.unique(gt_instance_ids, return_counts=True)
        gt_instances = gt_instances[gt_counts > self.min_leaf_points]
        gt_clusters = []
        for instance in gt_instances:
            cluster = torch.where(gt_instance_ids == instance)[0]
            gt_clusters.append(cluster)

        visited = torch.zeros(len(gt_clusters), dtype=torch.bool)
        # sort clusters by probability
        pred_clusters = [
            x
            for _, x in sorted(
                zip(pred_cluster_probs, pred_clusters),
                key=lambda x: x[0].item(),
                reverse=True,
            )
        ]
        tp = torch.zeros(len(pred_clusters))
        fp = torch.zeros(len(pred_clusters))
        for p, pred_cluster in enumerate(pred_clusters):
            iou_max, best_match = self.find_best_match(pred_cluster, gt_clusters)

            if iou_max < iou_threshold:
                fp[p] = 1
                continue
            if visited[best_match]:
                fp[p] = 1
                continue

            visited[best_match] = True
            tp[p] = 1

        fp = fp.sum()
        tp = tp.sum()
        recall = tp / float(len(gt_clusters))
        prec = tp / (tp + fp)

        ap = self.voc_ap(recall.numpy(), prec.numpy())
        return recall, prec, ap


class PanopticEval:
    """Panoptic evaluation using numpy

    authors: Andres Milioto and Jens Behley
    """

    def __init__(
        self, n_classes, device=None, ignore=[], offset=2**32, min_points=30
    ):
        self.n_classes = n_classes
        assert device == None
        self.ignore = np.array(ignore, dtype=np.int64)
        self.include = np.array(
            [n for n in range(self.n_classes) if n not in self.ignore], dtype=np.int64
        )

        self.reset()
        self.offset = offset  # largest number of instances in a given scan
        self.min_points = (
            min_points  # smallest number of points to consider instances in gt
        )
        self.eps = 1e-15

    def num_classes(self):
        return self.n_classes

    def reset(self):
        # general things
        # iou stuff
        self.px_iou_conf_matrix = np.zeros(
            (self.n_classes, self.n_classes), dtype=np.int64
        )
        # panoptic stuff
        self.pan_tp = np.zeros(self.n_classes, dtype=np.int64)
        self.pan_iou = np.zeros(self.n_classes, dtype=np.double)
        self.pan_fp = np.zeros(self.n_classes, dtype=np.int64)
        self.pan_fn = np.zeros(self.n_classes, dtype=np.int64)

    ################################# IoU STUFF ##################################
    def addBatchSemIoU(self, x_sem, y_sem):
        # idxs are labels and predictions
        idxs = np.stack([x_sem, y_sem], axis=0)

        # make confusion matrix (cols = gt, rows = pred)
        np.add.at(self.px_iou_conf_matrix, tuple(idxs), 1)

    def getSemIoUStats(self):
        # clone to avoid modifying the real deal
        conf = self.px_iou_conf_matrix.copy().astype(np.double)
        # remove fp from confusion on the ignore classes predictions
        # points that were predicted of another class, but were ignore
        # (corresponds to zeroing the cols of those classes, since the predictions
        # go on the rows)
        conf[:, self.ignore] = 0

        # get the clean stats
        tp = conf.diagonal()
        fp = conf.sum(axis=1) - tp
        fn = conf.sum(axis=0) - tp
        return tp, fp, fn

    def getSemIoU(self):
        tp, fp, fn = self.getSemIoUStats()
        # print(f"tp={tp}")
        # print(f"fp={fp}")
        # print(f"fn={fn}")
        intersection = tp
        union = tp + fp + fn
        union = np.maximum(union, self.eps)
        iou = intersection.astype(np.double) / union.astype(np.double)
        iou_mean = (
            intersection[self.include].astype(np.double)
            / union[self.include].astype(np.double)
        ).mean()

        return iou_mean, iou  # returns "iou mean", "iou per class" ALL CLASSES

    def getSemAcc(self):
        tp, fp, fn = self.getSemIoUStats()
        total_tp = tp.sum()
        total = tp[self.include].sum() + fp[self.include].sum()
        total = np.maximum(total, self.eps)
        acc_mean = total_tp.astype(np.double) / total.astype(np.double)

        return acc_mean  # returns "acc mean"

    ################################# IoU STUFF ##################################
    ##############################################################################

    #############################  Panoptic STUFF ################################
    def addBatchPanoptic(self, x_sem_row, x_inst_row, y_sem_row, y_inst_row):
        # make sure instances are not zeros (it messes with my approach)
        x_inst_row = x_inst_row + 1
        y_inst_row = y_inst_row + 1

        # only interested in points that are outside the void area (not in excluded classes)

        if len(np.unique(x_sem_row)) > 2:
            raise ValueError("This is currently implemented only for one class")

        valid_gt = y_sem_row != self.ignore[0]
        valid_pred = np.zeros_like(x_inst_row, dtype=np.bool)

        pred_label_list = np.unique(x_inst_row)
        for id in pred_label_list[1:]:
            id_mask = x_inst_row == id
            id_gt_overlap = valid_gt[id_mask].sum() / id_mask.sum()
            if id_gt_overlap > 0.5:
                valid_pred[id_mask] = True
        x_sem_row = np.ones_like(x_sem_row, dtype=np.int)
        x_sem_row[~valid_pred] = 2
        valid_points = np.logical_or(valid_pred, valid_gt)
        x_inst_row = x_inst_row[valid_points]
        x_sem_row = x_sem_row[valid_points]
        # x_sem_row[~valid_pred] = 2

        y_sem_row = y_sem_row[valid_points]
        y_inst_row = y_inst_row[valid_points]
        # for cl in self.ignore:
        #   # make a mask for this class
        #   gt_not_in_excl_mask = y_sem_row != cl
        #   # remove all other points
        #   # x_sem_row = x_sem_row[gt_not_in_excl_mask]
        #   y_sem_row = y_sem_row[gt_not_in_excl_mask]
        #   # x_inst_row = x_inst_row[gt_not_in_excl_mask]
        #   y_inst_row = y_inst_row[gt_not_in_excl_mask]

        # first step is to count intersections > 0.5 IoU for each class (except the ignored ones)
        for cl in self.include:
            # print("*"*80)
            # print("CLASS", cl.item())
            # get a class mask
            x_inst_in_cl_mask = x_sem_row == cl
            y_inst_in_cl_mask = y_sem_row == cl

            # get instance points in class (makes outside stuff 0)
            x_inst_in_cl = x_inst_row * x_inst_in_cl_mask.astype(np.int64)
            y_inst_in_cl = y_inst_row * y_inst_in_cl_mask.astype(np.int64)

            # generate the areas for each unique instance prediction
            unique_pred, counts_pred = np.unique(
                x_inst_in_cl[x_inst_in_cl > 0], return_counts=True
            )
            id2idx_pred = {id: idx for idx, id in enumerate(unique_pred)}
            matched_pred = np.array([False] * unique_pred.shape[0])
            # print("Unique predictions:", unique_pred)

            # generate the areas for each unique instance gt_np
            unique_gt, counts_gt = np.unique(
                y_inst_in_cl[y_inst_in_cl > 0], return_counts=True
            )
            id2idx_gt = {id: idx for idx, id in enumerate(unique_gt)}
            matched_gt = np.array([False] * unique_gt.shape[0])
            # print("Unique ground truth:", unique_gt)

            # generate intersection using offset
            valid_combos = np.logical_and(x_inst_in_cl > 0, y_inst_in_cl > 0)
            offset_combo = (
                x_inst_in_cl[valid_combos] + self.offset * y_inst_in_cl[valid_combos]
            )
            unique_combo, counts_combo = np.unique(offset_combo, return_counts=True)

            # generate an intersection map
            # count the intersections with over 0.5 IoU as TP
            gt_labels = unique_combo // self.offset
            pred_labels = unique_combo % self.offset
            gt_areas = np.array([counts_gt[id2idx_gt[id]] for id in gt_labels])
            pred_areas = np.array([counts_pred[id2idx_pred[id]] for id in pred_labels])
            intersections = counts_combo
            unions = gt_areas + pred_areas - intersections
            ious = intersections.astype(np.float) / unions.astype(np.float)

            tp_indexes = ious > 0.5
            self.pan_tp[cl] += np.sum(tp_indexes)
            self.pan_iou[cl] += np.sum(ious[tp_indexes])

            matched_gt[[id2idx_gt[id] for id in gt_labels[tp_indexes]]] = True
            matched_pred[[id2idx_pred[id] for id in pred_labels[tp_indexes]]] = True

            # count the FN
            self.pan_fn[cl] += np.sum(
                np.logical_and(counts_gt >= self.min_points, matched_gt == False)
            )

            # count the FP
            self.pan_fp[cl] += np.sum(
                np.logical_and(counts_pred >= self.min_points, matched_pred == False)
            )

    def getPQ(self, return_pr_rc=False):
        # first calculate for all classes
        sq_all = self.pan_iou.astype(np.double) / np.maximum(
            self.pan_tp.astype(np.double), self.eps
        )
        rq_all = self.pan_tp.astype(np.double) / np.maximum(
            self.pan_tp.astype(np.double)
            + 0.5 * self.pan_fp.astype(np.double)
            + 0.5 * self.pan_fn.astype(np.double),
            self.eps,
        )
        pq_all = sq_all * rq_all

        pr_all = self.pan_tp.astype(np.double) / np.maximum(
            self.pan_tp.astype(np.double) + self.pan_fp.astype(np.double), self.eps
        )
        rc_all = self.pan_tp.astype(np.double) / np.maximum(
            self.pan_tp.astype(np.double) + self.pan_fn.astype(np.double), self.eps
        )

        # then do the REAL mean (no ignored classes)
        SQ = sq_all[self.include].mean()
        RQ = rq_all[self.include].mean()
        PQ = pq_all[self.include].mean()

        if return_pr_rc:
            return (
                pq_all[1],
                sq_all[1],
                rq_all[1],
                pr_all[1],
                rc_all[1],
            )  # PQ, SQ, RQ, pq_all, sq_all, rq_all
        else:
            return pq_all[1], sq_all[1], rq_all[1]

    #############################  Panoptic STUFF ################################
    ##############################################################################

    def addBatch(self, x_sem, x_inst, y_sem, y_inst):  # x=preds, y=targets
        """IMPORTANT: Inputs must be batched. Either [N,H,W], or [N, P]"""
        # add to IoU calculation (for checking purposes)
        self.addBatchSemIoU(x_sem, y_sem)

        # now do the panoptic stuff
        self.addBatchPanoptic(x_sem, x_inst, y_sem, y_inst)
