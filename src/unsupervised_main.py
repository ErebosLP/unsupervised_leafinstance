import os
import yaml
import glob
import ipdb
import heapq
import open3d
import hdbscan
import numpy as np
import visualize_graph
from utils.graph import Graph
import matplotlib.pyplot as plt
from eval import compute_metrics
from matplotlib.patches import Ellipse
from voxel_hash_map.voxelize import voxel_down_sample


import time



def test_fit(pos):
    # U, v, d = np.linalg.svd(pos)
    # center = np.mean(pos, axis=0)
    # M = v[0]
    # m = v[1]
    # phi = np.arctan(d[0, 0] / d[1, 0])
    center = np.mean(pos, axis=0)
    cov = np.cov(pos[:, 0], pos[:, 1])
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)
    M = lambda_[0]
    m = lambda_[1]
    phi = np.arccos(v[0, 0])

    return [M, m, center[0], center[1], phi]


def fit_ellipse(pos, iter=3):
    min_dist_between_points = 0.2

    for i in range(iter):
        j = 1
        has_nan = True
        while has_nan:
            if i == 0:
                dist_to_center = np.argsort(
                    (pos[:, 0] - pos[:, 0].mean()) ** 2
                    + (pos[:, 1] - pos[:, 1].mean()) ** 2
                )
            else:
                if np.isnan([M, m, x0, y0, phi]).any() == False:
                    dist = (
                        np.cos(phi) * (pos[:, 0] - x0) + np.sin(phi) * (pos[:, 1] - y0)
                    ) ** 2 / (M) ** 2 + (
                        np.sin(phi) * (pos[:, 0] - x0) - np.cos(phi) * (pos[:, 1] - y0)
                    ) ** 2 / (
                        m
                    ) ** 2
                else:
                    dist = (pos[:, 0] - x0) ** 2 + (pos[:, 1] - y0) ** 2
                dist_to_center = np.argsort(dist)
            farest_point = dist_to_center[-j]
            while np.any(
                [
                    pos[np.argmin(pos[:, 0]), :2],
                    pos[np.argmin(pos[:, 1]), :2],
                    pos[np.argmax(pos[:, 0]), :2],
                    pos[np.argmax(pos[:, 1]), :2],
                ]
                == pos[farest_point, :2]
            ) or np.any(
                np.linalg.norm(
                    [
                        pos[np.argmin(pos[:, 0]), :2] - pos[farest_point, :2],
                        pos[np.argmin(pos[:, 1]), :2] - pos[farest_point, :2],
                        pos[np.argmax(pos[:, 0]), :2] - pos[farest_point, :2],
                        pos[np.argmax(pos[:, 1]), :2] - pos[farest_point, :2],
                    ],
                    axis=1,
                )
                < min_dist_between_points
            ):
                if j >= dist_to_center.shape[0]:
                    j = 0
                    min_dist_between_points -= 0.05
                else:
                    j += 1
                farest_point = dist_to_center[-j]
            ellipspoints = np.array(
                [
                    pos[np.argmin(pos[:, 0]), :2],
                    pos[np.argmin(pos[:, 1]), :2],
                    pos[np.argmax(pos[:, 0]), :2],
                    pos[np.argmax(pos[:, 1]), :2],
                    pos[farest_point, :2],
                ]
            )

            x, y = ellipspoints[:, 0][:, np.newaxis], ellipspoints[:, 1][:, np.newaxis]
            try:
                D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
                S, C = np.dot(D.T, D), np.zeros([6, 6])
                C[0, 2], C[2, 0], C[1, 1] = 2, 2, -1
                U, s, V = np.linalg.svd(np.dot(np.linalg.inv(S), C))
                a = U[:, 0]
                print(a.T @ C @ a)

                b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
                num = b * b - a * c
                x0 = (c * d - b * f) / num
                y0 = (a * f - b * d) / num
                up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
                down1 = (b * b - a * c) * (
                    (c - a) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a)
                )
                down2 = (b * b - a * c) * (
                    (a - c) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a)
                )
                m, M = np.sqrt(np.abs(up / down1)), np.sqrt(np.abs(up / down2))
                if m > M:
                    M, m = m, M

                phi = np.arctan2(2 * b, (a - c)) / 2
                phi -= 2 * np.pi * int(phi / (2 * np.pi))

                if np.isnan([M, m, x0, y0, phi]).any() == False:
                    has_nan = False

            except np.linalg.LinAlgError:
                print("SINGULAR MATRIX OCCURED!!")

                j += 1
                pass
    return (
        [M, m, x0, y0, phi],
        ellipspoints,
    )


def leaftip_estmation(pointcloud, fitted_ellipse, pcd_orig, graphpath, k):
    if graphpath != None:
        try:
            laplacian_matrix = np.load(graphpath)
            # visualize_graph.draw_graph(laplacian_matrix, pcd_orig, pointcloud)
        except FileNotFoundError:
            start_time = time.time()
            graph = Graph(pointcloud, 20)
            laplacian_matrix = graph.laplacian_matrix
            # visualize_graph.draw_graph(laplacian_matrix, pcd_orig, pointcloud)
            # np.save(graphpath, laplacian_matrix)
            print("Computingtime: %1d seconds" % (time.time() - start_time))
    else:
        graph = Graph(pointcloud, 5)
        laplacian_matrix = graph.laplacian_matrix
    offsets = np.zeros_like(pointcloud)
    leaf_id = np.zeros(pointcloud.shape[0], dtype=np.int32)
    elliptical_distance = (
        np.cos(fitted_ellipse[4]) * (pointcloud[:, 0] - fitted_ellipse[2])
        + np.sin(fitted_ellipse[4]) * (pointcloud[:, 1] - fitted_ellipse[3])
    ) ** 2 / (fitted_ellipse[0] * k) ** 2 + (
        np.sin(fitted_ellipse[4]) * (pointcloud[:, 0] - fitted_ellipse[2])
        - np.cos(fitted_ellipse[4]) * (pointcloud[:, 1] - fitted_ellipse[3])
    ) ** 2 / (
        fitted_ellipse[1] * k
    ) ** 2
    is_not_in_ellipse = elliptical_distance > 1
    points_to_cluster = pointcloud[is_not_in_ellipse, :]
    # print("Points to cluster: %2d" % points_to_cluster.shape[0])
    points_to_cluster_id = np.arange(pointcloud.shape[0])[is_not_in_ellipse]
    clustered_offsets = np.zeros_like(points_to_cluster)
    clustering = hdbscan.HDBSCAN(min_cluster_size=10)
    clustering_labels = clustering.fit_predict(points_to_cluster)
    leaf_id[is_not_in_ellipse] = clustering_labels
    for cluster_id in np.unique(clustering_labels):
        if cluster_id != -1:
            clustertip_id = np.argmax(
                elliptical_distance[is_not_in_ellipse][clustering_labels == cluster_id]
            )
            clustertip = points_to_cluster[clustering_labels == cluster_id][
                clustertip_id
            ]
            if cluster_id == 0:
                leaftips = np.expand_dims(clustertip, axis=0)
                leaftips_id = np.array(
                    [
                        points_to_cluster_id[clustering_labels == cluster_id][
                            clustertip_id
                        ]
                    ]
                )
                clustered_offsets[clustering_labels == cluster_id, :] = (
                    -points_to_cluster[clustering_labels == cluster_id] + clustertip
                )

            else:
                leaftips = np.concatenate(
                    (leaftips, np.expand_dims(clustertip, axis=0)), axis=0
                )
                leaftips_id = np.concatenate(
                    (
                        leaftips_id,
                        np.array(
                            [
                                points_to_cluster_id[clustering_labels == cluster_id][
                                    clustertip_id
                                ]
                            ]
                        ),
                    )
                )
                clustered_offsets[clustering_labels == cluster_id, :] = (
                    -points_to_cluster[clustering_labels == cluster_id] + clustertip
                )
    offsets[is_not_in_ellipse] = clustered_offsets

    # graphdistance
    count_euc = 0
    count_point = 0
    for point in range(pointcloud.shape[0]):
        if np.all(offsets[point] == 0):
            closest_tip = np.argmin(laplacian_matrix[point, leaftips_id])
            closest_clusterd_point = leaftips_id[closest_tip]
            if laplacian_matrix[point, leaftips_id[closest_tip]] == np.inf:
                # closest_clusterd_point = np.argmin(
                #     laplacian_matrix[point, points_to_cluster_id]
                # )
                # closest_clusterd_point = points_to_cluster_id[closest_clusterd_point]
                # leaf_id[point] = leaf_id[closest_clusterd_point]
                # offsets[point] = -pointcloud[point, :] + leaftips[leaf_id[point]]
                count_point += 1
                if laplacian_matrix[point, closest_clusterd_point] == np.inf:
                    closest_tip = np.argmin(
                        np.linalg.norm(-pointcloud[point, :] + leaftips, axis=1)
                    )
                    count_point -= 1
                    count_euc += 1
                    leaf_id[point] = closest_tip
                    offsets[point] = -pointcloud[point, :] + leaftips[leaf_id[point]]
            else:
                leaf_id[point] = leaf_id[closest_clusterd_point]
                offsets[point] = -pointcloud[point, :] + leaftips[leaf_id[point]]

    for point in range(pointcloud.shape[0]):
        nearst_points = heapq.nsmallest(
            20,
            range(laplacian_matrix.shape[0]),
            key=laplacian_matrix[:, point].__getitem__,
        )
        nearest_class, nearest_class_counter = np.unique(
            leaf_id[nearst_points], return_counts=True
        )
        # if leaf_id[point] != 0 or nearest_class.shape[0] > 1:
        #     ipdb.set_trace()
        leaf_id[point] = nearest_class[np.argmax(nearest_class_counter)]

        offsets[point] = -pointcloud[point, :] + leaftips[leaf_id[point]]
    # print("%1d Points used the closest initially cluserd point" % count_point)
    # print("%1d Points used the euclidean disatance" % count_euc)
    return leaftips, leaf_id, offsets


def main():
    plant_count = 0
    # cfg = yaml.safe_load(open("./../config/config.yaml"))

    plant_paths = glob.glob("./../SBUB3D/test/plant*")
    plant_paths = glob.glob("./../SBUB3D/test_unsup/patch*")
    N = 10000
    graph_path = "./graph_gm_20_cuda/test_unsup/" + str(N) + "/"
    compute_metrics(None, graph_path, None)
    
    k = 1.6
    for path in plant_paths:
        np.random.seed(42)
        plant = path.split("/")[-1].split(".")[0]
        print("Processing " + plant)

        if os.path.exists(graph_path) == False:
            os.makedirs(graph_path)

        voxelisation_dist = 0.003

        pcd = open3d.t.io.read_point_cloud(path)

        plant_idx = (
            (pcd.point.plant_ids == np.unique(pcd.point.plant_ids.numpy())[1])
            .numpy()
            .squeeze(1)
        )
        pcd.point.positions = pcd.point.positions[plant_idx, :]
        pcd.point.colors = pcd.point.colors[plant_idx, :]
        pcd.point.leaf_ids = pcd.point.leaf_ids[plant_idx, :]

        cloud = pcd.voxel_down_sample(voxelisation_dist)
        cloud.remove_statistical_outliers(nb_neighbors=20, std_ratio=2.0)
        idx = np.zeros(
            [cloud.point["positions"].numpy().shape[0]],
            bool,
        )

        idx[0:N] = 1
        idx = np.random.permutation(idx)
        pos = cloud.point["positions"][idx].numpy()

        # fitted_ellipse, ellipspoints = fit_ellipse(pos)
        fitted_ellipse = test_fit(pos)

        # Leaftip estimation

        leaftips, leaf_id, offsets = leaftip_estmation(
            pos, fitted_ellipse, cloud, graph_path + plant + ".npy", k
        )

        leaf_id[leaf_id >= 0] += 1
        leaf_number, leaf_count = np.unique(leaf_id, return_counts=True)
        p = 2.5
        fishy_leafs = np.any(
            [
                leaf_count < leaf_count.mean() - 1.5 * leaf_count.std(),
                leaf_count > leaf_count.mean() + 1.5 * leaf_count.std(),
            ],
            axis=0,
        )
        for fishy_leaf_id in leaf_number[fishy_leafs]:
            leaf_id[leaf_id == fishy_leaf_id] = -1
        # print("Number of leafs: %2d" % leaf_number.shape[0])
        # print("Number of points per leaf:", str(leaf_count))
        processed_pcd = open3d.t.geometry.PointCloud()

        processed_pcd.point["positions"] = pos
        processed_pcd.point["gt_leaf_ids"] = cloud.point.leaf_ids[idx]
        processed_pcd.point["pred_leaf_ids"] = np.expand_dims(leaf_id, axis=1)
        open3d.t.io.write_point_cloud(graph_path + plant + ".ply", processed_pcd)

        # good_leafs = leaf_id != -1
        # cmap = plt.cm.get_cmap("tab20")
        # fig3 = plt.figure()
        # ax3 = fig3.add_subplot(projection="3d")
        # ax3.scatter(
        #     pos[good_leafs, 0],
        #     pos[good_leafs, 1],
        #     pos[good_leafs, 2],
        #     c=cmap(leaf_id[good_leafs])[..., :3],
        #     s=10,
        # )
        # ax3.scatter(
        #     leaftips[:, 0],
        #     leaftips[:, 1],
        #     leaftips[:, 2],
        #     color=[0, 0, 0],
        #     s=40,
        # )
        # plt.show()
        # ipdb.set_trace()
        # 2D-Plot
        # fig = plt.figure()
        # ax = fig.add_subplot()
        # ax.axis("equal")
        # ax.scatter(
        #     pos[:, 0],
        #     pos[:, 1],
        #     c=pcd.point.colors[idx].numpy() / 256,
        #     s=10,
        # )

        # ax.scatter(
        #     fitted_ellipse[2],
        #     fitted_ellipse[3],
        #     color=[0, 0, 1],
        #     s=10,
        # )
        # test_ell_3 = Ellipse(
        #     (fitted_ellipse[2], fitted_ellipse[3]),
        #     width=fitted_ellipse[0] * k * 2,
        #     height=fitted_ellipse[1] * k * 2,
        #     angle=fitted_ellipse[4] * 180 / np.pi,
        # )
        # test_ell_3.set_alpha(0.3)
        # ax.add_patch(test_ell_3)
        # plt.show()
        # # 3D-Plot
        # elliptical_distance = (
        #     np.cos(fitted_ellipse[4]) * (pos[:, 0] - fitted_ellipse[2])
        #     + np.sin(fitted_ellipse[4]) * (pos[:, 1] - fitted_ellipse[3])
        # ) ** 2 / (fitted_ellipse[0] * k) ** 2 + (
        #     np.sin(fitted_ellipse[4]) * (pos[:, 0] - fitted_ellipse[2])
        #     - np.cos(fitted_ellipse[4]) * (pos[:, 1] - fitted_ellipse[3])
        # ) ** 2 / (
        #     fitted_ellipse[1] * k
        # ) ** 2
        # is_not_in_ellipse = elliptical_distance > 1
        # fig2 = plt.figure()
        # ax2 = fig2.add_subplot(projection="3d")
        # ax2.scatter(
        #     pos[is_not_in_ellipse, 0],
        #     pos[is_not_in_ellipse, 1],
        #     pos[is_not_in_ellipse, 2],
        #     c=cmap(leaf_id[is_not_in_ellipse])[..., :3],
        #     s=10,
        # )
        # plt.show()
        # plant_count += 1
        # if plant_count % 10 == 0:
        #     compute_metrics(None, graph_path, None)
    # compute_metrics(None, graph_path, None)


if __name__ == "__main__":
    main()
