import numpy as np
import open3d as o3d
import torch



def compute_graph_distances_fast(laplacian: np.ndarray):
    laplacian_t = torch.tensor(laplacian).cuda()
    for r in range(laplacian_t.shape[0]):
        current_row = laplacian_t[r, :]
        extended = current_row.reshape(
            (1, -1),
        ).repeat(laplacian_t.shape[0], 1)
        new_dist = laplacian_t + extended
        new_row_dist = new_dist.min(dim=1)[0]
        laplacian_t[r, :] = laplacian_t[:, r] = new_row_dist
    return laplacian_t.cpu().numpy()

