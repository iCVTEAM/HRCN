# credits: https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/metrics/rank.py

import numpy as np
import tqdm


def CGM(matches, g_pids_order, q_pid, g_camids_order, q_camid):
    # remove gallery images with the same id and camera with the query
    match_q_pids = (g_pids_order == q_pid) & (g_camids_order != q_camid)

    cam_list = list(set(g_camids_order[match_q_pids]))

    if len(cam_list) == 0:
        return -1.0

    cam_cgm_list = []
    # compute CGM of each camera
    for cam in cam_list:
        keep = (g_pids_order != q_pid) | (g_camids_order == cam)
        cam_cmc = matches[keep]
        invert_cam_cmc = np.invert(matches - 2)[keep]

        num_rel = cam_cmc.sum()
        invert_tmp_cmc = invert_cam_cmc.cumsum()
        tmp_score = [1 / (x + 1) for i, x in enumerate(invert_tmp_cmc)]

        val_tmp_score = np.asarray(tmp_score) * cam_cmc
        cam_cgm = val_tmp_score.sum() / num_rel

        cam_cgm_list.append(cam_cgm)

    return np.mean(cam_cgm_list)


def eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))

    indices = np.argsort(distmat, axis=1)

    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_CGM = []
    num_valid_q = 0.  # number of valid query

    for q_idx in tqdm.tqdm(range(num_q)):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute CGM
        q_cgm = CGM(matches[q_idx], g_pids[order], q_pid, g_camids[order], q_camid)

        if q_cgm != -1.0:
            all_CGM.append(q_cgm)

        # compute cmc curve
        raw_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q

    return all_cmc, all_AP, all_CGM


def evaluate_rank(
        distmat,
        q_pids,
        g_pids,
        q_camids,
        g_camids,
        max_rank=50,
):
    """Evaluates CMC rank.
    Args:
        distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
        q_pids (numpy.ndarray): 1-D array containing person identities
            of each query instance.
        g_pids (numpy.ndarray): 1-D array containing person identities
            of each gallery instance.
        q_camids (numpy.ndarray): 1-D array containing camera views under
            which each query instance is captured.
        g_camids (numpy.ndarray): 1-D array containing camera views under
            which each gallery instance is captured.
        max_rank (int, optional): maximum CMC rank to be computed. Default is 50.
    """
    return eval_market1501(
        distmat, q_pids, g_pids, q_camids, g_camids, max_rank
    )
