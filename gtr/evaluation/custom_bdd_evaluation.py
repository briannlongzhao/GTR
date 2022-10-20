import os
import platform
import argparse
import numpy as np
from pathlib import Path
import json
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import re

iou_th = 0.5

id2lable = {1: "pedestrian", 2: "rider", 3: "car", 4: "truck", 5: "bus", 6: "train", 7: "motorcycle", 8: "bicycle"}
label2id = {"pedestrian": 1, "rider": 2, "car": 3, "truck": 4, "bus": 5, "train": 6, "motorcycle": 7, "bicycle": 8}
tmp_dir = ''
hostname = platform.node()
if "iGpu" in hostname or "iLab" in hostname:
    os.environ["TMPDIR"] = "/lab/tmpig8e/u/brian-data"
elif re.search("[a-z]\d\d-\d\d", hostname):
    os.environ["TMPDIR"] = "/scratch1/briannlz"
if "TMPDIR" in os.environ.keys():
    tmp_dir = os.path.join(os.environ["TMPDIR"], "GTR/", '')

class Trajectory:
    frames: list  # list of frame ids where this track present
    bboxes: list  # list of bbox dicts corresponding to frame ids

    def __init__(self):
        self.frames = []
        self.bboxes = []

    def update(self, det, frame_id):
        self.frames.append(frame_id)
        self.bboxes.append(det["box2d"])

    def getbox(self, frame_id) -> dict:
        assert len(self.frames) == len(self.bboxes)
        idx = self.frames.index(frame_id)
        return self.bboxes[idx]

    def time_overlap(self, traj) -> list:
        return list(set(self.frames) & set(traj.frames))

def parse_json(json_file):
    with open(json_file) as f:
        d = json.load(f)
    if type(d) is dict:  # pred
        return [item["labels"] for item in d["frames"]]
    elif type(d) is list:  # gt
        return [item["labels"] for item in d]
    else:
        raise NotImplementedError

def isvalid(preds):
    for item in preds:
        if len(item):
            return True
    return False

def getbox(id, dets):
    for item in dets:
        if item["id"] == id:
            return item["box2d"]
    return None

def pred_matched(pred_id, gt2pred):
    for k, v in gt2pred.items():
        if v == pred_id:
            return True
    return False

def gt_matched(gt_id, gt2pred):
    return gt_id in gt2pred.keys()

def iou(boxa, boxb):
    ax1, ax2, ay1, ay2 = boxa["x1"], boxa["x2"], boxa["y1"], boxa["y2"]
    bx1, bx2, by1, by2 = boxb["x1"], boxb["x2"], boxb["y1"], boxb["y2"]
    assert ax1 <= ax2 and ay1 <= ay2 and bx1 <= bx2 and by1 <= by2, print(ax1,ax2,ay1,ay2, bx1, bx2, by1, by2)
    ix1 = max(ax1, bx1)
    ix2 = min(ax2, bx2)
    iy1 = max(ay1, by1)
    iy2 = min(ay2, by2)
    if ix1 > ix2 or iy1 > iy2:  # No intersection
        return 0.0
    i = (ix2-ix1)*(iy2-iy1)
    u = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-i
    assert 0 < i/u < 1
    return i/u

def iou_cost_matrix(gt_ids, pred_ids, gt, pred):
    cost_matrix = np.ones((len(gt_ids), len(pred_ids)))
    for i in range(len(gt_ids)):
        for j in range(len(pred_ids)):
            gtbox = getbox(gt_ids[i], gt)
            predbox = getbox(pred_ids[j], pred)
            cost_matrix[i][j] = 1-iou(gtbox, predbox)
    return cost_matrix

def match_frame(pred, gt, gt2pred):
    """
    Match pred and gt within the same frame, return new gt2pred matching
    """
    new_gt2pred = {}
    gt_ids = [item["id"] for item in gt]
    pred_ids = [item["id"] for item in pred]
    # 1.check if last matching still valid
    for k, v in gt2pred.items():
        if k in gt_ids and v in pred_ids:  # last gt and pred still exist
            gtbox = getbox(k, gt)
            predbox = getbox(v, pred)
            if iou(gtbox, predbox) > iou_th:  # iou still valid
                new_gt2pred[k] = v  # add matching to new
                gt_ids.remove(k)
                pred_ids.remove(v)
    # 2. minimize distance matching for remaining gt
    cost_matrix = iou_cost_matrix(gt_ids, pred_ids, gt, pred)
    row_ids, col_ids = linear_sum_assignment(cost_matrix)
    gt_ids = [gt_ids[i] for i in row_ids]
    pred_ids = [pred_ids[i] for i in col_ids]
    assert len(gt_ids) == len(pred_ids)
    for k, v in zip(gt_ids, pred_ids):
        new_gt2pred[k] = v
    return new_gt2pred

def idsw_frame(gt2pred, new_gt2pred):
    count = 0
    for k, v in gt2pred.items():
        if k in new_gt2pred.keys() and new_gt2pred[k] != v:
            count += 1
    return count

def frag_frame(gt2pred, new_gt2pred, gt):
    count = 0
    gt_ids = [item["id"] for item in gt]
    for k in gt2pred.keys():
        if k not in new_gt2pred.keys() and k in gt_ids:
            count += 1
    return count

def dist_frame(gt2pred, gt, pred):
    dist = 0
    for k, v in gt2pred.items():
        gtbox = getbox(k, gt)
        predbox = getbox(v, pred)
        dist += iou(gtbox, predbox)
    return dist

def eval_video_clear(pred, gt) -> dict:
    assert len(pred) == len(gt), "number of frames does not match"
    gt2pred = {}
    dist, idsw, frag, fp, fn = 0, 0, 0, 0, 0
    gt_count = sum([len(item) for item in gt])
    match_count = 0
    for pred_frame, gt_frame in zip(pred, gt):  # for each frame
        new_gt2pred = match_frame(pred_frame, gt_frame, gt2pred)
        idsw += idsw_frame(gt2pred, new_gt2pred)
        frag += frag_frame(gt2pred, new_gt2pred, gt_frame)
        dist += dist_frame(new_gt2pred, gt_frame, pred_frame)
        fp += len(pred_frame)-len(new_gt2pred)
        fn += len(gt_frame)-len(new_gt2pred)
        match_count += len(new_gt2pred)
        gt2pred = new_gt2pred
    result = {}
    result["MOTP"] = dist/match_count if match_count != 0 else np.nan
    result["MOTA"] = 1-(fn+fp+idsw)/gt_count if gt_count != 0 else np.nan
    result["FP"] = fp
    result["FN"] = fn
    result["TP"] = match_count
    result["GT_DET"] = gt_count
    result["F1"] = 2*match_count/(2*match_count+fn+fp) if (2*match_count+fn+fp) != 0 else np.nan
    result["IDSW"] = idsw
    result["FRAG"] = frag
    return result

def update_trajs(frame, trajs, frame_id):
    for det in frame:
        if det["id"] not in trajs.keys():
            trajs[det["id"]] = Trajectory()
        trajs[det["id"]].update(det, frame_id)
    return trajs

def id_cost(gt_traj, pred_traj):
    cost = 0
    for t in gt_traj.frames:
        if t not in pred_traj.frames:
            cost += 1
        elif iou(gt_traj.getbox(t), pred_traj.getbox(t)) < iou_th:
            cost += 1
        else:
            pass
    for t in pred_traj.frames:
        if t not in gt_traj.frames:
            cost += 1
        elif iou(gt_traj.getbox(t), pred_traj.getbox(t)) < iou_th:
            cost += 1
        else:
            pass
    return cost

def id_cost_matrix(gt_ids, pred_ids, gt_trajs, pred_trajs):
    n_gt_trajs, n_pred_trajs = len(gt_ids), len(pred_ids)
    cm_dim = n_pred_trajs+n_gt_trajs
    cost_matrix = np.full((cm_dim, cm_dim), np.inf)
    for i in range(cm_dim):
        for j in range(cm_dim):
            if i < n_gt_trajs and j < n_pred_trajs:  # Edge between two regular nodes
                if len(gt_trajs[gt_ids[i]].time_overlap(pred_trajs[pred_ids[j]])) > 0:  # Check overlap in time
                    cost_matrix[i][j] = id_cost(gt_trajs[gt_ids[i]], pred_trajs[pred_ids[j]])
            elif i < n_gt_trajs and j == n_pred_trajs+i:  # Edge between one regular gt and its corresponding fn
                cost_matrix[i][j] = id_cost(gt_trajs[gt_ids[i]], Trajectory())
            elif j < n_pred_trajs and i == n_gt_trajs+j:  # Edge between one regular pred and its corresponding fp
                cost_matrix[i][j] = id_cost(Trajectory(), pred_trajs[pred_ids[j]])
            else:  # Edge between two irreular nodes (zero cost)
                cost_matrix[i][j] = 0
    return cost_matrix

def match_trajs(gt_trajs, pred_trajs):
    gt2pred = {}
    pred2gt = {}
    gt_ids = sorted(gt_trajs.keys())
    pred_ids = sorted(pred_trajs.keys())
    cost_matrix = id_cost_matrix(gt_ids, pred_ids, gt_trajs, pred_trajs)
    row_ids, col_ids = linear_sum_assignment(cost_matrix)
    for r, c in zip(row_ids, col_ids):
        if r < len(gt_ids) and c < len(pred_ids):  # tp
            gt2pred[gt_ids[r]] = pred_ids[c]
            pred2gt[pred_ids[c]] = gt_ids[r]
        elif r < len(gt_ids) and c >= len(pred_ids):  # fn
            gt2pred[gt_ids[r]] = "fn"
        elif r >= len(gt_ids) and c < len(pred_ids):  # fp
            pred2gt[pred_ids[c]]  = "fp"
        else:  # tn
            pass
    return gt2pred, pred2gt

def eval_video_id(pred, gt) -> dict:
    assert len(pred) == len(gt), "number of frames does not match"
    gt_trajs, pred_trajs = {}, {}
    # Extract gt and pred trajectories
    for frame_id, (pred_frame, gt_frame) in enumerate(zip(pred, gt)):  # for each frame
        gt_trajs = update_trajs(gt_frame, gt_trajs, frame_id)
        pred_trajs = update_trajs(pred_frame, pred_trajs, frame_id)
    # Truth-to-result matching
    gt2pred, pred2gt = match_trajs(gt_trajs, pred_trajs)
    idtp, idfp, idfn = 0, 0, 0
    for traj_id, gt_traj in gt_trajs.items():
        matched_pred = pred_trajs[gt2pred[traj_id]] if gt2pred[traj_id] != "fn" else Trajectory()
        for t in gt_traj.frames:
            if t not in matched_pred.frames:
                idfn += 1
            elif iou(gt_traj.getbox(t), matched_pred.getbox(t)) < iou_th:
                idfn += 1
            else:
                pass
    for traj_id, pred_traj in pred_trajs.items():
        matched_gt = gt_trajs[pred2gt[traj_id]] if pred2gt[traj_id] != "fp" else Trajectory()
        for t in pred_traj.frames:
            if t not in matched_gt.frames:
                idfp += 1
            elif iou(matched_gt.getbox(t), pred_traj.getbox(t)) < iou_th:
                idfp += 1
            else:
                pass
    idtp = sum([len(gt_traj.frames) for _, gt_traj in gt_trajs.items()])-idfn
    assert idtp == sum([len(pred_traj.frames) for _, pred_traj in pred_trajs.items()])-idfp
    result = {}
    result["GT_TRAJ"] = len(gt_trajs)
    result["IDTP"] = idtp
    result["IDFN"] = idfn
    result["IDFP"] = idfp
    result["IDF1"] = 2*idtp/(2*idtp+idfn+idfp) if (2*idtp+idfn+idfp) != 0 else np.nan
    result["IDP"] = idtp/(idtp+idfp) if (idtp+idfp) != 0 else np.nan
    result["IDR"] = idtp/(idtp+idfn) if (idtp+idfn) != 0 else np.nan
    return result

def accumulate(result_all):
    result_list = list(result_all.values())
    result = {}
    result["MOTP"] = np.mean([item["MOTP"] for item in result_list])
    result["TP"] = sum([item["TP"] for item in result_list])
    result["FP"] = sum([item["FP"] for item in result_list])
    result["FN"] = sum([item["FN"] for item in result_list])
    result["GT_TRAJ"] = sum([item["GT_TRAJ"] for item in result_list])
    result["GT_DET"] = sum([item["GT_DET"] for item in result_list])
    result["IDSW"] = sum([item["IDSW"] for item in result_list])
    result["FRAG"] = sum([item["FRAG"] for item in result_list])
    result["IDTP"] = sum([item["IDTP"] for item in result_list])
    result["IDFP"] = sum([item["IDFP"] for item in result_list])
    result["IDFN"] = sum([item["IDFN"] for item in result_list])
    result["F1"] = 2*result["TP"]/(2*result["TP"]+result["FP"]+result["FN"]) if (2*result["TP"]+result["FP"]+result["FN"]) != 0 else np.nan
    result["MOTA"] = 1-(result["FP"]+result["FN"]+result["IDSW"])/result["GT_DET"] if result["GT_DET"] != 0 else np.nan
    result["IDP"] = result["IDTP"]/(result["IDTP"]+result["IDFP"]) if (result["IDTP"]+result["IDFP"]) != 0 else np.nan
    result["IDR"] = result["IDTP"]/(result["IDTP"]+result["IDFN"]) if (result["IDTP"]+result["IDFN"]) != 0 else np.nan
    result["IDF1"] = 2*result["IDTP"]/(2*result["IDTP"]+result["IDFP"]+result["IDFN"]) if (2*result["IDTP"]+result["IDFP"]+result["IDFN"]) != 0 else np.nan
    return result

def filter_by_class(frames, class_name):
    filtered_frames = []
    for frame in frames:
        filtered_frame = []
        for det in frame:
            if det["category"] == class_name:
                filtered_frame.append(det)
        filtered_frames.append(filtered_frame)
    return filtered_frames

def eval_track_custom(out_dir, gt_dir, filter_class=None):
    out_dir = Path(out_dir)
    gt_dir = Path(gt_dir)
    result = {}
    for video in tqdm(os.listdir(out_dir)):
        if video == "b1d9e136-6c94ea3f.json": #debug
            pass
        # pred and gt are list of frames
        pred = parse_json(out_dir/video)
        gt = parse_json(gt_dir/video)
        for frame in gt: #debug
            for det in frame:
                if det["category"] == "train":
                    pass
        if filter_class is not None:
            pred = filter_by_class(pred, filter_class)
            gt = filter_by_class(gt, filter_class)
        # if isvalid(pred):
        result[video] = eval_video_clear(pred, gt)
        result[video].update(eval_video_id(pred, gt))
        # else:
        #     result[video] = "no prediction"
    #print(result)
    result = accumulate(result)
    print("OVERALL:\n", json.dumps(result, indent=2))
    # print_result_per_video(result)
    # print_result_mean(result)
    # print_result_overall(result)

def main(args):
    out_dir = args.out_dir
    gt_dir = args.gt_dir
    eval_track_custom(out_dir, gt_dir, filter_class=args.filter_class)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--gt_dir", default=os.path.join(tmp_dir, "datasets/bdd/BDD100K/labels/box_track_20/val"))
    parser.add_argument("--out_dir", default="./output/GTR_BDD/GTR_BDD_DR2101_C2/inference_bdd100k_val/bddeval/val/pred/data/preds_bdd")
    parser.add_argument("--debug", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--filter_class", default=None)
    args = parser.parse_args()
    main(args)