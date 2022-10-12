import os
import platform
import argparse
import numpy as np
from pathlib import Path
import json
from scipy.optimize import linear_sum_assignment

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
    assert ax1 < ax2 and ay1 < ay2 and bx1 < bx2 and by1 < by2
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

def iou_dist_matrix(gt_ids, pred_ids, gt, pred):
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
    cost_matrix = iou_dist_matrix(gt_ids, pred_ids, gt, pred)
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

def dist_frame(gt2pred, gt, pred):
    dist = 0
    for k, v in gt2pred.items():
        gtbox = getbox(k, gt)
        predbox = getbox(v, pred)
        dist += iou(gtbox, predbox)
    return dist


def eval_video(pred, gt):
    assert len(pred) == len(gt), "number of frames does not match"
    gt2pred = {}
    dist, idsw, fp, fn = 0, 0, 0, 0
    gt_count = sum([len(item) for item in gt])
    match_count = 0
    for pred_frame, gt_frame in zip(pred, gt):  # for each frame
        new_gt2pred = match_frame(pred_frame, gt_frame, gt2pred)
        idsw += idsw_frame(gt2pred, new_gt2pred)
        dist += dist_frame(new_gt2pred, gt_frame, pred_frame)
        fp += len(pred_frame)-len(new_gt2pred)
        fn += len(gt_frame)-len(new_gt2pred)
        match_count += len(new_gt2pred)
        gt2pred = new_gt2pred
    result = {}
    result["MOTP"] = dist/match_count
    result["MOTA"] = (1-(fn+fp+idsw))/gt_count
    result["FP"] = fp
    result["FN"] = fn
    result["IDSW"] = idsw
    return result


def eval_track_custom(out_dir, gt_dir):
    out_dir = Path(out_dir)
    gt_dir = Path(gt_dir)
    result = {}
    for video in os.listdir(out_dir):
        # pred and gt are list of frames
        pred = parse_json(out_dir/video)
        gt = parse_json(gt_dir/video)
        if isvalid(pred):
            result[video] = eval_video(pred, gt)
        else:
            result[video] = "no prediction"


    print(result)
    # print_result_per_video(result)
    # print_result_mean(result)
    # print_result_overall(result)





def main(args):
    out_dir = args.out_dir
    gt_dir = args.gt_dir
    eval_track_custom(out_dir, gt_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--gt_dir", default=os.path.join(tmp_dir, "datasets/bdd/BDD100K/labels/box_track_20/val"))
    parser.add_argument("--out_dir", default="./output/GTR_BDD/GTR_BDD_DR2101_C2/inference_bdd100k_val/bddeval/val/pred/data/preds_bdd")
    parser.add_argument("--debug", default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    main(args)