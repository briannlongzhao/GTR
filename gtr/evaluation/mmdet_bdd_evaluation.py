import os
import platform
import argparse
import numpy as np
from pathlib import Path
import json
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import re
from qdtrack.core.evaluation.mot import eval_mot

count = 0

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

def getboxcoord(det):
    x1 = det["box2d"]["x1"]
    x2 = det["box2d"]["x2"]
    y1 = det["box2d"]["y1"]
    y2 = det["box2d"]["y2"]
    return x1, y1, x2, y2

def parse_json(json_file):
    with open(json_file) as f:
        d = json.load(f)
    if type(d) is dict:  # pred
        return [item["labels"] for item in d["frames"]]
    elif type(d) is list:  # gt
        return [item["labels"] for item in d]
    else:
        raise NotImplementedError


def filter_by_class(dets, class_name):
    filtered_dets = []
    for det in dets:
        if det["category"] == class_name:
            filtered_dets.append(det)
    return filtered_dets

def parse_results_frame(dets):
    det_result = []
    for id, det in enumerate(dets):
        x1, y1, x2, y2 = getboxcoord(det)
        det_result.append([id, x1, y1, x2, y2, 1.0])
    if len(det_result) == 0:
        return np.zeros((0,6))
    return np.array(det_result)

def parse_annos_frame(dets):
    global count
    det_result = {}
    bboxes = np.zeros((len(dets),4), dtype=float)
    labels = np.zeros((len(dets),),dtype=int)
    instance_ids = np.zeros((len(dets),), dtype=int)
    for i, det in enumerate(dets):
        if det["category"] not in label2id.keys():
            count += 1
            continue
        x1, y1, x2, y2 = getboxcoord(det)
        bboxes[i] = [x1,y1,x2,y2]
        labels[i] = label2id[det["category"]]-1
        instance_ids[i] = det["id"]
    det_result["bboxes"] = bboxes
    det_result["labels"] = labels
    det_result["instance_ids"] = instance_ids
    det_result["bboxes_ignore"] = np.array([])
    return det_result

def parse_results_and_annos(out_dir, gt_dir):
    out_dir = Path(out_dir)
    gt_dir = Path(gt_dir)
    results = []
    annotations = []
    for video in tqdm(os.listdir(out_dir)):  # for each video
        video_result_pred = []
        video_result_gt = []
        video_id = video.replace(".json", '')
        pred = parse_json(out_dir / video)
        gt = parse_json(gt_dir / video)
        assert len(pred) == len(gt)

        for pred_frame, gt_frame in zip(pred, gt):  # for each frame
            frame_result_pred = []
            frame_result_gt = []
            for cat_id, cat_name in enumerate(label2id.keys()):  # for each category (pred only)
                assert id2lable[cat_id+1] == cat_name
                pred_dets = filter_by_class(pred_frame, cat_name)
                det_result_pred = parse_results_frame(pred_dets)
                frame_result_pred.append(det_result_pred)
            frame_result_gt = parse_annos_frame(gt_frame)
            video_result_gt.append(frame_result_gt)
            video_result_pred.append(frame_result_pred)
        results.append(video_result_pred)
        annotations.append(video_result_gt)
    return results, annotations

def eval_track_mmdet(outdir, gt_dir, debug=False):
    results, annotations = parse_results_and_annos(outdir, gt_dir)
    eval_results = eval_mot(results=results, annotations=annotations, classes=list(label2id.keys()))
    print(eval_results)


def main(args):
    out_dir = args.out_dir
    gt_dir = args.gt_dir
    eval_track_mmdet(out_dir, gt_dir, debug=args.debug)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--gt_dir", default=os.path.join(tmp_dir, "datasets/bdd/BDD100K/labels/box_track_20/val"))
    parser.add_argument("--out_dir", default="./output/GTR_BDD/GTR_BDD_DR2101_C2/inference_bdd100k_val/bddeval/val/pred/data/preds_bdd")
    parser.add_argument("--debug", default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    main(args)