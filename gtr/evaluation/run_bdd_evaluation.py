# import itertools
# import json
# import numpy as np
import os
import platform
import re
import sys
import argparse

sys.path.insert(0, "../../tools")
from bdd100k.eval.run import run as eval_track_scalabel
from custom_bdd_evaluation import eval_track_custom
from mmdet_bdd_evaluation import eval_track_mmdet

tmp_dir = ''
hostname = platform.node()
if "iGpu" in hostname or "iLab" in hostname:
    os.environ["TMPDIR"] = "/lab/tmpig8e/u/brian-data"
elif re.search("[a-z]\d\d-\d\d", hostname):
    os.environ["TMPDIR"] = "/scratch1/briannlz"
if "TMPDIR" in os.environ.keys():
    tmp_dir = os.path.join(os.environ["TMPDIR"], "GTR/", '')

def main(args):
    out_dir = args.out_dir
    gt_dir = args.gt_dir

    if args.eval_method == "custom":
        return eval_track_custom(out_dir, gt_dir, filter_class=args.filter_class)
    elif args.eval_method == "mmdet":
        return eval_track_mmdet(out_dir, gt_dir)
    elif args.eval_method == "scalabel":
        args = [
            "--task", "box_track",
            "--gt", gt_dir,
            "--result", out_dir
        ]
        return eval_track_scalabel(args)
    elif args.eval_method == "gtr_custom":
        pass
    elif args.eval_method is None:
        eval_track_mmdet(out_dir, gt_dir)
        args = [
            "--task", "box_track",
            "--gt", gt_dir,
            "--result", out_dir
        ]
        return eval_track_scalabel(args)
    else:
        raise NotImplementedError

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--gt_dir", default=os.path.join(tmp_dir, "datasets/bdd/BDD100K/labels/box_track_20/val"))
    parser.add_argument("--out_dir", default="../../output/GTR_BDD/GTR_BDD_DR2101_C2/inference_bdd100k_val/bddeval/val/pred/data/updated_preds_bdd")
    parser.add_argument("--debug", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--eval_method", choices=["scalabel", "mmdet", "gtr_custom", "custom"], default=None)
    parser.add_argument("--filter_class", default=None)
    args = parser.parse_args()
    main(args)