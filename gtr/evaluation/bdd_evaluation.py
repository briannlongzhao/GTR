import itertools
import json
import numpy as np
import os
import platform
import re
import sys
import argparse
import wandb
from collections import defaultdict
from multiprocessing import freeze_support
from pathlib import Path
import pycocotools.mask as mask_util
from detectron2.structures import BoxMode
from fvcore.common.file_io import PathManager
from detectron2.evaluation.coco_evaluation import COCOEvaluator, _evaluate_predictions_on_coco
from detectron2.utils import comm
from ..tracking.naive_tracker import track
from ..tracking import trackeval
from gtr.predictor import VisualizationDemo
from scalabel.label import from_coco
from wandb_writer import WandbWriter
from bdd100k.eval import run as bdd_eval
from .custom_bdd_evaluation import eval_track_custom

tmp_dir = ''
hostname = platform.node()
if "iGpu" in hostname or "iLab" in hostname:
    os.environ["TMPDIR"] = "/lab/tmpig8e/u/brian-data"
elif re.search("[a-z]\d\d-\d\d", hostname):
    os.environ["TMPDIR"] = "/scratch1/briannlz"
if "TMPDIR" in os.environ.keys():
    tmp_dir = os.path.join(os.environ["TMPDIR"], "GTR/", '')


def eval_track(out_dir, dataset_name, custom=False):
    freeze_support()
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config['DISPLAY_LESS_PROGRESS'] = True
    default_dataset_config = trackeval.datasets.BDD100K.get_default_dataset_config()
    default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity']}
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs
    config['SPLIT_TO_EVAL'] = "val" if "val" in dataset_name else "train"
    config['GT_FOLDER'] = os.path.join(tmp_dir, 'datasets/bdd/BDD100K/labels/box_track_20', config['SPLIT_TO_EVAL'])
    config['TRACKERS_FOLDER'] = os.path.join(out_dir,"bddeval",config["SPLIT_TO_EVAL"],"pred/data/preds_bdd")
    config["OUTPUT_FOLDER"] = os.path.join(out_dir,"bddeval",config['SPLIT_TO_EVAL'],"eval_results.json")
    '''eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}
    print('config', config)
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.BDD100K(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric())'''
    # args = argparse.Namespace()
    # args.config = None
    # args.task = "box_track"
    # args.gt = config["GT_FOLDER"]
    # args.result = config["TRACKERS_FOLDER"]
    # args.out_file = config["OUTPUT_FOLDER"]
    # args.nproc = 4
    if custom:
        return eval_track_custom(config["TRACKERS_FOLDER"], config["GT_FOLDER"])
    else:
        args = [
            "--task", "box_track",
            "--gt", config["GT_FOLDER"],
            "--result", config["TRACKERS_FOLDER"],
            "--out-file", config["OUTPUT_FOLDER"]
        ]
        return bdd_eval.run(args)


def save_cocojson(json_path, videos, images, categories, preds):
    coco_json = {}
    coco_json["videos"] = videos
    coco_json["images"] = images
    coco_json["categories"] = categories
    annotations = [{**item, **{"instance_id": item["track_id"]}} for item in preds]
    coco_json["annotations"] = annotations
    Path(json_path).parent.mkdir(exist_ok=True, parents=True)
    with open(json_path, 'w') as f:
        json.dump(coco_json, f)


def convert_coco_to_bdd(coco_path, bdd_dir):
    args = argparse.Namespace()
    args.input = coco_path
    args.output = bdd_dir
    args.nproc = 4
    from_coco.run(args)


def track_and_eval_bdd(out_dir, data, preds, dataset_name, custom=False):
    videos = sorted(data['videos'], key=lambda x: x['id'])
    images = sorted(data['images'], key=lambda x: x['id'])
    categories = sorted(data['categories'], key=lambda x: x['id'])
    video2images = defaultdict(list)
    for image in images:
        video2images[image['video_id']].append(image)
    for video in video2images:
        video2images[video] = sorted(video2images[video], key=lambda x: x['frame_id'])
    per_image_preds = defaultdict(list)
    for x in preds:
        if x['score'] > 0.4:
            per_image_preds[x['image_id']].append(x)
    has_track_id = len(preds) > 0 and 'track_id' in preds[0]
    split = "val"
    bdd_out_dir = out_dir + '/bddeval/{}/pred/data/'.format(split)
    if not has_track_id:
        print('Runing naive tracker')
        bdd_out_dir = out_dir + '/bddeval/{}/naive/data/'.format(split)
        for video in videos:
            images = video2images[video['id']]
            file_name = video['name'] if "bdd" in dataset_name else video['file_name']
            print('Runing tracking ...', file_name, len(images))
            preds = [per_image_preds[x['id']] for x in images]
            preds = track(preds)
    coco_json_path = os.path.join(bdd_out_dir,"preds_coco.json")
    bdd_json_dir = os.path.join(bdd_out_dir, "preds_bdd")
    save_cocojson(coco_json_path, videos, images, categories, preds)
    convert_coco_to_bdd(coco_json_path, bdd_json_dir)
    return eval_track(out_dir, dataset_name, custom=True)


def custom_instances_to_coco_json(instances, img_id):
    """
    Add track_id
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    has_mask = instances.has("pred_masks")
    if has_mask:
        rles = [
            mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            for mask in instances.pred_masks
        ]
        for rle in rles:
            rle["counts"] = rle["counts"].decode("utf-8")

    has_keypoints = instances.has("pred_keypoints")
    if has_keypoints:
        keypoints = instances.pred_keypoints

    has_track_id = instances.has("track_ids")
    if has_track_id:
        track_ids = instances.track_ids

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
        }
        if has_mask:
            result["segmentation"] = rles[k]
        if has_keypoints:
            keypoints[k][:, :2] -= 0.5
            result["keypoints"] = keypoints[k].flatten().tolist()
        if has_track_id:
            result['track_id'] = int(track_ids[k].item())
        results.append(result)
    
    return results

class BDDEvaluator(COCOEvaluator):
    def __init__(self, dataset_name, cfg, distributed, output_dir=None, *, use_fast_impl=True, wandb_logger=None):
        super().__init__(dataset_name, cfg, distributed, output_dir=output_dir, use_fast_impl=use_fast_impl)
        self.dataset_name = dataset_name
        self.wandb_logger = wandb_logger if comm.is_main_process() else None

    def process(self, inputs, outputs):
        """
        custom_instances_to_coco_json
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = custom_instances_to_coco_json(instances, input["image_id"])
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            self._predictions.append(prediction)

    def _eval_predictions(self, predictions, img_ids=None):
        """
        Evaluate predictions on the given tasks.
        Fill self._results with the metrics of the tasks.
        """
        assert img_ids is None
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        tasks = self._tasks or self._tasks_from_predictions(coco_results)

        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            # TODO: add new reverse_id_mapping from lvis 0-idxed bdd 1-idxed
            for result in coco_results:
                category_id = result["category_id"]
                assert (
                    category_id in reverse_id_mapping
                ), "A prediction has category_id={}, which is not available in the dataset.".format(
                    category_id
                )
                result["category_id"] = reverse_id_mapping[category_id]

        file_path = os.path.join(self._output_dir, "coco_instances_results.json")
        self._logger.info("Saving results to {}".format(file_path))
        with PathManager.open(file_path, "w") as f:
            f.write(json.dumps(coco_results))
            f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info(
            "Evaluating predictions with {} COCO API...".format(
                "unofficial" if self._use_fast_impl else "official"
            )
        )
        for task in sorted(tasks):
            coco_eval = (
                _evaluate_predictions_on_coco(
                    self._coco_api,
                    coco_results,
                    task,
                    kpt_oks_sigmas=self._kpt_oks_sigmas,
                )
                if len(coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )

            res = self._derive_coco_results(
                coco_eval, task, class_names=self._metadata.get("thing_classes")
            )
            self._results[task] = res

        track_res = track_and_eval_bdd(
            self._output_dir,
            self._coco_api.dataset,  # gt json coco format
            coco_results,
            dataset_name=self.dataset_name
        )
        track_res_dict = track_res.summary()
        track_res_full = track_res.dict()
        self._results.update({"BDD100K": track_res_full})
        if self.wandb_logger is not None:
            self.wandb_logger.log_results(self._results)