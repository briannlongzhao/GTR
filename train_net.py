import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
import time
import datetime
import sys
import re
import platform
import argparse
import numpy as np
import cv2
import json

from fvcore.common.timer import Timer
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
)
from detectron2.engine import default_argument_parser, default_setup, launch

from detectron2.evaluation import (
    inference_on_dataset,
    COCOEvaluator,
)

from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.detection_utils import read_image
from detectron2.solver import build_optimizer
from detectron2.utils.logger import setup_logger

sys.path.insert(0, "third_party/CenterNet2/")
sys.path.insert(0, "tools/")
from centernet.config import add_centernet_config

from gtr.config import add_gtr_config
from gtr.data.custom_build_augmentation import build_custom_augmentation
from gtr.data.custom_dataset_dataloader import  build_custom_train_loader
from gtr.data.custom_dataset_mapper import CustomDatasetMapper
from gtr.data.gtr_dataset_dataloader import build_gtr_train_loader
from gtr.data.gtr_dataset_dataloader import build_gtr_test_loader
from gtr.data.gtr_dataset_dataloader import build_gtr_vis_loader
from gtr.data.gtr_dataset_mapper import GTRDatasetMapper
from gtr.costom_solver import build_custom_optimizer
from gtr.evaluation.custom_lvis_evaluation import CustomLVISEvaluator
from gtr.evaluation.mot_evaluation import MOTEvaluator
from gtr.evaluation.bdd_evaluation import BDDEvaluator
from gtr.evaluation.bddmot_evaluation import BDDMOTEvaluator
from gtr.modeling.freeze_layers import check_if_freeze_model
from gtr.predictor import VisualizationDemo

from wandb_writer import WandbWriter

logger = logging.getLogger("detectron2")
accum_iter = 1

def get_total_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    device = parameters[0].grad.device
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
        norm_type)
    return total_norm


def do_visualize(cfg, model, dataloader, dataset_name, wandb_logger=None, vis_output=None):
    demo = VisualizationDemo(cfg, model)
    anno = json.load(open(MetadataCatalog.get(dataset_name).json_file))
    if "bdd" in dataset_name:
        vid2name = {item["id"]: item["name"] for item in anno["videos"]}
    elif "tao" in dataset_name:
        vid2name = {item["id"]: item["name"] for item in anno["videos"]}
    elif "mot" in dataset_name:
        vid2name = {item["id"]: item["file_name"] for item in anno["videos"]}
    else:
        raise NotImplementedError

    for count, video in enumerate(dataloader):
        assert video[0]["video_id"] == video[-1]["video_id"], "video_id does not match"
        video_name = vid2name[video[0]["video_id"]]
        logger.info("Running visualization on {}, {}/{}".format(video_name, count+1, len(dataloader)))
        frames = [read_image(item["file_name"]) for item in video]
        vis_video = np.array(list(demo.run_on_images(frames)))
        vis_video = np.einsum("ijkl->iljk", vis_video)
        if vis_output:
            if not os.path.exists(vis_output):
                os.makedirs(vis_output)
            frame_size = (vis_video.shape[3], vis_video.shape[2])
            out = cv2.VideoWriter(os.path.join(vis_output,video_name+"_annotated.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps=5.0, frameSize=frame_size, isColor=True)
            for img in vis_video:
                out.write(np.einsum("ijk->jki",img))
            out.release()
        if wandb_logger:
            wandb_logger.log_video(vis_video, video_name)


def do_test(cfg, model, visualize=False, debug=False, method=None, wandb_logger=None):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference_{}".format(dataset_name))
        print(f"output_folder for {dataset_name}: {output_folder}")
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        print(f"evaluator_type: {evaluator_type}")
        if evaluator_type == "lvis":
            evaluator = CustomLVISEvaluator(dataset_name, cfg, True, output_folder)
        elif evaluator_type == "coco":
            evaluator = COCOEvaluator(dataset_name, cfg, True, output_folder)
        elif evaluator_type == "mot":
            evaluator = MOTEvaluator(dataset_name, cfg, False, output_folder, wandb_logger=wandb_logger)
        elif evaluator_type == "bdd":
            if method == "scalabel":
                evaluator = BDDEvaluator(dataset_name, cfg, False, output_folder, method=method, wandb_logger=wandb_logger)
            elif method == "gtr_custom":
                evaluator = BDDMOTEvaluator(dataset_name, cfg, False, output_folder, wandb_logger=wandb_logger)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError(evaluator_type)

        if not cfg.VIDEO_INPUT:
            if cfg.INPUT.TEST_INPUT_TYPE == "default":
                mapper = None
            else:
                mapper = DatasetMapper(cfg, False, augmentations=build_custom_augmentation(cfg, False))
            data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
            results[dataset_name] = inference_on_dataset(model, data_loader, evaluator)
        else:
            if not comm.is_main_process():
                continue
            # TODO (Xingyi): currently holistic test only works on 1 gpus
            #   due to unknown system issue. Try to fix.
            torch.multiprocessing.set_sharing_strategy("file_system")
            if cfg.INPUT.TEST_INPUT_TYPE == "default":
                mapper = GTRDatasetMapper(cfg, False)
            else:
                mapper = GTRDatasetMapper(cfg, False, augmentations=build_custom_augmentation(cfg, False))
            data_loader = build_gtr_test_loader(cfg, dataset_name, mapper)
            if debug: # Debug: load only one sequence
                data_loader = [next(iter(data_loader))]
            results[dataset_name] = inference_on_dataset(model, data_loader, evaluator,)
            if visualize:
                do_visualize(cfg, model, data_loader, dataset_name, wandb_logger)
        # if comm.is_main_process():
        #     logger.info("Evaluation results for {} in csv format:".format(dataset_name))
        #     print_csv_format(results[dataset_name])
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def do_train(cfg, model, resume=False, debug=False, method="scalabel", wandb_logger=None):
    model = check_if_freeze_model(model, cfg)
    model.train()
    if cfg.SOLVER.USE_CUSTOM_SOLVER:
        optimizer = build_custom_optimizer(cfg, model)
    else:
        assert cfg.SOLVER.OPTIMIZER == "SGD"
        assert cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE != "full_model"
        assert cfg.SOLVER.BACKBONE_MULTIPLIER == 1.
        optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler)

    if cfg.MODEL.WEIGHTS is not None:
        start_iter = (checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume,).get("iteration", -1) + 1)
    if not resume:
        start_iter = 0
    max_iter = cfg.SOLVER.MAX_ITER if cfg.SOLVER.TRAIN_ITER < 0 else cfg.SOLVER.TRAIN_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer,
        cfg.SOLVER.CHECKPOINT_PERIOD,
        max_iter=max_iter,
        file_prefix="ckptr")

    writers = [
        CommonMetricPrinter(max_iter),
        JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
        TensorboardXWriter(cfg.OUTPUT_DIR),
    ] if comm.is_main_process() else []
    if wandb_logger:
        writers.append(wandb_logger)

    #if comm.is_main_process():
    #    wandb_logger.watch(model, log="all", log_graph=True)

    DatasetMapperClass = GTRDatasetMapper if cfg.VIDEO_INPUT else CustomDatasetMapper
    mapper = DatasetMapperClass(cfg, True, augmentations=build_custom_augmentation(cfg, True))
    if cfg.VIDEO_INPUT:
        data_loader, dataset_dicts = build_gtr_train_loader(cfg, mapper=mapper)
    else:
        data_loader, dataset_dicts = build_custom_train_loader(cfg, mapper=mapper)
    num_frames = [len(video["images"]) for video in dataset_dicts]
    total_frames = sum(num_frames)

    if debug: # Debug: only run few iterations for training
        max_iter = 100

    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        step_timer = Timer()
        data_timer = Timer()
        start_time = time.perf_counter()
        model.zero_grad()
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            approx_epoch = iteration * cfg.SOLVER.IMS_PER_BATCH * cfg.INPUT.VIDEO.TRAIN_LEN / total_frames
            data_time = data_timer.seconds()
            storage.put_scalars(data_time=data_time)
            step_timer.reset()
            iteration = iteration + 1
            storage.step()
            loss_dict, cls_acc = model(data)
            
            losses = sum(loss for k, loss in loss_dict.items() if "loss" in k)
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for k, loss in loss_dict_reduced.items() if "loss" in k)
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced, cls_acc=cls_acc, approx_epoch=approx_epoch)

            (losses / accum_iter).backward()
            if (iteration+1) % accum_iter == 0 or iteration+1 == max_iter:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            step_time = step_timer.seconds()
            storage.put_scalars(time=step_time)
            data_timer.reset()
            scheduler.step()

            periodic_checkpointer.step(iteration)

            if (
                cfg.TEST.EVAL_PERIOD > 0 and
                iteration % cfg.TEST.EVAL_PERIOD == 0 and
                iteration != max_iter
            ):
                do_test(cfg, model, debug=debug, method=method, wandb_logger=wandb_logger)
                comm.synchronize()

            if iteration - start_iter > 5 and (iteration % 50 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write(step=iteration) if wandb_logger and type(writer) is WandbWriter else writer.write()


        total_time = time.perf_counter() - start_time
        logger.info("Total training time: {}".format(str(datetime.timedelta(seconds=int(total_time)))))


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_gtr_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if args.num_gpus:
        cfg.SOLVER.IMS_PER_BATCH = args.num_gpus
    if args.base_lr:
        cfg.SOLVER.BASE_LR = float(args.base_lr)
    if args.optimizer:
        cfg.SOLVER.OPTIMIZER = args.optimizer
        cfg.SOLVER.USE_CUSTOM_SOLVER = False if args.optimizer == "SGD" else True
        cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "full_model" if args.optimizer != "SGD" else "value"
        cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0 if args.optimizer == "SGD" else 0.1
    if args.test_nms_th:
        cfg.MODEL.CENTERNET.NMS_TH_TEST = float(args.test_nms_th)
    if args.test_asso_th:
        cfg.MODEL.ASSO_HEAD.ASSO_THRESH_TEST = float(args.test_asso_th)
    if args.test_len:
        cfg.INPUT.VIDEO.TEST_LEN = int(args.test_len)
    if args.noise_level:
        cfg.MODEL.RESNETS.NOISE_LEVEL = float(args.noise_level)
    if "TMPDIR" in os.environ.keys():
        # cfg.OUTPUT_DIR = (cfg.OUTPUT_DIR).replace('.', os.path.join(os.environ["TMPDIR"], "GTR"))
        if cfg.MODEL.WEIGHTS is not None:
            cfg.MODEL.WEIGHTS = os.path.join(os.environ["TMPDIR"], "GTR" ,cfg.MODEL.WEIGHTS)
    if "/auto" in cfg.OUTPUT_DIR:
        file_name = os.path.basename(args.config_file)[:-5]
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.replace("/auto", "/{}".format(file_name))
    logger.info("OUTPUT_DIR: {}".format(cfg.OUTPUT_DIR))
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="centernet")
    return cfg


def main(args):
    cfg = setup(args)
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))

    if comm.is_main_process() and args.wandb:
        wandb_logger = WandbWriter(project="GTR", config=cfg)
    else:
        wandb_logger = None

    if args.vis_only:
        dataset_name = cfg.DATASETS.TEST[0]
        if cfg.INPUT.TEST_INPUT_TYPE == "default":
            mapper = GTRDatasetMapper(cfg, False)
        else:
            mapper = GTRDatasetMapper(cfg, False, augmentations=build_custom_augmentation(cfg, False))
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        if args.vis_input:
            data_loader = build_gtr_vis_loader(cfg, args.vis_input, mapper)
        else:
            data_loader = build_gtr_test_loader(cfg, dataset_name, mapper)
        if args.debug:  # Debug: load only one sequence
            data_loader = [next(iter(data_loader))]
        return do_visualize(cfg, model, data_loader, dataset_name, wandb_logger, args.vis_output)
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model, visualize=args.visualize, debug=args.debug, method=args.eval_method, wandb_logger=wandb_logger)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False,
            find_unused_parameters=cfg.FIND_UNUSED_PARAM
        )

    do_train(cfg, model, resume=args.resume, debug=args.debug, method=args.eval_method, wandb_logger=wandb_logger)
    do_test(cfg, model, visualize=args.visualize, debug=args.debug, method=args.eval_method, wandb_logger=wandb_logger)
    if comm.is_main_process() and wandb_logger is not None:
        wandb_logger.close()
    return


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--eval_method", choices=["scalabel","mmdet","gtr_custom","custom"], default="scalabel")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--vis-only", action="store_true")
    parser.add_argument("--vis-input")
    parser.add_argument("--vis-output")
    parser.add_argument("--noise-level")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--wandb", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--base_lr")
    parser.add_argument("--optimizer")
    parser.add_argument("--test_nms_th")
    parser.add_argument("--test_asso_th")
    parser.add_argument("--test_len")
    args = parser.parse_args()
    args.dist_url = "tcp://127.0.0.1:{}".format(
        torch.randint(11111, 60000, (1,))[0].item())
    print("Command Line Args:", args)

    hostname = platform.node()
    if "iGpu" in hostname or "iLab" in hostname:
        os.environ["TMPDIR"] = "/lab/tmpig8e/u/brian-data"
    elif "discovery" in hostname or "hpc" in hostname or re.search("[a-z]\d\d-\d\d", hostname):
        os.environ["TMPDIR"] = "/scratch1/briannlz"
    print(f"train_net.py: HOSTNAME={hostname}")
    print(f"train_net.py: TMPDIR={os.environ['TMPDIR']}")

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
