import os
import sys
import copy
import torch
import cv2
import numpy as np
from pathlib import Path
import torchvision.transforms.functional as F
from torchvision.utils import make_grid, save_image
from detectron2.data import MetadataCatalog
from detectron2.structures import Boxes
from gtr.predictor import TrackingVisualizer


def show(imgs, save=False, path="visualization/", name="batch"):
    imgs = [make_grid(imgs, normalize=True)]
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
    if save:
        save_image(imgs, Path(path)/name)

def visualize_batch(frames, gt_instances, file_name="vis", dataset="bdd100k_train"):
    metadata = MetadataCatalog.get(dataset)
    visualizer = TrackingVisualizer(metadata=metadata)
    frames_with_bbox = []
    for frame, gt in zip(frames.numpy(), gt_instances):
        instances = copy.deepcopy(gt)
        instances.track_ids = instances.gt_instance_ids.cpu()
        instances.pred_classes = instances.gt_classes.cpu()
        instances.pred_boxes = Boxes(instances.gt_boxes.tensor.cpu())
        frame = np.transpose(frame, (1,2,0))
        frame = (255 * (frame - np.min(frame)) / np.ptp(frame)).astype(np.uint8)  # normalize
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        vis_frame = visualizer.draw_instance_predictions(frame, instances)
        vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
        frames_with_bbox.append(vis_frame)
    frames_with_bbox = torch.permute(torch.Tensor(frames_with_bbox), (0,3,1,2))
    show(frames_with_bbox, save=True, name=file_name)
