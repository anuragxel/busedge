# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

# Inspired by https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/object_detection_tutorial.ipynb
# SPDX-FileCopyrightText: 2020 TensorFlow Authors


import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.catalog import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
category_index = {1: "sign"}
CATEGORIES = ['Bird', 'Ground Animal', 'Ambiguous Barrier', 'Concrete Block', 'Curb', 'Fence', 'Guard Rail', 'Barrier', 'Road Median', 'Road Side', 'Lane Separator', 'Temporary Barrier', 'Wall', 'Bike Lane', 'Crosswalk - Plain', 'Curb Cut', 'Driveway', 'Parking', 'Parking Aisle', 'Pedestrian Area', 'Rail Track', 'Road', 'Road Shoulder', 'Service Lane', 'Sidewalk', 'Traffic Island', 'Bridge', 'Building', 'Garage', 'Tunnel', 'Person', 'Person Group', 'Bicyclist', 'Motorcyclist', 'Other Rider', 'Lane Marking - Dashed Line', 'Lane Marking - Straight Line', 'Lane Marking - Zigzag Line', 'Lane Marking - Ambiguous', 'Lane Marking - Arrow (Left)', 'Lane Marking - Arrow (Other)', 'Lane Marking - Arrow (Right)', 'Lane Marking - Arrow (Split Left or Straight)', 'Lane Marking - Arrow (Split Right or Straight)', 'Lane Marking - Arrow (Straight)', 'Lane Marking - Crosswalk', 'Lane Marking - Give Way (Row)', 'Lane Marking - Give Way (Single)', 'Lane Marking - Hatched (Chevron)', 'Lane Marking - Hatched (Diagonal)', 'Lane Marking - Other', 'Lane Marking - Stop Line', 'Lane Marking - Symbol (Bicycle)', 'Lane Marking - Symbol (Other)', 'Lane Marking - Text', 'Lane Marking (only) - Dashed Line', 'Lane Marking (only) - Crosswalk', 'Lane Marking (only) - Other', 'Lane Marking (only) - Test', 'Mountain', 'Sand', 'Sky', 'Snow', 'Terrain', 'Vegetation', 'Water', 'Banner', 'Bench', 'Bike Rack', 'Catch Basin', 'CCTV Camera', 'Fire Hydrant', 'Junction Box', 'Mailbox', 'Manhole', 'Parking Meter', 'Phone Booth', 'Pothole', 'Signage - Advertisement', 'Signage - Ambiguous', 'Signage - Back', 'Signage - Information', 'Signage - Other', 'Signage - Store', 'Street Light', 'Pole', 'Pole Group', 'Traffic Sign Frame', 'Utility Pole', 'Traffic Cone', 'Traffic Light - General (Single)', 'Traffic Light - Pedestrians', 'Traffic Light - General (Upright)', 'Traffic Light - General (Horizontal)', 'Traffic Light - Cyclists', 'Traffic Light - Other', 'Traffic Sign - Ambiguous', 'Traffic Sign (Back)', 'Traffic Sign - Direction (Back)', 'Traffic Sign - Direction (Front)', 'Traffic Sign (Front)', 'Traffic Sign - Parking', 'Traffic Sign - Temporary (Back)', 'Traffic Sign - Temporary (Front)', 'Trash Can', 'Bicycle', 'Boat', 'Bus', 'Car', 'Caravan', 'Motorcycle', 'On Rails', 'Other Vehicle', 'Trailer', 'Truck', 'Vehicle Group', 'Wheeled Slow', 'Water Valve', 'Car Mount', 'Dynamic', 'Ego Vehicle', 'Ground', 'Static', 'Unlabeled']


class CrosswalkFilter:
    def __init__(self, model_dir, min_score_thresh=0.8):
        MetadataCatalog.get("empty_dataset").thing_classes = CATEGORIES
        self.mapillary_metadata = MetadataCatalog.get("empty_dataset")
        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            )
        )
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = min_score_thresh  # set threshold for this model
        cfg.MODEL.WEIGHTS = model_dir
        cfg.MODEL.DEVICE = 'cpu'
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CATEGORIES)
        self.predictor = DefaultPredictor(cfg)


    def detect(self, image):
        img = np.asarray(image)

        outputs = self.predictor(img)
        # idx_class = torch.max(outputs["instances"].to("cpu").pred_classes == 45, outputs["instances"].to("cpu").pred_classes == 14)
        idx_class = outputs["instances"].to("cpu").pred_classes == 45

        outputs["instances"] = outputs["instances"][idx_class]
        num_detections = int(len(outputs["instances"]))
        output_dict = {"num_detections":num_detections}
        return output_dict

