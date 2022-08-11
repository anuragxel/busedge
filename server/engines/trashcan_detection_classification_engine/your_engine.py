# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import time

import cv2
import imageio
import numpy as np
from gabriel_protocol import gabriel_pb2
from gabriel_server import cognitive_engine
from protocol import busedge_pb2
from utils.exif import set_gps_location
from utils.db_utils import DB_Manager
from detection_model import _build_detection_model
from bounding_box_extractor import extract_predictions_from_image
from classifier import _build_classification_model, _classify, _image_to_normalized_tensor

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

from PIL import Image

logger = logging.getLogger(__name__)

CATEGORIES = [
    "Empty",
    "Full",
    "Garbage Bag"
]

DETECTOR_CFG_PATH = "./model/detector/SERVER_DETECTOR_PAPER.yaml"
DETECTOR_WEIGHTS_PATH = "./model/detector/model_final.pth"

CLASSIFIER_WEIGHTS_PATH = "./model/classifier/classifier_medium_final.pth"

class TrashCanDetectionClassificationEngine(cognitive_engine.Engine):
    def __init__(self, source_name, visualize, save_raw, use_livemap):
        self.mapillary_metadata = MetadataCatalog.get("empty_dataset")

        cfg = get_cfg()
        cfg.merge_from_file(
            DETECTOR_CFG_PATH
        )
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.80  # set threshold for this model
        cfg.MODEL.WEIGHTS = DETECTOR_WEIGHTS_PATH

        logger.info(f"Detector will load weights from: {cfg.MODEL.WEIGHTS}")
        self.predictor, _ = _build_detection_model(cfg)
        logger.info("Detector has been initialized")

        logger.info(f"Classifier will be loading weights from: {CLASSIFIER_WEIGHTS_PATH}")
        self.classifier = _build_classification_model(CLASSIFIER_WEIGHTS_PATH)
        logger.info("Classifier has been initialized")

        self.pred_counter = 0

        self.use_livemap = use_livemap
        if self.use_livemap:
            self.db_manager = DB_Manager()
        self.source_name = source_name
        self.visualize = visualize
        self.save_raw = save_raw

    def handle(self, input_frame):
        # -----------------------------------------------------------#
        # read input from gabriel server
        # -----------------------------------------------------------#
        if input_frame.payload_type != gabriel_pb2.PayloadType.IMAGE:
            status = gabriel_pb2.ResultWrapper.Status.WRONG_INPUT_FORMAT
            return cognitive_engine.create_result_wrapper(status)
        extras = cognitive_engine.unpack_extras(busedge_pb2.EngineFields, input_frame)
        print(f"{extras=}")
        
        gps = [
            extras.gps_data.latitude,
            extras.gps_data.longitude,
            extras.gps_data.altitude,
        ]

        img_name = extras.image_filename
        camera_name = img_name.split("_")[0]
        camera_id = int(camera_name[-1])
        timestamp = img_name.split("_")[1] + "." + img_name.split("_")[2][:-4]

        img_array = np.asarray(bytearray(input_frame.payloads[0]), dtype=np.int8)
        img = cv2.imdecode(img_array, -1)  # cv::IMREAD_UNCHANGED = -1

        pil_img = Image.fromarray(img)

        # -----------------------------------------------------------#
        # Run detector on client input
        # -----------------------------------------------------------#
        time_start = time.time()
        cutouts = extract_predictions_from_image(self.predictor, pil_img, bounding_box_color=None)
        time_end = time.time()
        time_cost = time_end - time_start
        logger.info("Received an image, detection took {} seconds".format(time_cost))

        det_img_folder = time.strftime("%Y_%m_%d/")
        os.makedirs("./images/" + det_img_folder, exist_ok=True)


        for cutout in cutouts:
            self.pred_counter += 1
            classification, confidence = _classify(self.classifier, _image_to_normalized_tensor(cutout), CATEGORIES, 0.87)
            det_img_dir = det_img_folder + img_name
            cutout.save("./images/" + det_img_dir)
            if self.use_livemap:
                self.db_manager.insert_detection(
                    gps[0],
                    gps[1],
                    gps[2],
                    self.pred_counter,
                    "./images/" + det_img_dir,
                    [0, 0, 1000, 1000],
                    classification,
                    camera_id,
                    timestamp
                )

        result_wrapper = gabriel_pb2.ResultWrapper()
        result_wrapper.result_producer_name.value = self.source_name
        result_wrapper.status = gabriel_pb2.ResultWrapper.Status.SUCCESS
        # -----------------------------------------------------------#
        # if you need to send the results back to the client
        # -----------------------------------------------------------#
        # result = gabriel_pb2.ResultWrapper.Result()
        # result.payload_type = gabriel_pb2.PayloadType.IMAGE
        # result.payload = input_frame.payloads[0]
        # result_wrapper.results.append(result)

        return result_wrapper
