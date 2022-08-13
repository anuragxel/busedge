from tkinter import W
import cv2
from .detection_model import _build_detection_model
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
import rospy

from .utils import *

class CanFilter():
    def __init__(self, cfg_path, weights_path, additional_options=None):
        rospy.loginfo("Creating cfg")
        cfg = get_config_from_path(cfg_path)
        cfg.MODEL.WEIGHTS = weights_path
        rospy.loginfo(f"Building model with config from {cfg_path}")
        rospy.loginfo(f"Building model with weights from {cfg.MODEL.WEIGHTS}")
        self.model, self._config = _build_detection_model(cfg, additional_options=additional_options, predictor_wrapper=DefaultPredictor)

    def detect(self, image):
        return self.model(image)

    def send(self, image, show_flag=False):
        output_dict = self.detect(image)
        if len(output_dict["instances"]) > 0:
            if show_flag:
                visualizer = Visualizer(image)
                out_image = visualizer.draw_instance_predictions(output_dict["instances"].to("cpu"))

                cv2.namedWindow("Sign detector results", 0)
                cv2.imshow("Sign detector results", out_image.get_image()[:, :, ::-1])
                cv2.waitKey(1)
            return True
        else:
            return False

