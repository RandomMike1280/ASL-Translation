from .enum import ModelEnum

import numpy as np
import torch
from typing import List

def load_model(model_enum:ModelEnum):
    match model_enum:
        case ModelEnum.POSE_LANDMARK_FULL:
            from .models.pytorch_code.pose_landmark_full import pose_landmark_full
            model = pose_landmark_full()
        case ModelEnum.FACE_LANDMARK:
            from .models.pytorch_code.face_landmark import face_landmark
            model = face_landmark()
        case ModelEnum.HAND_LANDMARK_FULL:
            from .models.pytorch_code.hand_landmark_full import hand_landmark_full
            model = hand_landmark_full()
        case ModelEnum.HAND_LANDMARK_LITE:
            from .models.pytorch_code.hand_landmark_lite import hand_landmark_lite
            model = hand_landmark_lite()
        # case ModelEnum.POSE_DETECTION:
        #     from .models.pytorch_code import pose_landmark_full
        #     model = pose_landmark_full()
        case ModelEnum.PALM_DETECTION_FULL:
            from .models.pytorch_code.palm_detection_full import palm_detection_full
            model = palm_detection_full()
        case ModelEnum.PALM_DETECTION_LITE:
            from .models.pytorch_code.palm_detection_lite import palm_detection_lite
            model = palm_detection_lite()