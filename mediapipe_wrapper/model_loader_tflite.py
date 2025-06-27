from .enum import ModelEnum

import numpy as np
import tensorflow as tf
from typing import List

def load_model(model_enum:ModelEnum):
    model = LandmarkModelWrapper(model_enum.value)
    return model

class LandmarkModelWrapper:
    def __init__(self, model_path: str):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']

    def preprocess(self, images: np.ndarray) -> np.ndarray:
        # assume input shape: (N, H, W, 3)
        # model expects (1, 256, 256, 3) or similar
        input_dtype = self.input_details[0]['dtype']
        if input_dtype == np.float32:
            return images.astype(np.float32) / 255.0
        return images.astype(input_dtype)

    def infer(self, batch: np.ndarray) -> List[np.ndarray]:
        preprocessed = self.preprocess(batch)
        results = []
        for img in preprocessed:
            self.interpreter.set_tensor(self.input_details[0]['index'], np.expand_dims(img, axis=0))
            self.interpreter.invoke()
            outputs = [self.interpreter.get_tensor(out['index']) for out in self.output_details]
            results.append(outputs)
        return results
