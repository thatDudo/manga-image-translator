import numpy as np
import cv2

from utils import ModelWrapper, Quadrilateral

class CommonDetector(ModelWrapper):
    async def detect(self, img: np.ndarray, detect_size: int, args: dict, verbose: bool) -> tuple[cv2.Mat, np.ndarray, list[Quadrilateral]]: ...
