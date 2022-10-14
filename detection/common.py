import numpy as np

class CommonDetector():
    _MODEL_FILE = None

    def __init__(self, use_cuda: bool):
        self._use_cuda = use_cuda
        self._loaded = False

    # Can be used in the future to specify a model directory
    def _get_model_path(self):
        return self._MODEL_FILE

    def is_model_loaded(self):
        return self._loaded

    def load_model(self):
        if not self.is_model_loaded():
            self._load_model()
            self._loaded = True

    def _load_model(self): ...

    async def detect(self, img: np.ndarray, detect_size: int, args: dict, verbose: bool): ...
