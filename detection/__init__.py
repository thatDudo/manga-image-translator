import numpy as np

from .common import CommonDetector
from .default import DefaultDetector
from .ctd import ComicTextDetector

DETECTORS = {
    'default': DefaultDetector,
    'ctd': ComicTextDetector,
}
detector_cache = {}

def get_detector(key: str, *args, **kwargs) -> CommonDetector:
    if key not in DETECTORS:
        raise Exception(f'Could not find detector for: "{key}". Choose from the following: %s' % ', '.join(DETECTORS))
    if key not in detector_cache:
        detector = DETECTORS[key]
        detector_cache[key] = detector(*args, **kwargs)
    return detector_cache[key]

async def dispatch(detector_key: str, img: np.ndarray, detect_size: int, use_cuda: bool, args: dict, verbose: bool = False):
    detector = get_detector(detector_key, use_cuda)
    if not detector.is_loaded():
        detector.load()

    return await detector.detect(img, detect_size, args, verbose)
