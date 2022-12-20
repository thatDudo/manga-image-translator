import numpy as np

# from utils import Quadrilateral
from .ctd_utils import TextBlock
from .default import DefaultDetector
from .common import CommonDetector, OfflineDetector

DETECTORS = {
	'default': DefaultDetector,
}
detector_cache = {}

def get_detector(key: str, *args, **kwargs) -> CommonDetector:
	if key not in DETECTORS:
		raise ValueError(f'Could not find detector for: "{key}". Choose from the following: %s' % ','.join(DETECTORS))
	if not detector_cache.get(key):
		detector = DETECTORS[key]
		detector_cache[key] = detector(*args, **kwargs)
	return detector_cache[key]

async def prepare(detector_key: str):
	detector = get_detector(detector_key)
	if isinstance(detector, OfflineDetector):
		await detector.download()

async def dispatch(detector_key: str, image: np.ndarray, detect_size: int, text_threshold: float, box_threshold: float, unclip_ratio: float, verbose: bool = False, use_cuda: bool = False) -> tuple[list[TextBlock], np.ndarray]:
	detector = get_detector(detector_key)
	if isinstance(detector, OfflineDetector):
		await detector.load('cuda' if use_cuda else 'cpu')
	return await detector.detect(image, detect_size, text_threshold, box_threshold, unclip_ratio, verbose)
