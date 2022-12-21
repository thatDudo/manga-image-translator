import os
import stat
from typing import List
import numpy as np
import cv2
import functools
from abc import ABC, abstractmethod
from shapely.geometry import Polygon, MultiPoint
from PIL import Image
import tqdm
import requests
import sys
import hashlib
import tempfile
import re
import torch
import shutil
import filecmp

try:
  	functools.cached_property
except AttributeError: # Supports Python versions below 3.8
	from backports.cached_property import cached_property
	functools.cached_property = cached_property


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_digest(file_path: str) -> str:
	h = hashlib.sha256()
	BUF_SIZE = 65536 

	with open(file_path, 'rb') as file:
		while True:
			# Reading is buffered, so we can read smaller chunks.
			chunk = file.read(BUF_SIZE)
			if not chunk:
				break
			h.update(chunk)
	return h.hexdigest()

def get_filename_from_url(url: str, default: str = '') -> str:
	m = re.search(r'/([^/?]+)[^/]*$', url)
	if m:
		return m.group(1)
	return default

def download_url_with_progressbar(url: str, path: str):
	# TODO: Fix partial downloads
	if os.path.basename(path) in ('.', '') or os.path.isdir(path):
		new_filename = get_filename_from_url(url)
		if not new_filename:
			raise Exception('Could not determine filename')
		path = os.path.join(path, new_filename)

	headers = {}
	downloaded_size = 0
	if os.path.isfile(path):
		downloaded_size = os.path.getsize(path)
		headers['Range'] = 'bytes=%d-' % downloaded_size

	r = requests.get(url, stream=True, allow_redirects=True, headers=headers)
	if downloaded_size and r.headers.get('Accept-Ranges') != 'bytes':
		print('Error: Webserver does not support partial downloads. Restarting from the beginning!')
		r = requests.get(url, stream=True, allow_redirects=True)
		downloaded_size = 0
	total = int(r.headers.get('content-length', 0))
	chunk_size = 1024

	if r.ok:
		with tqdm.tqdm(
			desc=os.path.basename(path),
			initial=downloaded_size,
			total=total,
			unit='iB',
			unit_scale=True,
			unit_divisor=chunk_size,
		) as bar:
			with open(path, 'ab' if downloaded_size else 'wb') as f:
				is_tty = sys.stdout.isatty()
				downloaded_chunks = 0
				for data in r.iter_content(chunk_size=chunk_size):
					size = f.write(data)
					bar.update(size)

					# Fallback for non TTYs so output still shown
					downloaded_chunks += 1
					if not is_tty and downloaded_chunks % 1000 == 0:
						print(bar)
	else:
		raise Exception(f'Couldn\'t resolve url: "{url}" (Error: {r.status_code})')

def prompt_yes_no(query: str, default: bool = None) -> bool:
	s = '%s (%s/%s): ' % (query, 'Y' if default == True else 'y', 'N' if default == False else 'n')
	while True:
		inp = input(s).lower()
		if inp in ('yes', 'y'):
			return True
		elif inp in ('no', 'n'):
			return False
		elif default != None:
			return default
		if inp:
			print('Error: Could not parse input')

MODULE_PATH = os.path.dirname(os.path.realpath(__file__))

class ModelVerificationException(Exception):
	pass

class InvalidModelMappingException(ValueError):
	def __init__(self, cls: str, map_key: str, error_msg: str):
		error = f'[{cls}->{map_key}] Invalid _MODEL_MAPPING - {error_msg}'
		super().__init__(error)

class ModelWrapper(ABC):
	_MODEL_DIR = os.path.join(MODULE_PATH, 'models')
	_MODEL_MAPPING = {}

	def __init__(self):
		os.makedirs(self._MODEL_DIR, exist_ok=True)
		self._loaded = False
		self._check_for_malformed_model_mapping()
		self._downloaded = self._check_downloaded()

	def is_loaded(self) -> bool:
		return self._loaded

	def is_downloaded(self) -> bool:
		return self._downloaded

	def _get_file_path(self, relative_path: str) -> str:
		return os.path.join(self._MODEL_DIR, relative_path)

	def _get_used_gpu_memory(self) -> bool:
		'''
		Gets the total amount of GPU memory used by model (Can be used in the future
		to determine whether a model should be loaded into vram or ram or automatically choose a model size).
		TODO: Use together with `--use-cuda-limited` flag to enforce stricter memory checks
		'''
		return torch.cuda.mem_get_info()

	def _check_for_malformed_model_mapping(self):
		for map_key, map in self._MODEL_MAPPING.items():
			if 'url' not in map:
				raise InvalidModelMappingException(self.__class__.__name__, map_key, 'Missing url property')
			elif not re.search(r'^https?://', map['url']):
				raise InvalidModelMappingException(self.__class__.__name__, map_key, 'Malformed url property: "%s"' % map['url'])
			if 'file' not in map and 'archive' not in map:
				map['file'] = '.'
			elif 'file' in map and 'archive' in map:
				raise InvalidModelMappingException(self.__class__.__name__, map_key, 'Properties file and archive are mutually exclusive')

	async def _download_file(self, url: str, path: str):
		print(f' -- Downloading: "{url}"')
		download_url_with_progressbar(url, path)

	async def _verify_file(self, sha256_pre_calculated: str, path: str):
		print(f' -- Verifying: "{path}"')
		sha256_calculated = get_digest(path).lower()
		sha256_pre_calculated = sha256_pre_calculated.lower()

		if sha256_calculated != sha256_pre_calculated:
			self._on_verify_failure(sha256_calculated, sha256_pre_calculated)
		else:
			print(' -- Verifying: OK!')

	def _on_verify_failure(self, sha256_calculated: str, sha256_pre_calculated: str):
		print(f' -- Mismatch between downloaded and created hash: "{sha256_calculated}" <-> "{sha256_pre_calculated}"')
		raise ModelVerificationException()

	@functools.cached_property
	def _temp_working_directory(self):
		p = os.path.join(tempfile.gettempdir(), 'manga-image-translator', self.__class__.__name__.lower())
		os.makedirs(p, exist_ok=True)
		return p

	async def download(self, force=False):
		'''
		Downloads required models.
		'''
		if force or not self.is_downloaded():
			while True:
				try:
					await self._download()
					self._downloaded = True
					break
				except ModelVerificationException:
					if not prompt_yes_no('Failed to verify signature. Do you want to restart the download?', default=True):
						print('Aborting.')
						raise KeyboardInterrupt()

	async def _download(self):
		'''
		Downloads models as defined in `_MODEL_MAPPING`. Can be overwritten (together
		with `_check_downloaded`) to implement unconventional download logic.
		'''
		print('\nDownloading models\n')
		for map_key, map in self._MODEL_MAPPING.items():
			if self._check_downloaded_map(map_key):
				print(f' -- Skipping {map_key} as it\'s already downloaded')
				continue

			is_archive = 'archive' in map
			if is_archive:
				download_path = os.path.join(self._temp_working_directory, map_key, '')
			else:
				download_path = self._get_file_path(map['file'])
			if not os.path.basename(download_path):
				os.makedirs(download_path, exist_ok=True)
			if os.path.basename(download_path) in ('', '.'):
				download_path = os.path.join(download_path, get_filename_from_url(map['url'], map_key))
			if not is_archive:
				download_path += '.part'

			if 'hash' in map:
				downloaded = False
				if os.path.isfile(download_path):
					try:
						print(' -- Found existing file')
						await self._verify_file(map['hash'], download_path)
						downloaded = True
					except ModelVerificationException:
						print(' -- Resuming interrupted download')
				if not downloaded:
					await self._download_file(map['url'], download_path)
					await self._verify_file(map['hash'], download_path)
			else:
				await self._download_file(map['url'], download_path)

			if download_path.endswith('.part'):
				p = download_path[:len(download_path)-5]
				shutil.move(download_path, p)
				download_path = p

			if is_archive:
				extracted_path = os.path.join(os.path.dirname(download_path), 'extracted')
				shutil.unpack_archive(download_path, extracted_path)

				def get_real_archive_files():
					archive_files = []
					for root, dirs, files in os.walk(extracted_path):
						for name in files:
							file_path = os.path.join(root, name)
							archive_files.append(file_path)
					return archive_files

				for orig, dest in map['archive'].items():
					p1 = os.path.join(extracted_path, orig)
					if os.path.exists(p1):
						p2 = self._get_file_path(dest)
						if os.path.basename(p2) in ('', '.'):
							p2 = os.path.join(p2, os.path.basename(p1))
						if os.path.isfile(p2):
							if filecmp.cmp(p1, p2):
								continue
							raise InvalidModelMappingException(self.__class__.__name__, map_key, 'File "{orig}" already exists at "{dest}"')
						os.makedirs(os.path.dirname(p2), exist_ok=True)
						shutil.move(p1, p2)
					else:
						raise InvalidModelMappingException(self.__class__.__name__, map_key, 'File "{orig}" does not exist within archive' +
								        '\nAvailable files:\n%s' % '\n'.join(get_real_archive_files()))
				if len(map['archive']) == 0:
					raise InvalidModelMappingException(self.__class__.__name__, map_key, 'No archive files specified' +
					                    '\nAvailable files:\n%s' % '\n'.join(get_real_archive_files()))

				self._grant_execute_permissions(map_key)

			print()

	def _check_downloaded(self) -> bool:
		'''
		Scans filesystem for required files as defined in `_MODEL_MAPPING`.
		Returns `False` if files should be redownloaded.
		'''
		for map_key in self._MODEL_MAPPING:
			if not self._check_downloaded_map(map_key):
				return False
		return True

	def _check_downloaded_map(self, map_key: str) -> str:
		map = self._MODEL_MAPPING[map_key]

		if 'file' in map:
			path = map['file']
			if os.path.basename(path) in ('.', ''):
				path = os.path.join(path, get_filename_from_url(map['url'], map_key))
			if not os.path.exists(self._get_file_path(path)):
				return False
		
		elif 'archive' in map:
			for from_path, to_path in map['archive'].items():
				if os.path.basename(to_path) in ('.', ''):
					to_path = os.path.join(to_path, os.path.basename(from_path))
				if not os.path.exists(self._get_file_path(to_path)):
					return False

		self._grant_execute_permissions(map_key)

		return True

	def _grant_execute_permissions(self, map_key: str):
		map = self._MODEL_MAPPING[map_key]

		if sys.platform == 'linux':
			# Grant permission to executables
			for file in map.get('executables', []):
				p = self._get_file_path(file)
				if os.path.basename(p) in ('', '.'):
					p = os.path.join(p, file)
				if not os.path.isfile(p):
					raise InvalidModelMappingException(self.__class__.__name__, map_key, f'File "{file}" does not exist')
				if not os.access(p, os.X_OK):
					os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

	async def reload(self, device: str, *args, **kwargs):
		await self.unload()
		await self.load(*args, **kwargs, device=device)

	async def load(self, device: str, *args, **kwargs):
		'''
		Loads models into memory. Has to be called before `forward`.
		'''
		if not self.is_downloaded():
			await self.download()
		if not self.is_loaded():
			await self._load(*args, **kwargs, device=device)
			self._loaded = True

	async def unload(self):
		if self.is_loaded():
			await self._unload()
			self._loaded = False

	async def forward(self, *args, **kwargs):
		'''
		Makes a forward pass through the network.
		'''
		if not self.is_loaded():
			raise Exception(f'{self.__class__.__name__}: Tried to forward pass without having loaded the model.')
		return await self._forward(*args, **kwargs)

	@abstractmethod
	async def _load(self, device: str, *args, **kwargs):
		pass

	@abstractmethod
	async def _unload(self):
		pass

	@abstractmethod
	async def _forward(self, *args, **kwargs):
		pass


class AvgMeter():
	def __init__(self):
		self.reset()

	def reset(self):
		self.sum = 0
		self.count = 0

	def __call__(self, val = None):
		if val is not None:
			self.sum += val
			self.count += 1
		if self.count > 0:
			return self.sum / self.count
		else:
			return 0

def load_image(img: Image.Image):
	if img.mode == 'RGBA':
		# from https://stackoverflow.com/questions/9166400/convert-rgba-png-to-rgb-with-pil
		img.load()  # needed for split()
		background = Image.new('RGB', img.size, (255, 255, 255))
		alpha_ch = img.split()[3]
		background.paste(img, mask = alpha_ch)  # 3 is the alpha channel
		return np.array(background), alpha_ch
	elif img.mode == 'P':
		img = img.convert('RGBA')
		img.load()  # needed for split()
		background = Image.new('RGB', img.size, (255, 255, 255))
		alpha_ch = img.split()[3]
		background.paste(img, mask = alpha_ch)  # 3 is the alpha channel
		return np.array(background), alpha_ch
	else:
		return np.array(img.convert('RGB')), None

def dump_image(img: np.ndarray, alpha_ch: Image.Image):
	if alpha_ch is not None:
		img = np.concatenate([img.astype(np.uint8), np.array(alpha_ch).astype(np.uint8)[..., None]], axis = 2)
	else:
		img = img.astype(np.uint8)
	return Image.fromarray(img)

def resize_keep_aspect(img, size):
	ratio = (float(size)/max(img.shape[0], img.shape[1]))
	new_width = round(img.shape[1] * ratio)
	new_height = round(img.shape[0] * ratio)
	return cv2.resize(img, (new_width, new_height), interpolation = cv2.INTER_LINEAR_EXACT)

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
	# initialize the dimensions of the image to be resized and
	# grab the image size
	dim = None
	(h, w) = image.shape[:2]

	# if both the width and height are None, then return the
	# original image
	if width is None and height is None:
		return image

	# check to see if the width is None
	if width is None:
		# calculate the ratio of the height and construct the
		# dimensions
		r = height / float(h)
		dim = (int(w * r), height)

	# otherwise, the height is None
	else:
		# calculate the ratio of the width and construct the
		# dimensions
		r = width / float(w)
		dim = (width, int(h * r))

	# resize the image
	resized = cv2.resize(image, dim, interpolation = inter)

	# return the resized image
	return resized

class BBox(object):
	def __init__(self, x: int, y: int, w: int, h: int, text: str, prob: float, fg_r: int = 0, fg_g: int = 0, fg_b: int = 0, bg_r: int = 0, bg_g: int = 0, bg_b: int = 0):
		self.x = x
		self.y = y
		self.w = w
		self.h = h
		self.text = text
		self.prob = prob
		self.fg_r = fg_r
		self.fg_g = fg_g
		self.fg_b = fg_b
		self.bg_r = bg_r
		self.bg_g = bg_g
		self.bg_b = bg_b

	def width(self):
		return self.w

	def height(self):
		return self.h

	def to_points(self):
		tl, tr, br, bl = np.array([self.x, self.y]), np.array([self.x + self.w, self.y]), np.array([self.x + self.w, self.y+ self.h]), np.array([self.x, self.y + self.h])
		return tl, tr, br, bl

class Quadrilateral(object):
	def __init__(self, pts: np.ndarray, text: str, prob: float, fg_r: int = 0, fg_g: int = 0, fg_b: int = 0, bg_r: int = 0, bg_g: int = 0, bg_b: int = 0):
		self.pts = pts
		self.text = text
		self.prob = prob
		self.fg_r = fg_r
		self.fg_g = fg_g
		self.fg_b = fg_b
		self.bg_r = bg_r
		self.bg_g = bg_g
		self.bg_b = bg_b
		self.assigned_direction: str = None
		self.textlines: list[Quadrilateral] = []

	@functools.cached_property
	def structure(self) -> List[np.ndarray]:
		p1 = ((self.pts[0] + self.pts[1]) / 2).astype(int)
		p2 = ((self.pts[2] + self.pts[3]) / 2).astype(int)
		p3 = ((self.pts[1] + self.pts[2]) / 2).astype(int)
		p4 = ((self.pts[3] + self.pts[0]) / 2).astype(int)
		return [p1, p2, p3, p4]

	@functools.cached_property
	def valid(self) -> bool:
		[l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in self.structure]
		v1 = l1b - l1a
		v2 = l2b - l2a
		unit_vector_1 = v1 / np.linalg.norm(v1)
		unit_vector_2 = v2 / np.linalg.norm(v2)
		dot_product = np.dot(unit_vector_1, unit_vector_2)
		angle = np.arccos(dot_product) * 180 / np.pi
		return abs(angle - 90) < 10

	@functools.cached_property
	def fg_colors(self):
		return self.fg_r, self.fg_g, self.fg_b

	@functools.cached_property
	def bg_colors(self):
		return self.bg_r, self.bg_g, self.bg_b

	@functools.cached_property
	def aspect_ratio(self) -> float:
		[l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in self.structure]
		v1 = l1b - l1a
		v2 = l2b - l2a
		return np.linalg.norm(v2) / np.linalg.norm(v1)

	@functools.cached_property
	def font_size(self) -> float:
		[l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in self.structure]
		v1 = l1b - l1a
		v2 = l2b - l2a
		return min(np.linalg.norm(v2), np.linalg.norm(v1))

	def width(self) -> int:
		return self.aabb.w

	def height(self) -> int:
		return self.aabb.h

	def clip(self, width, height):
		self.pts[:, 0] = np.clip(np.round(self.pts[:, 0]), 0, width)
		self.pts[:, 1] = np.clip(np.round(self.pts[:, 1]), 0, height)

	@functools.cached_property
	def points(self):
		ans = [a.astype(np.float32) for a in self.structure]
		return [Point(a[0], a[1]) for a in ans]

	@functools.cached_property
	def aabb(self) -> BBox:
		kq = self.pts
		max_coord = np.max(kq, axis = 0)
		min_coord = np.min(kq, axis = 0)
		return BBox(min_coord[0], min_coord[1], max_coord[0] - min_coord[0], max_coord[1] - min_coord[1], self.text, self.prob, self.fg_r, self.fg_g, self.fg_b, self.bg_r, self.bg_g, self.bg_b)

	def get_transformed_region(self, img, direction, textheight) -> np.ndarray:
		[l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in self.structure]
		v_vec = l1b - l1a
		h_vec = l2b - l2a
		ratio = np.linalg.norm(v_vec) / np.linalg.norm(h_vec)
		src_pts = self.pts.astype(np.float32)
		self.assigned_direction = direction
		if direction == 'h':
			h = int(textheight)
			w = int(round(textheight / ratio))
			dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).astype(np.float32)
			M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
			region = cv2.warpPerspective(img, M, (w, h))
			return region
		elif direction == 'v':
			w = int(textheight)
			h = int(round(textheight * ratio))
			dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).astype(np.float32)
			M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
			region = cv2.warpPerspective(img, M, (w, h))
			region = cv2.rotate(region, cv2.ROTATE_90_COUNTERCLOCKWISE)
			return region

	@functools.cached_property
	def is_axis_aligned(self) -> bool:
		[l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in self.structure]
		v1 = l1b - l1a
		v2 = l2b - l2a
		e1 = np.array([0, 1])
		e2 = np.array([1, 0])
		unit_vector_1 = v1 / np.linalg.norm(v1)
		unit_vector_2 = v2 / np.linalg.norm(v2)
		if abs(np.dot(unit_vector_1, e1)) < 1e-2 or abs(np.dot(unit_vector_1, e2)) < 1e-2:
			return True
		return False

	@functools.cached_property
	def is_approximate_axis_aligned(self) -> bool:
		[l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in self.structure]
		v1 = l1b - l1a
		v2 = l2b - l2a
		e1 = np.array([0, 1])
		e2 = np.array([1, 0])
		unit_vector_1 = v1 / np.linalg.norm(v1)
		unit_vector_2 = v2 / np.linalg.norm(v2)
		if abs(np.dot(unit_vector_1, e1)) < 0.05 or abs(np.dot(unit_vector_1, e2)) < 0.05 or abs(np.dot(unit_vector_2, e1)) < 0.05 or abs(np.dot(unit_vector_2, e2)) < 0.05:
			return True
		return False

	@functools.cached_property
	def direction(self) -> str:
		[l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in self.structure]
		v_vec = l1b - l1a
		h_vec = l2b - l2a
		if np.linalg.norm(v_vec) > np.linalg.norm(h_vec):
			return 'v'
		else:
			return 'h'

	@functools.cached_property
	def cosangle(self) -> float:
		[l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in self.structure]
		v1 = l1b - l1a
		e2 = np.array([1, 0])
		unit_vector_1 = v1 / np.linalg.norm(v1)
		return np.dot(unit_vector_1, e2)

	@functools.cached_property
	def angle(self) -> float:
		return np.fmod(np.arccos(self.cosangle) + np.pi, np.pi)

	@functools.cached_property
	def centroid(self) -> np.ndarray:
		return np.average(self.pts, axis = 0)

	def distance_to_point(self, p: np.ndarray) -> float:
		d = 1.0e20
		for i in range(4):
			d = min(d, distance_point_point(p, self.pts[i]))
			d = min(d, distance_point_lineseg(p, self.pts[i], self.pts[(i + 1) % 4]))
		return d

	@functools.cached_property
	def polygon(self) -> Polygon:
		return MultiPoint([tuple(self.pts[0]), tuple(self.pts[1]), tuple(self.pts[2]), tuple(self.pts[3])]).convex_hull

	@functools.cached_property
	def area(self) -> float:
		return self.polygon.area

	def poly_distance(self, other) -> float:
		return self.polygon.distance(other.polygon)

	def distance(self, other, rho = 0.5) -> float:
		return self.distance_impl(other, rho)# + 1000 * abs(self.angle - other.angle)

	def distance_impl(self, other, rho = 0.5) -> float:
		assert self.assigned_direction == other.assigned_direction
		#return gjk_distance(self.points, other.points)
		# b1 = self.aabb
		# b2 = b2.aabb
		# x1, y1, w1, h1 = b1.x, b1.y, b1.w, b1.h
		# x2, y2, w2, h2 = b2.x, b2.y, b2.w, b2.h
		# return rect_distance(x1, y1, x1 + w1, y1 + h1, x2, y2, x2 + w2, y2 + h2)
		pattern = ''
		if self.assigned_direction == 'h':
			pattern = 'h_left'
		else:
			pattern = 'v_top'
		fs = max(self.font_size, other.font_size)
		if self.assigned_direction == 'h':
			poly1 = MultiPoint([tuple(self.pts[0]), tuple(self.pts[3]), tuple(other.pts[0]), tuple(other.pts[3])]).convex_hull
			poly2 = MultiPoint([tuple(self.pts[2]), tuple(self.pts[1]), tuple(other.pts[2]), tuple(other.pts[1])]).convex_hull
			poly3 = MultiPoint([
				tuple(self.structure[0]),
				tuple(self.structure[1]),
				tuple(other.structure[0]),
				tuple(other.structure[1])
			]).convex_hull
			dist1 = poly1.area / fs
			dist2 = poly2.area / fs
			dist3 = poly3.area / fs
			if dist1 < fs * rho:
				pattern = 'h_left'
			if dist2 < fs * rho and dist2 < dist1:
				pattern = 'h_right'
			if dist3 < fs * rho and dist3 < dist1 and dist3 < dist2:
				pattern = 'h_middle'
			if pattern == 'h_left':
				return dist(self.pts[0][0], self.pts[0][1], other.pts[0][0], other.pts[0][1])
			elif pattern == 'h_right':
				return dist(self.pts[1][0], self.pts[1][1], other.pts[1][0], other.pts[1][1])
			else:
				return dist(self.structure[0][0], self.structure[0][1], other.structure[0][0], other.structure[0][1])
		else:
			poly1 = MultiPoint([tuple(self.pts[0]), tuple(self.pts[1]), tuple(other.pts[0]), tuple(other.pts[1])]).convex_hull
			poly2 = MultiPoint([tuple(self.pts[2]), tuple(self.pts[3]), tuple(other.pts[2]), tuple(other.pts[3])]).convex_hull
			dist1 = poly1.area / fs
			dist2 = poly2.area / fs
			if dist1 < fs * rho:
				pattern = 'v_top'
			if dist2 < fs * rho and dist2 < dist1:
				pattern = 'v_bottom'
			if pattern == 'v_top':
				return dist(self.pts[0][0], self.pts[0][1], other.pts[0][0], other.pts[0][1])
			else:
				return dist(self.pts[2][0], self.pts[2][1], other.pts[2][0], other.pts[2][1])

def dist(x1, y1, x2, y2):
	return np.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))

def rect_distance(x1, y1, x1b, y1b, x2, y2, x2b, y2b):
	left = x2b < x1
	right = x1b < x2
	bottom = y2b < y1
	top = y1b < y2
	if top and left:
		return dist(x1, y1b, x2b, y2)
	elif left and bottom:
		return dist(x1, y1, x2b, y2b)
	elif bottom and right:
		return dist(x1b, y1, x2, y2b)
	elif right and top:
		return dist(x1b, y1b, x2, y2)
	elif left:
		return x1 - x2b
	elif right:
		return x2 - x1b
	elif bottom:
		return y1 - y2b
	elif top:
		return y2 - y1b
	else:			 # rectangles intersect
		return 0

def distance_point_point(a: np.ndarray, b: np.ndarray) -> float:
	return np.linalg.norm(a - b)

# from https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
def distance_point_lineseg(p: np.ndarray, p1: np.ndarray, p2: np.ndarray):
	x = p[0]
	y = p[1]
	x1 = p1[0]
	y1 = p1[1]
	x2 = p2[0]
	y2 = p2[1]
	A = x - x1
	B = y - y1
	C = x2 - x1
	D = y2 - y1

	dot = A * C + B * D
	len_sq = C * C + D * D
	param = -1
	if len_sq != 0:
		param = dot / len_sq

	if param < 0:
		xx = x1
		yy = y1
	elif param > 1:
		xx = x2
		yy = y2
	else:
		xx = x1 + param * C
		yy = y1 + param * D

	dx = x - xx
	dy = y - yy
	return np.sqrt(dx * dx + dy * dy)


def quadrilateral_can_merge_region(a: Quadrilateral, b: Quadrilateral, ratio = 1.9, discard_connection_gap = 5, char_gap_tolerance = 0.6, char_gap_tolerance2 = 1.5, font_size_ratio_tol = 1.5, aspect_ratio_tol = 2) -> bool:
	b1 = a.aabb
	b2 = b.aabb
	char_size = min(a.font_size, b.font_size)
	x1, y1, w1, h1 = b1.x, b1.y, b1.w, b1.h
	x2, y2, w2, h2 = b2.x, b2.y, b2.w, b2.h
	dist = rect_distance(x1, y1, x1 + w1, y1 + h1, x2, y2, x2 + w2, y2 + h2)
	if dist > discard_connection_gap * char_size:
		return False
	if max(a.font_size, b.font_size) / char_size > font_size_ratio_tol:
		return False
	if a.aspect_ratio > aspect_ratio_tol and b.aspect_ratio < 1. / aspect_ratio_tol:
		return False
	if b.aspect_ratio > aspect_ratio_tol and a.aspect_ratio < 1. / aspect_ratio_tol:
		return False
	a_aa = a.is_approximate_axis_aligned
	b_aa = b.is_approximate_axis_aligned
	if a_aa and b_aa:
		if dist < char_size * char_gap_tolerance:
			if abs(x1 + w1 // 2 - (x2 + w2 // 2)) < char_gap_tolerance2:
				return True
			if w1 > h1 * ratio and h2 > w2 * ratio:
				return False
			if w2 > h2 * ratio and h1 > w1 * ratio:
				return False
			if w1 > h1 * ratio or w2 > h2 * ratio : # h
				return abs(x1 - x2) < char_size * char_gap_tolerance2 or abs(x1 + w1 - (x2 + w2)) < char_size * char_gap_tolerance2
			elif h1 > w1 * ratio or h2 > w2 * ratio : # v
				return abs(y1 - y2) < char_size * char_gap_tolerance2 or abs(y1 + h1 - (y2 + h2)) < char_size * char_gap_tolerance2
			return False
		else:
			return False
	if True:#not a_aa and not b_aa:
		if abs(a.angle - b.angle) < 15 * np.pi / 180:
			fs_a = a.font_size
			fs_b = b.font_size
			fs = min(fs_a, fs_b)
			if a.poly_distance(b) > fs * char_gap_tolerance2:
				return False
			if abs(fs_a - fs_b) / fs > 0.25:
				return False
			return True
	return False

def quadrilateral_can_merge_region_coarse(a: Quadrilateral, b: Quadrilateral, discard_connection_gap = 2, font_size_ratio_tol = 0.7) -> bool:
	if a.assigned_direction != b.assigned_direction:
		return False
	if abs(a.angle - b.angle) > 15 * np.pi / 180:
		return False
	fs_a = a.font_size
	fs_b = b.font_size
	fs = min(fs_a, fs_b)
	if abs(fs_a - fs_b) / fs > font_size_ratio_tol:
		return False
	fs = max(fs_a, fs_b)
	dist = a.poly_distance(b)
	if dist > discard_connection_gap * fs:
		return False
	return True

def findNextPowerOf2(n):
	i = 0
	while n != 0:
		i += 1
		n = n >> 1
	return 1 << i

class Point:
	def __init__(self, x = 0, y = 0):
		self.x = x
		self.y = y

	def length2(self) -> float:
		return self.x * self.x + self.y * self.y

	def length(self) -> float:
		return np.sqrt(self.length2())

	def __str__(self):
		return f'({self.x}, {self.y})'

	def __add__(self, other):
		x = self.x + other.x
		y = self.y + other.y
		return Point(x, y)

	def __sub__(self, other):
		x = self.x - other.x
		y = self.y - other.y
		return Point(x, y)

	def __mul__(self, other):
		if isinstance(other, Point):
			return self.x * other.x + self.y * other.y
		else:
			return Point(self.x * other, self.y * other)

	def __truediv__(self, other):
		return self.x * other.y - self.y * other.x

	def neg(self):
		return Point(-self.x, -self.y)

	def normalize(self):
		return self * (1. / self.length())

def center_of_points(pts: List[Point]) -> Point:
	ans = Point()
	for p in pts:
		ans.x += p.x
		ans.y += p.y
	ans.x /= len(pts)
	ans.y /= len(pts)
	return ans

def support_impl(pts: List[Point], d: Point) -> Point:
	dist = -1.0e-20
	ans = pts[0]
	for p in pts:
		proj = p * d
		if proj > dist:
			dist = proj
			ans = p
	return ans

def support(a: List[Point], b: List[Point], d: Point) -> Point:
	return support_impl(a, d) - support_impl(b, d.neg())

def cross(a: Point, b: Point, c: Point) -> Point:
	return b * (a * c) - a * (b * c)

def closest_point_to_origin(a: Point, b: Point) -> Point:
	da = a.length()
	db = b.length()
	dist = abs(a / b) / (a - b).length()
	ab = b - a
	ba = a - b
	ao = a.neg()
	bo = b.neg()
	if ab * ao > 0 and ba * bo > 0:
		return cross(ab, ao, ab).normalize() * dist
	return a.neg() if da < db else b.neg()

def dcmp(a) -> bool:
	if abs(a) < 1e-8:
		return False
	return True

def gjk_distance(s1: List[Point], s2: List[Point]) -> float:
	d = center_of_points(s2) - center_of_points(s1)
	a = support(s1, s2, d)
	b = support(s1, s2, d.neg())
	d = closest_point_to_origin(a, b)
	s = [a, b]
	for _ in range(8):
		c = support(s1, s2, d)
		a = s.pop()
		b = s.pop()
		da = d * a
		db = d * b
		dc = d * c
		if not dcmp(dc - da) or not dcmp(dc - db):
			return d.length()
		p1 = closest_point_to_origin(a, c)
		p2 = closest_point_to_origin(b, c)
		if p1.length2() < p2.length2():
			s.append(a)
			d = p1
		else:
			s.append(b)
			d = p2
		s.append(c)
	return 0

def color_difference(rgb1: List, rgb2: List) -> float:
	# https://en.wikipedia.org/wiki/Color_difference#CIE76
	color1 = np.array(rgb1, dtype=np.uint8).reshape(1, 1, 3)
	color2 = np.array(rgb2, dtype=np.uint8).reshape(1, 1, 3)
	diff = cv2.cvtColor(color1, cv2.COLOR_RGB2LAB).astype(np.float64) - cv2.cvtColor(color2, cv2.COLOR_RGB2LAB).astype(np.float64)
	diff[..., 0] *= 0.392
	diff = np.linalg.norm(diff, axis=2) 
	return diff.item()

def main():
	s1 = [Point(0, 0), Point(0, 2), Point(2, 2), Point(2, 0)]
	offset = 0
	s2 = [Point(1 + offset, 1 + offset), Point(1 + offset, 3 + offset), Point(3 + offset, 3 + offset + 1.5), Point(3 + offset + 1.5, 3 + offset), Point(3 + offset, 1 + offset)]
	print(gjk_distance(s1, s2))

if __name__ == '__main__':
	main()

