
from typing import List, Union
from utils import Quadrilateral
import numpy as np
import cv2
from utils import findNextPowerOf2
from . import text_render
from .text_render_eng import render_textblock_list_eng
from detection.textblockdetector.textblock import TextBlock

def fg_bg_compare(fg, bg):
	fg_avg = np.mean(fg)
	bg_avg = np.mean(bg)
	if abs(fg_avg - bg_avg) < 40 :
		#bg = None
		#fg = (0, 0, 0) if fg_avg <= 127 else (255, 255, 255)
		bg = (255, 255, 255) if fg_avg <= 127 else (0, 0, 0)
	return fg, bg


async def dispatch(img_canvas: np.ndarray, text_mag_ratio: np.integer, translated_sentences: List[str], textlines: List[Quadrilateral], text_regions: List[Quadrilateral], text_direction_overwrite: str, font_size_offset: int = 0) -> np.ndarray :
	for ridx, (trans_text, region) in enumerate(zip(translated_sentences, text_regions)) :
		if not trans_text :
			continue
		if text_direction_overwrite and text_direction_overwrite in ['h', 'v'] :
			region.majority_dir = text_direction_overwrite
		majority_dir = region.majority_dir

		print(region.text)
		print(trans_text)
		#print(region.majority_dir, region.pts)
		fg = (region.fg_r, region.fg_g, region.fg_b)
		bg = (region.bg_r, region.bg_g, region.bg_b)
		fg, bg = fg_bg_compare(fg, bg)
		font_size = 0
		n_lines = len(region.textline_indices)
		for idx in region.textline_indices :
			txtln = textlines[idx]
			#img_bbox = cv2.polylines(img_bbox, [txtln.pts], True, color = fg, thickness=2)
			# [l1a, l1b, l2a, l2b] = txtln.structure
			# cv2.line(img_bbox, l1a, l1b, color = (0, 255, 0), thickness = 2)
			# cv2.line(img_bbox, l2a, l2b, color = (0, 0, 255), thickness = 2)
			#dbox = txtln.aabb
			font_size = max(font_size, txtln.font_size)
			#cv2.rectangle(img_bbox, (dbox.x, dbox.y), (dbox.x + dbox.w, dbox.y + dbox.h), color = (255, 0, 255), thickness = 2)
		font_size = round(font_size)
		#img_bbox = cv2.polylines(img_bbox, [region.pts], True, color=(0, 0, 255), thickness = 2)

		region_aabb = region.aabb
		print(region_aabb.x, region_aabb.y, region_aabb.w, region_aabb.h)
		img_canvas = render(img_canvas, font_size, text_mag_ratio, trans_text, region, majority_dir, fg, bg, False, font_size_offset)
	return img_canvas


async def dispatch_ctd_render(img_canvas: np.ndarray, text_mag_ratio: np.integer, translated_sentences: List[str], text_regions: List[TextBlock], text_direction_overwrite: str, font_size_offset: int = 0) -> np.ndarray :
	for ridx, (trans_text, region) in enumerate(zip(translated_sentences, text_regions)) :
		print(f'text: {region.get_text()} \n trans: {trans_text}')
		if not trans_text :
			continue
		if text_direction_overwrite and text_direction_overwrite in ['h', 'v'] :
			majority_dir = text_direction_overwrite
		else:
			majority_dir = 'v' if region.vertical else 'h'

		fg, bg = region.get_font_colors()
		fg, bg = fg_bg_compare(fg, bg)
		font_size = region.font_size
		font_size = round(font_size)
		if not isinstance(font_size, int) :
			font_size = int(font_size)

		textlines = []
		for ii, text in enumerate(region.text):
			textlines.append(Quadrilateral(np.array(region.lines[ii]), text, 1, region.fg_r, region.fg_g, region.fg_b, region.bg_r, region.bg_g, region.bg_b))
		# region_aabb = region.aabb
		# print(region_aabb.x, region_aabb.y, region_aabb.w, region_aabb.h)
		img_canvas = render(img_canvas, font_size, text_mag_ratio, trans_text, region, majority_dir, fg, bg, True, font_size_offset)
	return img_canvas


def render(img_canvas, font_size, text_mag_ratio, trans_text, region, majority_dir, fg, bg, is_ctd, font_size_offset: int = 0):
	# round font_size to fixed powers of 2, so later LRU cache can work
	font_size_enlarged = findNextPowerOf2(font_size) * text_mag_ratio
	enlarge_ratio = font_size_enlarged / font_size
	font_size = font_size_enlarged
	#enlarge_ratio = 1
	while True :
		if is_ctd:
			enlarged_w = round(enlarge_ratio * (region.xyxy[2] - region.xyxy[0]))
			enlarged_h = round(enlarge_ratio * (region.xyxy[3] - region.xyxy[1]))
		else:
			enlarged_w = round(enlarge_ratio * region.aabb.w)
			enlarged_h = round(enlarge_ratio * region.aabb.h)
		rows = enlarged_h // (font_size * 1.3)
		cols = enlarged_w // (font_size * 1.3)
		if rows * cols < len(trans_text) :
			enlarge_ratio *= 1.1
			continue
		break
	font_size += font_size_offset
	print('font_size:', font_size)
	if majority_dir == 'h' :
		temp_box = text_render.put_text_horizontal(
			font_size,
			enlarge_ratio * 1.0,
			trans_text,
			enlarged_w,
			fg,
			bg
		)
	else :
		temp_box = text_render.put_text_vertical(
			font_size,
			enlarge_ratio * 1.0,
			trans_text,
			enlarged_h,
			fg,
			bg
		)
	h, w, _ = temp_box.shape
	r_prime = w / h
	
	if is_ctd:
		r = region.aspect_ratio()
		if majority_dir != 'v':
			r = 1 / r
	else:
		r = region.aspect_ratio

	w_ext = 0
	h_ext = 0
	if r_prime > r :
		h_ext = int(w / (2 * r) - h / 2)
		box = np.zeros((h + h_ext * 2, w, 4), dtype=np.uint8)
		box[h_ext:h + h_ext, 0:w] = temp_box
	else :
		w_ext = int((h * r - w) / 2)
		box = np.zeros((h, w + w_ext * 2, 4), dtype=np.uint8)
		box[0:h, w_ext:w_ext+w] = temp_box
	#region_ext = round(min(w, h) * 0.05)
	#h_ext += region_ext
	#w_ext += region_ext
	
	src_pts = np.array([[0, 0], [box.shape[1], 0], [box.shape[1], box.shape[0]], [0, box.shape[0]]]).astype(np.float32)
	#src_pts[:, 0] = np.clip(np.round(src_pts[:, 0]), 0, enlarged_w * 2)
	#src_pts[:, 1] = np.clip(np.round(src_pts[:, 1]), 0, enlarged_h * 2)
	if is_ctd:
		dst_pts = region.min_rect()
		if majority_dir == 'v':
			dst_pts = dst_pts[:, [3, 0, 1, 2]]
	else:
		dst_pts = region.pts

	M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
	rgba_region = np.clip(cv2.warpPerspective(box, M, (img_canvas.shape[1], img_canvas.shape[0]), flags = cv2.INTER_LINEAR, borderMode = cv2.BORDER_CONSTANT, borderValue = 0), 0, 255)
	canvas_region = rgba_region[:, :, 0: 3]
	mask_region = rgba_region[:, :, 3: 4].astype(np.float32) / 255.0
	img_canvas = np.clip((img_canvas.astype(np.float32) * (1 - mask_region) + canvas_region.astype(np.float32) * mask_region), 0, 255).astype(np.uint8)
	return img_canvas




async def dispatch_eng_render(img_canvas: np.ndarray, original_img: np.ndarray, text_regions: Union[List[TextBlock], List[Quadrilateral]], translated_sentences: List[str], font_path: str) -> np.ndarray :
	if len(text_regions) == 0:
		return img_canvas
	
	if isinstance(text_regions[0], 	Quadrilateral):
		blk_list = []
		for region, tr in zip(text_regions, translated_sentences):
			x = np.min(region.pts[:, 0])
			w = np.max(region.pts[:, 0]) - x
			y = np.min(region.pts[:, 1])
			h = np.max(region.pts[:, 1]) - y
			font_size = region.font_size * 0.7
			blk = TextBlock([x, y, w, h], lines=[region.pts], translation=tr, angle=region.angle, font_size=font_size)
			blk_list.append(blk)
		return render_textblock_list_eng(img_canvas, blk_list, font_path, size_tol=1.2, original_img=original_img, downscale_constraint=0.5)
	
	for blk, tr in zip(text_regions, translated_sentences):
		blk.translation = tr
	return render_textblock_list_eng(img_canvas, text_regions, font_path, size_tol=1.2, original_img=original_img, downscale_constraint=0.8)
