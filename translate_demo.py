
import asyncio
import argparse
from PIL import Image
import cv2
import numpy as np
import requests
import os
import torch
import huggingface_hub
from oscrypto import util as crypto_utils

from detection import DETECTORS, dispatch as dispatch_detection
from ocr import dispatch as dispatch_ocr, load_model as load_ocr_model
from inpainting import dispatch as dispatch_inpainting, load_model as load_inpainting_model
from translators import TRANSLATORS, VALID_LANGUAGES, dispatch as run_translation
from text_mask import dispatch as dispatch_mask_refinement
from textline_merge import dispatch as dispatch_textline_merge
from text_rendering import dispatch as dispatch_rendering, text_render
from detection.textblockdetector.textblock import visualize_textblocks
from utils import convert_img

parser = argparse.ArgumentParser(description='Generate text bboxes given a image file')
parser.add_argument('--mode', default='demo', type=str, help='Run demo in either single image demo mode (demo), web service mode (web) or batch translation mode (batch)')
parser.add_argument('--image', default='', type=str, help='Image file if using demo mode or Image folder name if using batch mode')
parser.add_argument('--image-dst', default='', type=str, help='Destination folder for translated images in batch mode')
parser.add_argument('--size', default=1536, type=int, help='image square size')
parser.add_argument('--host', default='127.0.0.1', type=str, help='Used by web module to decide which host to attach to')
parser.add_argument('--port', default=5003, type=int, help='Used by web module to decide which port to attach to')
parser.add_argument('--log-web', action='store_true', help='Used by web module to decide if web logs should be surfaced')
parser.add_argument('--ocr-model', default='48px_ctc', type=str, help='OCR model to use, one of `32px`, `48px_ctc`')
parser.add_argument('--use-inpainting', action='store_true', help='turn on/off inpainting')
parser.add_argument('--inpainting-model', default='lama_mpe', type=str, help='inpainting model to use, one of `lama_mpe`')
parser.add_argument('--use-cuda', action='store_true', help='turn on/off cuda')
parser.add_argument('--force-horizontal', action='store_true', help='force texts rendered horizontally')
parser.add_argument('--inpainting-size', default=2048, type=int, help='size of image used for inpainting (too large will result in OOM)')
parser.add_argument('--unclip-ratio', default=2.3, type=float, help='How much to extend text skeleton to form bounding box')
parser.add_argument('--box-threshold', default=0.7, type=float, help='threshold for bbox generation')
parser.add_argument('--text-threshold', default=0.5, type=float, help='threshold for text detection')
parser.add_argument('--text-mag-ratio', default=1, type=int, help='text rendering magnification ratio, larger means higher quality')
parser.add_argument('--font-size-offset', default=0, type=int, help='offset font size by a given amount, positive number increase font size and vice versa')
parser.add_argument('--translator', default='google', type=str, choices=TRANSLATORS, help='language translator')
parser.add_argument('--target-lang', default='CHS', type=str, choices=VALID_LANGUAGES, help='destination language')
parser.add_argument('--detector', default='default', type=str, choices=DETECTORS, help='text detector used for creating a text mask from an image')
parser.add_argument('--verbose', action='store_true', help='print debug info and save intermediate images')
parser.add_argument('--manga2eng', action='store_true', help='render English text translated from manga with some typesetting')
parser.add_argument('--eng-font', default='fonts/comic shanns 2.ttf', type=str, help='font used by manga2eng mode')
args = parser.parse_args()

def update_state(task_id, nonce, state) :
	while True :
		try :
			requests.post(f'http://{args.host}:{args.port}/task-update-internal', json = {'task_id': task_id, 'nonce': nonce, 'state': state}, timeout = 20)
			return
		except Exception :
			if 'error' in state or 'finished' in state :
				continue
			else :
				break

def get_task(nonce) :
	try :
		rjson = requests.get(f'http://{args.host}:{args.port}/task-internal?nonce={nonce}', timeout = 3600).json()
		if 'task_id' in rjson and 'data' in rjson :
			return rjson['task_id'], rjson['data']
		else :
			return None, None
	except Exception :
		return None, None

async def infer(
	img,
	mode,
	nonce,
	options = None,
	task_id = '',
	dst_image_name = '',
	alpha_ch = None
	) :
	options = options or {}
	img_detect_size = args.size
	if 'size' in options :
		size_ind = options['size']
		if size_ind == 'S' :
			img_detect_size = 1024
		elif size_ind == 'M' :
			img_detect_size = 1536
		elif size_ind == 'L' :
			img_detect_size = 2048
		elif size_ind == 'X' :
			img_detect_size = 2560
	print(f' -- Detection resolution {img_detect_size}')

	if 'detector' in options :
		detector = options['detector']
	else:
		detector = args.detector
	print(f' -- Detector using {detector}')

	render_text_direction_overwrite = 'h' if args.force_horizontal else ''
	if 'direction' in options :
		if options['direction'] == 'horizontal' :
			render_text_direction_overwrite = 'h'
	print(f' -- Render text direction is {render_text_direction_overwrite or "auto"}')

	if mode == 'web' and task_id:
		update_state(task_id, nonce, 'detection')

	raw_mask, final_mask, textlines = await dispatch_detection(detector, img, img_detect_size, args.use_cuda, args, verbose = args.verbose)

	if args.verbose :
		cv2.imwrite(f'result/{task_id}/mask_raw.png', raw_mask)
		if detector == 'ctd' :
			bboxes = visualize_textblocks(cv2.cvtColor(img,cv2.COLOR_BGR2RGB), textlines)
			cv2.imwrite(f'result/{task_id}/bbox.png', bboxes)
		else:
			img_bbox_raw = np.copy(img)
			for txtln in textlines :
				cv2.polylines(img_bbox_raw, [txtln.pts], True, color = (255, 0, 0), thickness = 2)
			cv2.imwrite(f'result/{task_id}/bbox_unfiltered.png', cv2.cvtColor(img_bbox_raw, cv2.COLOR_RGB2BGR))

	if mode == 'web' and task_id :
		update_state(task_id, nonce, 'ocr')
	textlines = await dispatch_ocr(img, textlines, args.use_cuda, args, model_name = args.ocr_model, verbose = args.verbose)

	if detector == 'ctd':
		text_regions = textlines
	elif detector == 'default' :
		text_regions, textlines = await dispatch_textline_merge(textlines, img.shape[1], img.shape[0], verbose = args.verbose)
		if args.verbose :
			img_bbox = np.copy(img)
			for region in text_regions :
				for idx in region.textline_indices :
					txtln = textlines[idx]
					cv2.polylines(img_bbox, [txtln.pts], True, color = (255, 0, 0), thickness = 2)
				img_bbox = cv2.polylines(img_bbox, [region.pts], True, color = (0, 0, 255), thickness = 2)
			cv2.imwrite(f'result/{task_id}/bbox.png', cv2.cvtColor(img_bbox, cv2.COLOR_RGB2BGR))

		print(' -- Generating text mask')
		if mode == 'web' and task_id :
			update_state(task_id, nonce, 'mask_generation')
		# create mask
		final_mask = await dispatch_mask_refinement(img, raw_mask, textlines)

	if mode == 'web' and task_id :
		update_state(task_id, nonce, 'translating')
		# in web mode, we can start translation task async
		if detector == 'ctd':
			requests.post(f'http://{args.host}:{args.port}/request-translation-internal', json = {'task_id': task_id, 'nonce': nonce, 'texts': [r.get_text() for r in text_regions]}, timeout = 20)
		else:
			requests.post(f'http://{args.host}:{args.port}/request-translation-internal', json = {'task_id': task_id, 'nonce': nonce, 'texts': [r.text for r in text_regions]}, timeout = 20)

	print(' -- Running inpainting')
	if mode == 'web' and task_id :
		update_state(task_id, nonce, 'inpainting')
	# run inpainting
	if text_regions :
		img_inpainted = await dispatch_inpainting(args.use_inpainting, False, args.use_cuda, img, final_mask, args.inpainting_size, verbose = args.verbose)
	else :
		img_inpainted = img
	if args.verbose :
		cv2.imwrite(f'result/{task_id}/inpaint_input.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
		cv2.imwrite(f'result/{task_id}/inpainted.png', cv2.cvtColor(img_inpainted, cv2.COLOR_RGB2BGR))
		cv2.imwrite(f'result/{task_id}/mask_final.png', final_mask)

	# translate text region texts
	translated_sentences = None
	print(' -- Translating')
	if mode != 'web' :
		# try:
		if detector == 'ctd' :
			translated_sentences = await run_translation(args.translator, 'auto', args.target_lang, [r.get_text() for r in text_regions], args.use_cuda)
		else:
			translated_sentences = await run_translation(args.translator, 'auto', args.target_lang, [r.text for r in text_regions], args.use_cuda)

	else :
		translation_request_timeout = 20

		# wait for at most 1 hour for manual translation
		if 'manual' in options and options['manual'] :
			wait_n_10ms = 36000
		# Wait longer for offline translation given it needs to do some crunching
		elif 'offline' in args.translator:
			translation_request_timeout = 60
			wait_n_10ms = 1200 
		else :
			wait_n_10ms = 300 # 30 seconds for machine translation
		for _ in range(wait_n_10ms) :
			ret = requests.post(f'http://{args.host}:{args.port}/get-translation-result-internal', json = {'task_id': task_id, 'nonce': nonce}, timeout = translation_request_timeout).json()
			if 'result' in ret :
				translated_sentences = ret['result']
				if isinstance(translated_sentences, str) :
					if translated_sentences == 'error' :
						update_state(task_id, nonce, 'error-lang')
						return
				break
			await asyncio.sleep(0.01)

	print(' -- Rendering translated text')
	if translated_sentences == None:
		if mode == 'web' and task_id :
			print("No text found!")
			update_state(task_id, nonce, 'error-no-txt')
		return
	
	if mode == 'web' and task_id :
		update_state(task_id, nonce, 'render')
	# render translated texts
	if args.target_lang == 'ENG' and args.manga2eng:
		from text_rendering import dispatch_eng_render
		output = await dispatch_eng_render(np.copy(img_inpainted), img, text_regions, translated_sentences, args.eng_font)
	else:
		if detector == 'ctd' :
			from text_rendering import dispatch_ctd_render
			output = await dispatch_ctd_render(np.copy(img_inpainted), args.text_mag_ratio, translated_sentences, text_regions, render_text_direction_overwrite, args.font_size_offset)
		else:
			output = await dispatch_rendering(np.copy(img_inpainted), args.text_mag_ratio, translated_sentences, textlines, text_regions, render_text_direction_overwrite, args.font_size_offset)
	
	print(' -- Saving results')
	if alpha_ch is not None :
		output = np.concatenate([output.astype(np.uint8), np.array(alpha_ch).astype(np.uint8)[..., None]], axis = 2)
	else :
		output = output.astype(np.uint8)
	img_pil = Image.fromarray(output)
	if dst_image_name :
		img_pil.save(dst_image_name)
	else :
		img_pil.save(f'result/{task_id}/final.png')

	if mode == 'web' and task_id :
		update_state(task_id, nonce, 'finished')


async def infer_safe(
	img,
	mode,
	nonce,
	options = None,
	task_id = '',
	dst_image_name = '',
	alpha_ch = None
	) :
	try :
		return await infer(
			img,
			mode,
			nonce,
			options,
			task_id,
			dst_image_name,
			alpha_ch
		)
	except :
		import traceback
		traceback.print_exc()
		update_state(task_id, nonce, 'error')

def replace_prefix(s: str, old: str, new: str) :
	if s.startswith(old) :
		s = new + s[len(old):]
	return s

async def main(mode = 'demo') :
	print(' -- Preload Checks')
	# Add failsafe if torch cannot find cuda support
	if not torch.cuda.is_available() and args.use_cuda:
		print("Warning: CUDA compatible device could not be found while --use_cuda args was set... Deferring to CPU")
		args.use_cuda = False

	print(' -- Preload optional models')
	preload_offline_translator(args.translator)

	print(' -- Loading models')
	os.makedirs('result', exist_ok = True)
	text_render.prepare_renderer()
	with open('alphabet-all-v5.txt', 'r', encoding = 'utf-8') as fp :
		dictionary = [s[:-1] for s in fp.readlines()]
	load_ocr_model(dictionary, args.use_cuda, args.ocr_model)
	load_inpainting_model(args.use_cuda, args.inpainting_model)

	if mode == 'demo' :
		print(' -- Running in single image demo mode')
		if not args.image :
			print('please provide an image')
			parser.print_usage()
			return
		img, alpha_ch = convert_img(Image.open(args.image))
		img = np.array(img)
		await infer(img, mode, '', alpha_ch = alpha_ch)
	elif mode == 'web' :
		print(' -- Running in web service mode')
		print(' -- Waiting for translation tasks')
		nonce = crypto_utils.rand_bytes(16).hex()
		import subprocess
		import sys

		extra_web_args = {'stdout':sys.stdout, 'stderr':sys.stderr} if args.log_web else {}
		web_executable = [sys.executable, '-u'] if args.log_web else [sys.executable]
		web_process_args = ['web_main.py', nonce, str(args.host), str(args.port)]
		subprocess.Popen([*web_executable, *web_process_args], **extra_web_args)
		
		while True :
			try :
				task_id, options = get_task(nonce)
				if task_id :
					try :
						print(f' -- Processing task {task_id}')
						img, alpha_ch = convert_img(Image.open(f'result/{task_id}/input.png'))
						img = np.array(img)
						infer_task = asyncio.create_task(infer_safe(img, mode, nonce, options, task_id, alpha_ch = alpha_ch))
						asyncio.gather(infer_task)
					except Exception :
						import traceback
						traceback.print_exc()
						update_state(task_id, nonce, 'error')
				else :
					await asyncio.sleep(0.1)
			except Exception :
				import traceback
				traceback.print_exc()
	elif mode == 'web2' :
		print(' -- Running in web service mode')
		print(' -- Waiting for translation tasks')
		while True :
			task_id, options = get_task(nonce)
			if task_id :
				print(f' -- Processing task {task_id}')
				try :
					img, alpha_ch = convert_img(Image.open(f'result/{task_id}/input.png'))
					img = np.array(img)
					infer_task = asyncio.create_task(infer_safe(img, mode, nonce, options, task_id, alpha_ch = alpha_ch))
					asyncio.gather(infer_task)
				except Exception :
					import traceback
					traceback.print_exc()
					update_state(task_id, nonce, 'error')
			else :
				await asyncio.sleep(0.1)
	elif mode == 'batch' :
		src = os.path.abspath(args.image)
		if src[-1] == '\\' or src[-1] == '/' :
			src = src[:-1]
		dst = args.image_dst or src + '-translated'
		if os.path.exists(dst) and not os.path.isdir(dst) :
			print(f'Destination `{dst}` already exists and is not a directory! Please specify another directory.')
			return
		if os.path.exists(dst) and os.listdir(dst) :
			print(f'Destination directory `{dst}` already exists! Please specify another directory.')
			return
		print('Processing image in source directory')
		files = []
		for root, subdirs, files in os.walk(src) :
			dst_root = replace_prefix(root, src, dst)
			os.makedirs(dst_root, exist_ok = True)
			for f in files :
				if f.lower() == '.thumb' :
					continue
				filename = os.path.join(root, f)
				try :
					img, alpha_ch = convert_img(Image.open(filename))
					img = np.array(img)
					if img is None :
						continue
				except Exception :
					pass
				try :
					dst_filename = replace_prefix(filename, src, dst)
					print('Processing', filename, '->', dst_filename)
					await infer(img, 'demo', '', dst_image_name = dst_filename, alpha_ch = alpha_ch)
				except Exception :
					import traceback
					traceback.print_exc()
					pass


def preload_offline_translator(translation_mode):
	# Use Facebook No Language Left Behind Model to enable offline translation
	translation_model_map = {
		"offline": "facebook/nllb-200-distilled-600M",
		"offline_big": "facebook/nllb-200-distilled-1.3B",
	}

	if translation_mode in translation_model_map.keys():
		nllb_model_name = translation_model_map[args.translator]
		

		# Get if repo exists in cache
		if not huggingface_hub.try_to_load_from_cache(nllb_model_name, 'pytorch_model.bin') is None:
			print(f"Detected cached model for offline translation: {nllb_model_name}")
			return
		
		# Preload models into cache as part of startup
		print(f"Detected offline translation mode. Pre-loading offline translation model: {nllb_model_name} " +
		       "(This can take a long time as multiple GB's worth of data can be downloaded during this step)")
		huggingface_hub.snapshot_download(nllb_model_name)



if __name__ == '__main__':
	print(args)
	loop = asyncio.get_event_loop()
	loop.run_until_complete(main(args.mode))
