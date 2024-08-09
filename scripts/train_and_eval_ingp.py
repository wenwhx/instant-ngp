#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import csv
import os
import pickle
import sys
import time

import commentjson as json
import numpy as np
from tqdm import tqdm

from common import *
from scenes import *

import pyngp as ngp  # noqa


def parse_args():
	parser = argparse.ArgumentParser(description="Run instant neural graphics primitives with additional configuration & output options")

	parser.add_argument("files", nargs="*", help="Files to be loaded. Can be a scene, network config, snapshot, camera path, or a combination of those.")

	parser.add_argument("--scene", "--training_data", default="", help="The scene to load. Can be the scene's name or a full path to the training data. Can be NeRF dataset, a *.obj/*.stl mesh for training a SDF, an image, or a *.nvdb volume.")
	parser.add_argument("--mode", default="", type=str, help=argparse.SUPPRESS) # deprecated
	parser.add_argument("--network", default="", help="Path to the network config. Uses the scene's default if unspecified.")

	parser.add_argument("--load_snapshot", "--snapshot", default="", help="Load this snapshot before training. recommended extension: .ingp/.msgpack")
	parser.add_argument("--save_snapshot", default="", help="Save this snapshot after training. recommended extension: .ingp/.msgpack")
	parser.add_argument("--train_dir", default="", help="Save training output csv and snapshot. recommended snapshot's extension: .msgpack")
	parser.add_argument("--data_dir", default="", help="Path to access data. Only works in the test phase.")

	parser.add_argument("--nerf_compatibility", action="store_true", help="Matches parameters with original NeRF. Can cause slowness and worse results on some scenes, but helps with high PSNR on synthetic scenes.")
	parser.add_argument("--test_transforms", default="", help="Path to a nerf style transforms json from which we will compute PSNR.")
	parser.add_argument("--near_distance", default=-1, type=float, help="Set the distance from the camera at which training rays start for nerf. <0 means use ngp default")
	parser.add_argument("--exposure", default=0.0, type=float, help="Controls the brightness of the image. Positive numbers increase brightness, negative numbers decrease it.")

	parser.add_argument("--screenshot_transforms", default="", help="Path to a nerf style transforms.json from which to save screenshots.")
	parser.add_argument("--screenshot_frames", nargs="*", help="Which frame(s) to take screenshots of.")
	parser.add_argument("--screenshot_dir", default="", help="Which directory to output screenshots to.")
	parser.add_argument("--screenshot_spp", type=int, default=16, help="Number of samples per pixel in screenshots.")

	parser.add_argument("--video_camera_path", default="", help="The camera path to render, e.g., base_cam.json.")
	parser.add_argument("--video_camera_smoothing", action="store_true", help="Applies additional smoothing to the camera trajectory with the caveat that the endpoint of the camera path may not be reached.")
	parser.add_argument("--video_fps", type=int, default=60, help="Number of frames per second.")
	parser.add_argument("--video_n_seconds", type=int, default=1, help="Number of seconds the rendered video should be long.")
	parser.add_argument("--video_render_range", type=int, nargs=2, default=(-1, -1), metavar=("START_FRAME", "END_FRAME"), help="Limit output to frames between START_FRAME and END_FRAME (inclusive)")
	parser.add_argument("--video_spp", type=int, default=8, help="Number of samples per pixel. A larger number means less noise, but slower rendering.")
	parser.add_argument("--video_output", type=str, default="video.mp4", help="Filename of the output video (video.mp4) or video frames (video_%%04d.png).")

	parser.add_argument("--save_mesh", default="", help="Output a marching-cubes based mesh from the NeRF or SDF model. Supports OBJ and PLY format.")
	parser.add_argument("--marching_cubes_res", default=256, type=int, help="Sets the resolution for the marching cubes grid.")

	parser.add_argument("--width", "--screenshot_w", type=int, default=0, help="Resolution width of GUI and screenshots.")
	parser.add_argument("--height", "--screenshot_h", type=int, default=0, help="Resolution height of GUI and screenshots.")

	parser.add_argument("--gui", action="store_true", help="Run the testbed GUI interactively.")
	parser.add_argument("--train", action="store_true", help="If the GUI is enabled, controls whether training starts immediately.")
	parser.add_argument("--n_steps", type=int, default=-1, help="Number of steps to train for before quitting.")
	parser.add_argument("--second_window", action="store_true", help="Open a second window containing a copy of the main output.")
	parser.add_argument("--vr", action="store_true", help="Render to a VR headset.")

	parser.add_argument("--sharpen", default=0, help="Set amount of sharpening applied to NeRF training images. Range 0.0 to 1.0.")

	parser.add_argument("--save_optimizer_state", action="store_true", help="Save the optimizer state when saving a snapshot")
	parser.add_argument("--save_image_detail", action="store_true", help="Save error image.")
	parser.add_argument("--write_image", action="store_true", help="Write recon image.")

	parser.add_argument("--enable_da", action="store_true", help="Enable density-aware NeRF ensembles.")
	parser.add_argument("--seed", type=int, default=1337, help="Set the rng seed for testbed if density-aware NeRF ensembles are enabled.")

	return parser.parse_args()

def get_scene(scene):
	for scenes in [scenes_sdf, scenes_nerf, scenes_image, scenes_volume]:
		if scene in scenes:
			return scenes[scene]
	return None

def run_ngp(args_ngp):
	args = args_ngp

	if args.vr: # VR implies having the GUI running at the moment
		args.gui = True

	if args.mode:
		print("Warning: the '--mode' argument is no longer in use. It has no effect. The mode is automatically chosen based on the scene.")

	testbed = ngp.Testbed()
	testbed.root_dir = ROOT_DIR

	for file in args.files:
		scene_info = get_scene(file)
		if scene_info:
			file = os.path.join(scene_info["data_dir"], scene_info["dataset"])
		testbed.load_file(file)

	if args.scene:
		scene_info = get_scene(args.scene)
		if scene_info is not None:
			args.scene = os.path.join(scene_info["data_dir"], scene_info["dataset"])
			if not args.network and "network" in scene_info:
				args.network = scene_info["network"]

		testbed.load_training_data(args.scene)

	if args.gui:
		# Pick a sensible GUI resolution depending on arguments.
		sw = args.width or 1920
		sh = args.height or 1080
		while sw * sh > 1920 * 1080 * 4:
			sw = int(sw / 2)
			sh = int(sh / 2)
		testbed.init_window(sw, sh, second_window=args.second_window)
		if args.vr:
			testbed.init_vr()

	if args.enable_da:
		testbed.rng_seed = int(args.seed)

	if args.load_snapshot:
		scene_info = get_scene(args.load_snapshot)
		if scene_info is not None:
			args.load_snapshot = default_snapshot_filename(scene_info)
		testbed.load_snapshot(args.load_snapshot)
	elif args.network:
		testbed.reload_network_from_file(args.network)

	ref_transforms = {}
	if args.screenshot_transforms: # try to load the given file straight away
		print("Screenshot transforms from ", args.screenshot_transforms)
		with open(args.screenshot_transforms) as f:
			ref_transforms = json.load(f)

	if testbed.mode == ngp.TestbedMode.Sdf:
		testbed.tonemap_curve = ngp.TonemapCurve.ACES

	testbed.nerf.sharpen = float(args.sharpen)
	testbed.exposure = args.exposure
	testbed.shall_train = args.train if args.gui else True


	testbed.nerf.render_with_lens_distortion = True

	network_stem = os.path.splitext(os.path.basename(args.network))[0] if args.network else "base"
	if testbed.mode == ngp.TestbedMode.Sdf:
		setup_colored_sdf(testbed, args.scene)

	if args.near_distance >= 0.0:
		print("NeRF training ray near_distance ", args.near_distance)
		testbed.nerf.training.near_distance = args.near_distance

	if args.nerf_compatibility:
		print(f"NeRF compatibility mode enabled")

		# Prior nerf papers accumulate/blend in the sRGB
		# color space. This messes not only with background
		# alpha, but also with DOF effects and the likes.
		# We support this behavior, but we only enable it
		# for the case of synthetic nerf data where we need
		# to compare PSNR numbers to results of prior work.
		testbed.color_space = ngp.ColorSpace.SRGB

		# No exponential cone tracing. Slightly increases
		# quality at the cost of speed. This is done by
		# default on scenes with AABB 1 (like the synthetic
		# ones), but not on larger scenes. So force the
		# setting here.
		testbed.nerf.cone_angle_constant = 0

		# Match nerf paper behaviour and train on a fixed bg.
		# testbed.nerf.training.random_bg_color = False

	old_training_step = 0
	n_steps = args.n_steps

	# If we loaded a snapshot, didn't specify a number of steps, _and_ didn't open a GUI,
	# don't train by default and instead assume that the goal is to render screenshots,
	# compute PSNR, or render a video.
	if n_steps < 0 and (not args.load_snapshot or args.gui):
		n_steps = 35000

	tqdm_last_update = 0
	training_output_list = []
	training_output_header = ['iter', 'loss_type', 'loss']
	if n_steps > 0:
		with tqdm(desc="Training", total=n_steps, unit="steps") as t:
			while testbed.frame():
				if testbed.want_repl():
					repl(testbed)
				# What will happen when training is done?
				if testbed.training_step >= n_steps:
					if args.gui:
						testbed.shall_train = False
					else:
						break

				# Update progress bar
				if testbed.training_step < old_training_step or old_training_step == 0:
					old_training_step = 0
					t.reset()

				now = time.monotonic()
				if now - tqdm_last_update > 10:
					t.update(testbed.training_step - old_training_step)
					t.set_postfix(loss=testbed.loss)
					old_training_step = testbed.training_step
					tqdm_last_update = now
					
				if testbed.training_step % 10 == 0:
					training_output_list.append([
						testbed.training_step,
						testbed.nerf.training.loss_type,
						testbed.loss
					])

	if args.train_dir:
		# save snapshot
		cur_iter = testbed.training_step
		cur_save_snapshot = os.path.join(args.train_dir, "snapshot.ingp")
		if args.save_snapshot:
			cur_save_snapshot = args.save_snapshot
		print("Saving snapshot ", cur_save_snapshot)
		if args.save_optimizer_state:
			testbed.save_snapshot(cur_save_snapshot, True)
		else:
			testbed.save_snapshot(cur_save_snapshot, False)

		# # save used train_transform.json
		# if args.add_one_view:
		# 	for s in generated_json_list:
		# 		shutil.copy(s, )
		# shutil.copy(scene, cur_train_dir)

		# save train output csv
		cur_train_csv = os.path.join(args.train_dir, "train_data.csv")
		with open(cur_train_csv, "w", newline="") as train_output:
			train_output_writer = csv.writer(train_output, delimiter=" ",
											quotechar="|", quoting=csv.QUOTE_MINIMAL)

			train_output_writer.writerow(training_output_header)
			for row in training_output_list:
				train_output_writer.writerow(row)
	elif args.save_snapshot:
		testbed.save_snapshot(args.save_snapshot, False)

	if args.test_transforms:
		print("Evaluating test transforms from ", args.test_transforms)
		with open(args.test_transforms) as f:
			test_transforms = json.load(f)
		data_dir=os.path.dirname(args.test_transforms)
		totmse = 0
		totpsnr = 0
		totssim = 0
		totcount = 0
		minpsnr = 1000
		maxpsnr = 0

		# Evaluate metrics on black background
		testbed.background_color = [0.0, 0.0, 0.0, 1.0]
		if args.enable_da:
			testbed.background_color = [0.0, 0.0, 0.0, 0.0]

		# Prior nerf papers don't typically do multi-sample anti aliasing.
		# So snap all pixels to the pixel centers.
		testbed.snap_to_pixel_centers = True
		spp = 8

		testbed.nerf.render_min_transmittance = 1e-4

		testbed.shall_train = False
		testbed.load_training_data(args.test_transforms)
		
		test_results = []
		test_result_fields = ["test_view_id", "psnr", "ssim", "ref_masked_mse", "full_masked_mse", "recon_image"]
		details, render_images = [], []
		with tqdm(range(testbed.nerf.training.dataset.n_images), unit="images", desc=f"Rendering test frame") as t:
			for i in t:
				resolution = testbed.nerf.training.dataset.metadata[i].resolution
				testbed.render_ground_truth = True
				testbed.set_camera_to_training_view(i)
				ref_image = testbed.render(resolution[0], resolution[1], 1, True)
				testbed.render_ground_truth = False
				image = testbed.render(resolution[0], resolution[1], spp, True)

				# save all images to npy file as a cache; without evaluate psnrs, ssims
				if args.enable_da:
					render_images.append(image)
					continue

				if i == 0:
					diffimg = np.absolute(image - ref_image)
					diffimg[...,3:4] = 1.0

				A = np.clip(linear_to_srgb(image[...,:3]), 0.0, 1.0)
				R = np.clip(linear_to_srgb(ref_image[...,:3]), 0.0, 1.0)
				mse = float(compute_error("MSE", A, R))
				ssim = float(compute_error("SSIM", A, R))
				totssim += ssim
				totmse += mse
				psnr = mse2psnr(mse)
				totpsnr += psnr
				minpsnr = psnr if psnr<minpsnr else minpsnr
				maxpsnr = psnr if psnr>maxpsnr else maxpsnr
				totcount = totcount+1
				t.set_postfix(psnr = totpsnr/(totcount or 1))
				
				# save rendered image
				render_image_name = "r_{}.png".format(str(i))
				render_path = os.path.join(args.screenshot_dir, render_image_name)
				if args.write_image:
					write_image(render_path, image)

				# masked MSE; mask can be full mask and ref-only mask
				mse_map = compute_error_img("MSE", A, R)
				mse_map = np.mean(mse_map, axis=2, keepdims=True)

				mask = np.nonzero(ref_image[..., 3:4])
				ref_masked_mse = np.mean(mse_map[mask])

				mask = np.logical_or(image[..., 3:4], ref_image[..., 3:4])
				mask = np.nonzero(mask)
				full_masked_mse = np.mean(mse_map[mask])

				# save error image
				if args.save_image_detail:
					# avg error
					mae_map = compute_error_img("MAE", A, R)
					mae_map = np.mean(mae_map, axis=2, keepdims=True)

					# pixel-wise psnr
					mse_map = compute_error_img("MSE", A, R)
					mse_map = np.mean(mse_map, axis=2, keepdims=True)
					psnr_map = mse2psnr(mse_map + 1e-10)

					# shape: 	[H, W, 8]
					# ch 0-3: 	g.t. image, range [0.0, 1.0]
					# ch 4-7: 	pred image, range [0.0, 1.0]
					# ch 8: 	avg. error, np.absolute(img - ref_img), range [0.0, 1.0]
					# ch 9: 	pixel-wise PSNR, range [0.0, 99.9999]
					detail = np.concatenate([R, ref_image[..., 3:4], A, image[..., 3:4], mae_map, psnr_map], axis=-1)
					details.append(detail)

				test_results.append({
					"test_view_id" : i,
					"psnr" : psnr,
					"ssim" : ssim,
					"ref_masked_mse": ref_masked_mse,
					"full_masked_mse": full_masked_mse,
					"recon_image" : render_image_name
				})

		psnr_avgmse = mse2psnr(totmse/(totcount or 1))
		psnr = totpsnr/(totcount or 1)
		ssim = totssim/(totcount or 1)
		print(f"PSNR={psnr} [min={minpsnr} max={maxpsnr}] SSIM={ssim}")

		test_csv = os.path.join(args.screenshot_dir, "test_data.csv")
		psnrs, ssims, full_masked_mses, ref_masked_mses, recon_images = [], [], [], [], []
		with open(test_csv, "w", newline="") as test_output:
			test_output_writer = csv.DictWriter(test_output, fieldnames=test_result_fields, delimiter=' ')
			test_output_writer.writeheader()
			for r in test_results:
				test_output_writer.writerow(r)
				psnrs.append(r['psnr'])
				ssims.append(r['ssim'])
				ref_masked_mses.append(r['ref_masked_mse'])
				full_masked_mses.append(r['full_masked_mse'])
				recon_images.append(r['recon_image'])

		if args.save_image_detail:
			image_detail_path = os.path.join(args.screenshot_dir, "details.npy")
			np.save(image_detail_path, np.array(details))
		
		# save cache images
		if args.enable_da:
			images_path = os.path.join(args.screenshot_dir, "recon_images_seed_{}.npy".format(str(args.seed)))
			np.save(images_path, np.array(render_images))
		
		return psnr, ssim, psnrs, ssims, full_masked_mses, ref_masked_mses, recon_images


