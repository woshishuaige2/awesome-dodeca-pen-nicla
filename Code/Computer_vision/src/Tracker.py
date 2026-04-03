import numpy as np
from src.DoDecahedronUtils import *
import cv2.aruco as aruco


def object_tracking(frame, params, text_data, post, show_markers=1):
	# print(frame.shape)
	# convert rgb frame into gray-scale
	frame_gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
	frame_gray_draw = np.copy(frame_gray)

	# detect markers, return corners uv position and id of each marker
	# corners: N-length tuple of (1, 4, 2), ids: (N, 1)

	corners, ids, _ = aruco.detectMarkers(frame_gray, params.aruco_dict, parameters=params.aruco_params)
	
	# Refine corners for better accuracy if markers are detected
	if ids is not None and len(ids) > 0:
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
		for i in range(len(corners)):
			cv2.cornerSubPix(frame_gray, corners[i], (5, 5), (-1, -1), criteria)

	visib_flag = 1

	# if ids is not None:
	# 	print(ids)
	# ensure at least two markers in the list are detected to reconstruct the transform
	if ids is not None and np.isin(ids, params.markers_possible).all() and len(ids) >= 2:
		# stacked_corners_px_sp: (4N, 2)
		stacked_corners_px_sp = np.reshape(np.asarray(corners), (ids.shape[0] * 4, 2))

		# Save rotation and translation vectors for each marker
		# Maximum number of markers is 13
		marker_rot_vec = np.zeros((13, 1, 3))
		marker_trans_vec = np.zeros((13, 1, 3))

		# Index when saving results, different from ids
		marker_index = 0

		# the following are with the camera frame
		center_transform_martix_set = np.zeros((ids.shape[0], 4, 4))

		# For each marker detected, find its coordinate transform
		for marker_id in ids:

			marker_rot_vec[marker_id, :, :], marker_trans_vec[marker_id, :, :], _ = cv2.aruco.estimatePoseSingleMarkers(corners[marker_index],
																					params.marker_size_in_mm, params.mtx, params.dist)
			# TODO: Camera matrix Calibration   params.mtx, params.dist

			if show_markers:
				# Draw boundary of detected markers
				frame = aruco.drawDetectedMarkers(frame, corners)

			# Input: rt vector in (6,), Output: Transform matrix for one Aruco code in (4,4)
			marker_transform_matrix = RodriguesToTransf(np.append(marker_rot_vec[marker_id, :, :], marker_trans_vec[marker_id, :, :]))

			# Query pre-defined marker-center coordinate transform based on marker id in shape (4,4)
			center_marker_transform, marker_center_transform = tf_mat_dodeca_pen(int(marker_id))

			# Calculate marker-wise center coordinate transform
			center_transform_martix_set[marker_index, :, :] = np.matmul(marker_transform_matrix, marker_center_transform)

			marker_index += 1

		# Remove bad estimation and calculate average center rotation
		# (M, 4, 4), (3, 3), M-length list
		marker_transform_accepted, center_rotation_matrix, marker_accepted_indices = remove_bad_aruco_centers(center_transform_martix_set, params)

		# Get average center coordinate transform by averaging accepted marker transform in shape (4,4)
		center_coordinate_transform = find_tfmat_avg(marker_transform_accepted)

		# Convert center coordinate transform into Rodrigues form in shape (1,6)
		# This is the raw results with no post-processing
		center_rot_rod, _ = cv2.Rodrigues(center_coordinate_transform[0:3, 0:3])
		center_trans_rod = center_coordinate_transform[0:3, 3]
		center_transform_rod = np.append(center_rot_rod, center_trans_rod.reshape((3, 1))).reshape((6, 1))
		center_pose_rod_raw = center_transform_rod.T
		final_pose = center_pose_rod_raw

		if post == 1 or post == 2:  # 0 means raw pose, 1 means pose refined by APE, 2 means pose refined by DPR
			center_pose_ape = leastsq(LM_APE_Dodecapen, center_transform_rod, Dfun=None, full_output=False,
										col_deriv=False, ftol=1.49012e-6, xtol=1.49012e-4, gtol=0.0, maxfev=1000,
										epsfcn=None, factor=1, diag=None, args=(stacked_corners_px_sp, ids, params, False))[0]
			# This is the results with APE
			center_pose_rod_ape = np.reshape(center_pose_ape, (1, 6))
			final_pose = center_pose_rod_ape
			#########################################################################################
			if post == 2:
				############################          DPR refinement         ############################
				#########################################################################################
				b_edge, edge_intensities_expected = marker_edges(ids, text_data, params)
				LM_DPR_DRAW(center_pose_ape, frame_gray_draw, ids, corners, b_edge, edge_intensities_expected,
							text_data, params, 250, 2)
				center_pose_dpr = leastsq(LM_DPR, center_pose_ape, Dfun=LM_DPR_Jacobian, full_output=True, col_deriv=False,
										ftol=1.49012e-10, xtol=1.49012e-4, gtol=0.0,maxfev=1000, epsfcn=None, factor=1, diag=None,
										args=(frame_gray, ids, corners, b_edge, edge_intensities_expected, text_data, params))[0]
				LM_DPR_DRAW(center_pose_dpr, frame_gray_draw, ids, corners, b_edge, edge_intensities_expected,
							text_data, params, 0, 1)
				center_pose_rod_dpr = np.reshape(center_pose_dpr, (1, 6))
				final_pose = center_pose_rod_dpr
				#########################################################################################

	else:
		# print("\rRequired marker not visible")
		final_pose = None
		visib_flag = 0

	return final_pose
