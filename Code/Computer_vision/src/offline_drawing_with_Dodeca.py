''' This is for live drawing with dodeca pen'''

import DoDecahedronUtils  as dodecapen
import numpy as np
from numpy import linalg as LA
import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 
from mpl_toolkits.mplot3d import Axes3D
import transforms3d as tf3d
import time
from scipy.interpolate import griddata

from scipy.optimize import minimize, leastsq,least_squares
from scipy import linalg
from scipy.spatial import distance
from matplotlib.path import Path
import multiprocessing as mp

from pynput import keyboard
import threading
from filter import OneEuroFilter

def main():
	plot_switch = 1
	hist_plot_switch = 0
	iterations_for_while =5500
	data = dodecapen.txt_data()
	params = dodecapen.parameters()

	tip_loc_cent_local = np.array([0.0,0.0,15,1]).reshape(4,1) # need to do pen tip calibration    #local coordinate system of face 9 with respect to pen tip
	tip_face_id = 9  #pen inserted into face #9														#
	Tf_cent_face9, Tf_face_cent9 = dodecapen.tf_mat_dodeca_pen(tip_face_id)							#
	tip_loc_cent = np.matmul(Tf_cent_face9,tip_loc_cent_local)										#from pen tip to dodeca center
	print(tip_loc_cent)

	#tip_loc_cent = np.array([19.03108466, 131.31450883, -72.69986037,1]).reshape(4,1)	
	# the tip_loc_cent is the result of tip calibration,
	# use tip_calibration.py to calibrate the tip coorindate in the Docecahedron's frame
	tip_loc_cent = np.array([-1.4259097, 138.2282741, -83.86195491, 1]).reshape(4,1)
	#[ 8.40682847e-02  1.38090962e+02 -8.24473826e+01]
	#[-4.63769333e-02  1.42520472e+02 -8.89559608e+01]


	pose_marker_without_opt = np.zeros((10000,6))  # 6DOF using average pose obtained from Aruco
	pose_marker_with_APE = np.zeros((10000,6))     # 6DOF using approximate pose estimation (nPn using corners)
	pose_marker_with_DPR = np.zeros((10000,6))     # 6DOF using dense pose estimation (nPn using visible feature points on markers)
	tip_position = np.zeros((10000,3))             # at 30fps, can analyze about 5 minutes video
	tip = np.zeros((10000,6))

	# live video pen tracker 
	# cap = cv2.VideoCapture(0)  # use 0 to wake the default camera
	# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
	# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
	# cap.set(cv2.CAP_PROP_FPS,15)

	# cap = cv2.VideoCapture(0)
	# cap.set(cv2.CAP_PROP_FPS,30)
				
	cap = cv2.VideoCapture(r'./src/test2.mov')          # pen tracker from recorded video

	frame_width = int(cap.get(3))
	frame_height = int(cap.get(4))  
	size = (frame_width, frame_height)
	print(size)

	result = cv2.VideoWriter('test_res.mp4', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         30, size)
	
	# Initialize the filter variables with a placeholder
	first_valid_pose = False
	filter_x, filter_y, filter_z = None, None, None

	idx = 0 # Tip position data index
	j = 0
	while (j<iterations_for_while):  #cap.isOpened():
	# while(True):
		ret, frame = cap.read()
		# cv2.imshow("test", frame)
		if not ret:
			break
		if frame is None:
			time.sleep(0.1)
			print("No image")
			continue		
		frame_gray_draw,pose_without_opt, pose_APE,pose_DPR,visib_flag = dodecapen.find_pose(frame,params,data) 
		if visib_flag == 1:
			pose_marker_with_APE[j,:] = pose_APE
			pose_marker_without_opt[j,:] = pose_without_opt
			pose_marker_with_DPR[j,:] = pose_DPR
			tf_cam_to_cent = dodecapen.RodriguesToTransf(pose_DPR)
			tip_loc_cam = tf_cam_to_cent.dot(tip_loc_cent)
   
			# Initialize the filter with the first valid pose
			if not first_valid_pose:
				t_start = time.time()
				filter_x = OneEuroFilter(t_start, tip_loc_cam[0, 0])
				filter_y = OneEuroFilter(t_start, tip_loc_cam[1, 0])
				filter_z = OneEuroFilter(t_start, tip_loc_cam[2, 0])
				first_valid_pose = True
    
			# Apply the filter to each coordinate
			current_time = time.time()
			filtered_x = filter_x.filter_signal(current_time, tip_loc_cam[0, 0])
			filtered_y = filter_y.filter_signal(current_time, tip_loc_cam[1, 0])
			filtered_z = filter_z.filter_signal(current_time, tip_loc_cam[2, 0])
   
			# Create a new, filtered tip vector
			filtered_tip_tvec = np.array([[filtered_x, filtered_y, filtered_z]])

			# Store the filtered data
			tip_position[idx,:] = filtered_tip_tvec.reshape(3,)
   
			# Convert the filtered 3D point to 2D pixel coordinates for drawing
			tip_pix, _ = cv2.projectPoints(filtered_tip_tvec, np.zeros((3,1)), np.zeros((3,1)),
											params.mtx, params.dist)

			center = tuple(np.ndarray.astype(tip_pix[0,0],int))

			# Draw the filtered pen tip on the frame
			frame = cv2.circle(frame, center, 5, (0, 255, 0), -1)

			# Draw the unfiltered point for comparison
			unfiltered_tip_pix, _ = cv2.projectPoints(tip_loc_cam[0:3].reshape(1,3), np.zeros((3,1)), np.zeros((3,1)),
													params.mtx, params.dist)
			unfiltered_center = tuple(np.ndarray.astype(unfiltered_tip_pix[0,0],int))
			frame = cv2.circle(frame, unfiltered_center, 5, (0, 0, 255), -1)

			print("frame number ", j)
			cv2.imshow('AR Pen tracking', frame)
			result.write(frame)

			idx += 1

		j +=1
		if cv2.waitKey(1) & 0xFF == ord('q') or j >= iterations_for_while:
			print("STOP")
			break

	pose_marker_without_opt = pose_marker_without_opt[0:j,:]
	pose_marker_with_APE = pose_marker_with_APE[0:j,:]
	pose_marker_with_DPR = pose_marker_with_DPR[0:j,:]
	tip_position = tip_position[0:j,:]

	cap.release()
	result.release()

	cv2.destroyAllWindows()

	if plot_switch == 1 : 
		r2d = 180/np.pi
		### translation
		fig = plt.figure()

		ax = fig.add_subplot(111, projection="3d")
		fig.canvas.manager.set_window_title("translation x,y,zq") 

		ax.set_xlabel('X Label')
		ax.set_ylabel('Y Label')
		ax.set_zlabel('Z Label')
		ax.scatter(pose_marker_without_opt[:,3],pose_marker_without_opt[:,4],pose_marker_without_opt[:,5],
							c ='m',label = "pose_marker_without_opt")
		ax.scatter(pose_marker_with_APE[:,3],pose_marker_with_APE[:,4],pose_marker_with_APE[:,5],
							c = 'r',label="pose_marker_with_APE" )
		ax.scatter(pose_marker_with_DPR[:,3],pose_marker_with_DPR[:,4],pose_marker_with_DPR[:,5],
							c = 'g',label="pose_marker_with_DPR" )
		ax.legend()
		
		### rotation
		# fig = plt.figure()
		# fig.canvas.set_window_title("rotation x,y,z") 

		# ax.set_xlabel('X Label')
		# ax.set_ylabel('Y Label')
		# ax.set_zlabel('Z Label')
		# ax = fig.add_subplot(111, projection="3d")
		# ax.scatter(pose_marker_without_opt[:,0]*r2d, pose_marker_without_opt[:,1]*r2d, pose_marker_without_opt[:,2]*r2d,
		# 					c ='m',label = "orientation_marker_without_opt")
		# ax.scatter(pose_marker_with_APE[:,0]*r2d, pose_marker_with_APE[:,1]*r2d, pose_marker_with_APE[:,2]*r2d,
		# 					c = 'r',label="orientation_marker_with_APE" )
		# ax.scatter(pose_marker_with_DPR[:,0]*r2d, pose_marker_with_DPR[:,1]*r2d, pose_marker_with_DPR[:,2]*r2d,
		# 					c = 'g',label="orientation_marker_with_DPR" )
		# ax.legend()


		### tip 
		fig = plt.figure()
		fig.canvas.manager.set_window_title("tip x,y,z") 

		ax = fig.add_subplot(111, projection="3d")
		ax.set_xlabel('X Label')
		ax.set_ylabel('Y Label')
		ax.set_zlabel('Z Label')
		ax.scatter(tip_position[:,0], tip_position[:,1], tip_position[:,2],
							c ='k',label = "tip_position")
	 

		if hist_plot_switch == 1:
			## translation 	
			fig = plt.figure()
			fig.canvas.set_window_title("histogram translation z") 
			plt.hist(pose_marker_without_opt[:,5],j,facecolor='magenta',normed = 1,label = 'pose_marker_without_opt' )
			plt.hist(pose_marker_with_APE[:,5],j,facecolor='red',normed = 1, label = 'pose_marker_with_APE'  )
			plt.hist(pose_marker_with_DPR[:,5],j,facecolor='green',normed = 1, label = 'pose_marker_with_DPR'  )
			plt.legend()
			
			## rotation
			# fig = plt.figure()
			# fig.canvas.set_window_title("histogram rotation z") 
			# plt.hist(pose_marker_without_opt[:,2]*r2d,j,facecolor='magenta',normed = 1,label = 'orientation_marker_without_opt' )
			# plt.hist(pose_marker_with_APE[:,2]*r2d,j,facecolor='red',normed = 1, label = 'orientation_marker_with_APE'  )
			# plt.hist(pose_marker_with_DPR[:,2]*r2d,j,facecolor='green',normed = 1, label = 'orientation_marker_with_DPR'  )
			# plt.legend()

			print ("the end")

	plt.show()


if __name__ == '__main__':
	main()