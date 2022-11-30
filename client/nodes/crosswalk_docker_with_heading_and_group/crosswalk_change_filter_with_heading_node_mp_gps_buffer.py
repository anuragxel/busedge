# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

import logging

import cv2
from protocol.busedge_protocol import busedge_pb2
from gabriel_protocol import gabriel_pb2
from crosswalk_filter import CrosswalkFilter

logger = logging.getLogger(__name__)

import argparse
import multiprocessing
import time

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image, NavSatFix
from std_msgs.msg import UInt8MultiArray
import pandas as pd
from collections import deque
import numpy as np
import copy
import pandas as pd
from datetime import datetime
import time
import os
import pytz
from utils.exif_gps import set_gps_location
from exif import Image as exifImage
import signal 
import sys
import bisect

# check buffer 0.00011348724365234375
# record frame 0.03262662887573242
# viewable 0.0011057853698730469
# append 4.172325134277344e-05

# check buffer 0.00011682510375976562
# record frame 0.12095427513122559
# viewable 0.0008840560913085938
# append 4.601478576660156e-05


send_to_server = True
on_bus = False
undistort_flag = True
CUR_GPS = NavSatFix()
length_deque = 3
prev_gps = deque(maxlen=length_deque)
heading = 45
prev_heading = 45
gps_threshold = 0.0001
skip = False
skip_threshold = 8.333333340715399e-06
ref_long_lat = np.array([0,0])
group_counter = 0
record_all_frames = True
stop = False
dont_detect = True
tz_NY = pytz.timezone('US/Eastern')
use_buffer = False
if use_buffer:
	time_buffer = [0]
	gps_buffer = [(CUR_GPS, heading, skip)]

class faux_cur_gps:
	def __init__(self, long, lat, alt):
		self.longitude = long
		self.latitude = lat
		self.altitude = alt

def get_lat_long_heading(img_path):
	with open(img_path, 'rb') as image_file:
		my_image = exifImage(image_file)
	lat = my_image.gps_latitude
	lat = float(lat[0]) + float(lat[1])/60 + float(lat[2])/(60*60)
	long = my_image.gps_longitude
	long = float(long[0]) + float(long[1])/60 + float(long[2])/(60*60)
	alt = my_image.gps_altitude
	heading = my_image.gps_img_direction
	#negative long from empirical evidence
	curr_gps = faux_cur_gps(-long, lat, alt)
	return curr_gps, heading

class img_object:
	def __init__(self, image, long, lat, altitude, heading, filename, det_flag, viewable_gt_flag, cw_assign):
		self.image = image
		self.long = long
		self.lat = lat
		self.altitude = altitude
		self.heading = heading
		self.filename = filename
		self.det_flag = det_flag
		self.viewable_gt_flag = viewable_gt_flag
		self.cw_assign = cw_assign
		self.buffer = ''
		
cam_dict = {
	'1':{'intrinsic':np.array([[747.356409, 0, 645.293182], [0, 747.794094, 381.808991], [0, 0, 1]]),
		 'distortion': np.array([[-0.330800, 0.118200, -0.000123, -0.000015, 0]])
		}, 
	'2':{'intrinsic':np.array([[739.900213, 0, 661.033245], [0, 740.043216, 364.130756], [0, 0, 1]]),
		 'distortion': np.array([[-0.337600, 0.128700, -0.000734, -0.000153, 0]])
		}, 
	'3':{'intrinsic':  np.array([[760.562, 0, 626.0370], [0, 765.198, 366.22], [0, 0, 1]]),
		 'distortion': np.array([[-0.299167, 0.066771, -0.000216, 0.000593, 0]])
		}, 
	'4':{'intrinsic':np.array([[737.685910, 0, 647.747645], [0, 738.361414, 374.783301], [0, 0, 1]]),
		 'distortion': np.array([[-0.339900, 0.146500, -0.000634, -0.000571, 0]])
		}, 
	'5':{'intrinsic':np.array([[741.564472, 0, 669.000132], [0, 742.110750, 374.558612], [0, 0, 1]]),
		 'distortion': np.array([[-0.317100, 0.092000, -0.000535, -0.000608, 0]])
		},         
	}

def undistort(cam_id, cv_img):
	h, w = cv_img.shape[:2]
	mtx = cam_dict[str(cam_id)]['intrinsic']
	dist = cam_dict[str(cam_id)]['distortion']
	newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
	# undistort
	dst = cv2.undistort(cv_img, mtx, dist, None, newcameramtx)
	# crop the image
	x, y, w, h = roi
	dst = dst[y:y+h, x:x+w]
	return dst

def viewable_gt(CUR_GPS, df, heading, method = 'viewed_location'):
	"""
	x = np.array([40.25802333,    -80.19472667])
	df = pd.read_csv('/home/xinhez/Downloads/gabriel-BusEdge/client/nodes/crosswalk_docker_with_heading_and_group/gps.csv')
	:param CUR_GPS: DESCRIPTION
	:type CUR_GPS: TYPE
	:param df: DESCRIPTION
	:type df: TYPE
	:return: DESCRIPTION
	:rtype: TYPE

	"""
	bus = np.array([CUR_GPS.latitude, CUR_GPS.longitude])
	bus = bus[None, :]
	cw = df[['Latitude', 'Longitude']].to_numpy()
	valid = df[['inspect']].to_numpy()
	cw[valid.flatten() != 1] = -50
	cw_label = np.array([label for label in range(len(cw))])
	
	if method == 'aligned_angle':
		#gps long lat comparison
		idx1 = np.linalg.norm(bus - cw, axis = 1) < .0005
		dist = np.linalg.norm(bus - cw, axis = 1)
	
		#orientation comparision
		#60 degree range
		vec = bus - cw
		vec = vec / np.linalg.norm(vec, axis = 1)[:, None]
		idx2 = (vec @ np.array([np.cos(heading/360 * 2 * np.pi), np.sin(heading/360 * 2 * np.pi)])) > 0.5
		idx = idx1 * idx2
	elif method == 'location':
		#gps long lat comparison
		idx = np.linalg.norm(bus - cw, axis = 1)  < .0005        
		dist = np.linalg.norm(bus - cw, axis = 1)  

	elif method == 'viewed_location':
		direction = np.array([[np.cos(heading/360 * 2 * np.pi), np.sin(heading/360 * 2 * np.pi)]])
		direction = direction / np.linalg.norm(direction)
		#gps long lat comparison
		idx = np.linalg.norm(bus + 0.0002 * direction - cw, axis = 1)  < .0005        
		dist = np.linalg.norm(bus + 0.0002 * direction - cw, axis = 1)  
	elif method == 'viewed_location_ellipse':
		theta = heading/360 * 2 * np.pi
		direction = np.array([[np.cos(theta), np.sin(theta)]])
		direction = direction / np.linalg.norm(direction)        
		cw = cw - (bus + 0.0002 * direction)
		bus = bus - bus
	
		rotation = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])        
		
		cw = (rotation @ cw.T).T
		a = 0.0003
		b = 0.0001
		idx = (cw[:, 0] ** 2 / (a ** 2) + cw[:, 1] ** 2 / (b ** 2)) < 1
		dist = cw[:, 0] ** 2 / (a ** 2) + cw[:, 1] ** 2 / (b ** 2)
		
	if sum(idx) > 0:
		viewable_gt_flag = True
		dist = dist[idx]
		cw_label = cw_label[idx]
		min_idx = np.argmin(dist)
		cw_assignment = cw_label[min_idx]
	else:
		viewable_gt_flag = False
		cw_assignment = -1
	return viewable_gt_flag, cw_assignment
		
def create_filename(camera_name, image):
	secs = image.header.stamp.secs
	nsecs = image.header.stamp.nsecs
	time_stamps = "_{:0>10d}_{:0>9d}".format(secs, nsecs)
	image_filename = camera_name + time_stamps + ".jpg"
	return image_filename



def check_buffer(img_gps_buffer, df, cw_assign_np_li, dont_detect = True, buffer = 1, release = False):
	if dont_detect:
		print(len(img_gps_buffer))
		oldest = np.array([img_gps_buffer[0].lat, img_gps_buffer[0].long])
		newest = np.array([img_gps_buffer[-1].lat, img_gps_buffer[-1].long])
		
		#if the range of the buffer is longer than the distance of a block, we check
		if np.linalg.norm(newest - oldest) > 0.0009 or release:
			# viewable = np.array([i.viewable_gt_flag for i in img_gps_buffer])
			# cw_assign = np.array([i.cw_assign for i in img_gps_buffer])
			
			#check if anything is seen
			if max(cw_assign_np_li) > -1:
				#viewable crosswalk in the buffer
				cw = cw_assign_np_li[cw_assign_np_li != -1]
				#get the first crosswalk we see
				cw = int(cw[0])
				cw_idx = np.argwhere(cw_assign_np_li == cw).flatten()
				print('cw', cw)
				cw_last_seen = np.array([img_gps_buffer[cw_idx[-1]].lat, img_gps_buffer[cw_idx[-1]].long])
				#wait until the bus is some distance away from the crosswalk. also determined by release if the bus stopped moving for too long
				if not release and np.linalg.norm(newest - cw_last_seen) < 0.0003: #0.0009:
					send_flag = False
					batch = []
					cw_assign = -1
					change = None
					return send_flag, batch, img_gps_buffer, cw_assign, change, cw_assign_np_li      
				start = int(cw_idx[0])
				end = int(cw_idx[-1]) + 1                   
				#check if it actually came close to the crosswalk
				
				bus = np.array([[x.lat, x.long] for x in img_gps_buffer[start:end]])
				cw_gps = df[['Latitude', 'Longitude']].to_numpy()[cw:cw+1]
				dist = np.linalg.norm(bus - cw_gps, axis = 1)
				closest = np.argmin(dist)

				#check if the bus went through the intersection, so that it's not just a different street that's close
				if min(dist) > 0.0003:
					print('here')
					send_flag = False
					batch = []
					cw_assign = -1
					change = None
					img_gps_buffer = img_gps_buffer[end:]
					cw_assign_np_li = cw_assign_np_li[end:]
					return send_flag, batch, img_gps_buffer, cw_assign, change, cw_assign_np_li               
				else:
					send_flag = True

					batch = [img_gps_buffer[i] for i in range(start, end)]
					#update the buffer
					img_gps_buffer = img_gps_buffer[end:]
					cw_assign_np_li = cw_assign_np_li[end:]

					cw_assign = cw
					#if we know there is a known crosswalk to be expected, then there can only a verification or a removal
					change = 0
					return send_flag, batch, img_gps_buffer, cw_assign, change, cw_assign_np_li
			else:
				#unimportant buffer
				send_flag = False
				batch = []
				cw_assign = -1
				change = None
				img_gps_buffer = []
				cw_assign_np_li = np.array([])
				return send_flag, batch, img_gps_buffer, cw_assign, change, cw_assign_np_li
					   
		else:
			#wait until buffer is longer
			send_flag = False
			batch = []
			cw_assign = -1
			change = None
			return send_flag, batch, img_gps_buffer, cw_assign, change, cw_assign_np_li
		   
	
	else:
		#first decide the core 
		#then add padding
		#paddings can get a cw_assign of -2
		if len(img_gps_buffer) > 20:
			#focus only on the non buffer images
			detections = np.array([i.det_flag for i in img_gps_buffer[buffer:-buffer]])
			viewable = np.array([i.viewable_gt_flag for i in img_gps_buffer[buffer:-buffer]])
			cw_assign = np.array([i.cw_assign for i in img_gps_buffer[buffer:-buffer]])
		
			if max(cw_assign) == -1:
				#if there are no viewable crosswalks, then we should only expect one appearance of a crosswalk at most
				#because we don't expect too much change to happen at once
				det_idx = np.argwhere(detections == True)
		
				#check if there are detections
				if len(det_idx) >= 2 and len(det_idx)/(det_idx[1] - det_idx[0]) > 0.5:
					start = int(det_idx[0])
					end = int(det_idx[-1]) + 1     
					#wait until detection is more centered
					if end == len(detections) - 1:
						send_flag = False
						batch = []
						cw_assign = -1
						change = None
						return send_flag, batch, img_gps_buffer, cw_assign, change                 
					else: 
					#send the new crosswalk group
						send_flag = True
						batch = [img_gps_buffer[i] for i in range(start + buffer - buffer, end + buffer + buffer)]
						img_gps_buffer = img_gps_buffer[end + buffer - buffer:]
						for i in range(buffer):
							img_gps_buffer[i].det_flag = False
							img_gps_buffer[i].viewable_gt_flag = False
							img_gps_buffer[i].cw_assign = -1
							
						 
						cw_assign = -1
						change = 1
						return send_flag, batch, img_gps_buffer, cw_assign, change  
				else:
					#nothing substantial detected, so move forward
					send_flag = False
					# if len(det_idx) > 0:
					#     end = int(det_idx[-1])
					# else:
					end = int(buffer * 2) + 1
					
					batch = []
					img_gps_buffer = img_gps_buffer[end + buffer - buffer:]
					for i in range(buffer + buffer):
						img_gps_buffer[i].det_flag = False
						img_gps_buffer[i].viewable_gt_flag = False
						img_gps_buffer[i].cw_assign = -1   
					cw_assign = -1
					change = None
					return send_flag, batch, img_gps_buffer, cw_assign, change
			else:
				#viewable crosswalk in the buffer
				idx = np.logical_or(viewable, detections)
				cw = cw_assign[cw_assign != -1]
				cw = cw[0]
				cw_idx = np.argwhere(cw_assign == cw)
				
				#wait until detection is more centered
				if cw_idx[-1] == len(detections) - 1:
					send_flag = False
					batch = []
					cw_assign = -1
					change = None
					return send_flag, batch, img_gps_buffer, cw_assign, change      
				else:
					send_flag = True
					
					first_seen_idx = np.argwhere(idx == True)
		
					
					start = int(first_seen_idx[0])
					end = int(cw_idx[-1]) + 1                   
					
					#send the batch and then reset the metadata
					batch = [img_gps_buffer[i] for i in range(start + buffer - buffer, end + buffer + buffer)]
					img_gps_buffer = img_gps_buffer[end + buffer - buffer:]
					for i in range(buffer):
						img_gps_buffer[i].det_flag = False
						img_gps_buffer[i].viewable_gt_flag = False
						img_gps_buffer[i].cw_assign = -1
					cw_assign = cw
					#if we know there is a known crosswalk to be expected, then there can only a verification or a removal
					if max(detections[start:end]) == 1:
						change = 0
					else:
						change = -1
					return send_flag, batch, img_gps_buffer, cw_assign, change
		else:
			#wait until buffer is longer
			send_flag = False
			batch = []
			cw_assign = -1
			change = None
			return send_flag, batch, img_gps_buffer, cw_assign, change

	
def extract_batch_metadata(batch, send_flag, stat_df, cw_assign, change):
	global group_counter
	
	i = batch[0]
	frame = i.image
	longitude = i.long
	latitude = i.lat
	altitude = i.altitude
	heading = i.heading
	file_name = i.filename
	det_flag = i.det_flag 
	viewable_gt_flag = i.viewable_gt_flag
	# cw_assign = i.cw_assign 
	
	stats_dic = {}
	stats_dic['file_name'] = file_name
	stats_dic['time'] = datetime.fromtimestamp(int(file_name.split('_')[1]), pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S')
	# stats_dic['time'] = datetime.fromtimestamp(int(image.header.stamp.secs)).strftime('%Y-%m-%d %H:%M:%S')
	stats_dic['Latitude'] = round(latitude, 5)
	stats_dic['Longitude'] = round(longitude, 5)
	stats_dic['Heading'] = round(heading, 1)
	stats_dic['detection'] = det_flag
	stats_dic['seen gt'] = viewable_gt_flag
	stats_dic['sent'] = send_flag
	stats_dic['change'] = change
	# if cw_assign == -1:
	#     cw_assign = group_counter
	stats_dic['cw_assign'] = cw_assign
	stat_df = stat_df.append(pd.DataFrame([stats_dic]))  
	stat_df.to_csv(datetime.now(tz_NY).strftime("%Y_%m_%d") + '_stats.csv', index=False)

def record_batch_info(stat_df, batch, stats_dic, cw_assign, change): 
	rospy.loginfo(batch[-1].filename)
	try:
		stats_dic['batch_start_idx'] = stat_df.index[stat_df['file_name'] ==batch[0].filename].values[0]
	except:
		stats_dic['batch_start_idx'] = -1
		rospy.loginfo('did not find ' + batch[0].filename)
	try:
		stats_dic['batch_stop_idx'] = stat_df.index[stat_df['file_name'] ==batch[-1].filename].values[0]
	except:
		stats_dic['batch_stop_idx'] = -1
		rospy.loginfo('did not find ' + batch[-1].filename)

	stats_dic['cw'] = cw_assign 
	stats_dic['change'] = change 
	return stats_dic

def record_frame(image_filename, det_flag, viewable_gt_flag, cw_assignment, latitude, longitude, heading, send_flag):
	stats_dic = {}
	stats_dic['file_name'] = image_filename
	stats_dic['detection'] = int(det_flag)
	stats_dic['seen gt'] = int(viewable_gt_flag)
	stats_dic['cw_assignment gt'] = int(cw_assignment)
	stats_dic['time'] = datetime.fromtimestamp(int(image_filename.split('_')[1]), pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S')
	stats_dic['Latitude'] = round(latitude, 5)
	stats_dic['Longitude'] = round(longitude, 5)
	stats_dic['Heading'] = round(heading, 1)
	stats_dic['sent'] = send_flag
	return stats_dic
	
def encode_batch(batch, input_frame, engine_fields, cw_assign):
	global group_counter
	raw_img_folder = "./sent_batches" + '/' + datetime.now(tz_NY).strftime("%Y_%m_%d/")

	input_frame.payload_type = gabriel_pb2.PayloadType.IMAGE
	for i in batch:
		frame = i.image
		longitude = i.long
		latitude = i.lat
		altitude = i.altitude
		heading = i.heading
		file_name = i.filename

		det_flag = i.det_flag 
		viewable_gt_flag = i.viewable_gt_flag
		buffer = i.buffer
 
		#add more description to the filename
		if buffer == 'buffer':
			split = os.path.splitext(file_name)
			file_name = split[0] + '_buffer' + split[1]
		if det_flag:
			split = os.path.splitext(file_name)
			file_name = split[0] + '_detected' + split[1]
		if viewable_gt_flag:
			#this i.cw_assign is different from the cw_assign above. i.cw_assign will be the individual frame assignment
			split = os.path.splitext(file_name)
			file_name = split[0] + '_saw_{}_cw'.format(i.cw_assign) + split[1]
	
		_, jpeg_frame = cv2.imencode(".jpg", frame[:, :, ::-1])
	
		
		input_frame.payloads.append(jpeg_frame.tostring())
		engine_fields.latitude_batch.append(latitude)
		engine_fields.longitude_batch.append(longitude)
		engine_fields.altitude_batch.append(altitude)
		engine_fields.heading_batch.append(int(heading))
		engine_fields.image_filename_batch.append(file_name)
		engine_fields.cw_assign_batch.append(cw_assign)

	if not on_bus:
		group = 'cw_' + str(int(cw_assign)) + '_' + os.path.splitext(engine_fields.image_filename_batch[0])[0]
		# os.makedirs(raw_img_folder + "%05d" % (group_counter), exist_ok=True)
		os.makedirs(raw_img_folder + group, exist_ok=True)
		for idx, i in enumerate(batch):
			frame = i.image
			file_name = engine_fields.image_filename_batch[idx]
			# file_path = os.path.join(raw_img_folder + "%05d" % (group_counter), file_name)
			file_path = os.path.join(raw_img_folder + group, file_name)
			cv2.imwrite(file_path, frame)
			# shutil.copy(img_path, config.sfm_query_path.format(group))
			set_gps_location(
				file_path,
				engine_fields.latitude_batch[idx],
				engine_fields.longitude_batch[idx],
				engine_fields.altitude_batch[idx],
				engine_fields.heading_batch[idx]
				)               

	group_counter += 1
	return input_frame, engine_fields

def img_callback(image, args):
	# print('image', image.header.stamp)
	# print(int(str(image.header.stamp)))
	# print('start')
	
	# print(CUR_GPS.latitude)

	global undistort_flag
	# if use_buffer:
	#     global gps_buffer
	#     global time_buffer
	#     index = bisect.bisect_left(time_buffer, int(str(image.header.stamp)))
	#     index -= 1
	#     # print('index ', index)
	#     # print('len gps buffer', len(gps_buffer))
	#     # print(time_buffer)
	#     # print([x[1] for x in gps_buffer])
	#     (CUR_GPS, heading, skip) = gps_buffer[index]
	#     gps_buffer = gps_buffer[index - 1:]
	#     time_buffer = time_buffer[index - 1:]
	#     # print(len(time_buffer))
	# else:
	global CUR_GPS
	global heading
	global skip        
	# time.sleep(3)
	# rospy.loginfo("heading {:.2f}".format(heading))

	camera_name = args[0]
	camera_id = int(camera_name[-1])

	bridge = CvBridge()
	frame = bridge.compressed_imgmsg_to_cv2(
		image, desired_encoding="passthrough"
	)  # BGR images
	if not skip:
	
		if undistort_flag:
			frame = undistort(camera_id, frame)
		# print(CUR_GPS.latitude)
		frame_copy = frame.copy()

		file_name = create_filename(camera_name, image)

		# os.makedirs(raw_img_folder + "%05d" % (group_counter), exist_ok=True)
		raw_img_folder = "./raw_images" + '/' + datetime.now(tz_NY).strftime("%Y_%m_%d/")
		os.makedirs(raw_img_folder, exist_ok=True)
		# file_path = os.path.join(raw_img_folder + "%05d" % (group_counter), file_name)
		file_path = os.path.join(raw_img_folder, file_name)
		# time.sleep(2)
		# time.sleep(1)
		cv2.imwrite(file_path, frame)    
		# print(CUR_GPS.latitude)

		set_gps_location(
			file_path,
			CUR_GPS.latitude,
			CUR_GPS.longitude,
			CUR_GPS.altitude,
			heading
			) 
	# print('end')




def skip_img(ref_long_lat, img, timestamp, skip_threshold):
	"""
	skip an image if it's too similar to the previous or there's an error in the gps

	:param ref_long_lat: previous gps array
	:type ref_long_lat: npy array
	:param img: current gps array
	:type img: npy array
	:param skip_threshold: distance
	:type skip_threshold: float
	:return: update the previous gps location and skip
	:rtype: npy array, boolean
	"""
	#bewteen 7am and 8pm
	# print(timestamp)
	# print(int(timestamp))
	# print(datetime.fromtimestamp(int(timestamp)).strftime('%H'))
	# print('done')


	if not 6 < int(datetime.fromtimestamp(int(float(timestamp)), pytz.timezone('US/Eastern') ).strftime('%H')) <20:
		skip = True
	elif np.linalg.norm(ref_long_lat - img) <= skip_threshold or min(img == np.array([0, 0])):
		skip = True
	else:
		skip = False
		ref_long_lat = img
	return ref_long_lat, skip
		
def determine_heading(img, length_deque, prev_gps, prev_heading):
	if len(prev_gps) > length_deque - 1:
		heading = img - prev_gps[0]


		if all(img == 0):
				#refer to previous heading
			heading = prev_heading       
		else:
			heading = heading / np.linalg.norm(heading)
			angle = np.arctan2(heading[1], heading[0])
			if angle < 0: 
				angle += 2 * np.pi
			heading = angle/(2 * np.pi) * 360
			prev_heading = heading
	else:
		heading = 45
		prev_heading = 45

	if len(prev_gps) > 0:
		   #only include the gps if it's passed a threshold, otherwise, don't update the 
		if np.linalg.norm(img - prev_gps[-1]) > gps_threshold:
				prev_gps.append(img)
	else:
		prev_gps.append(img)
	return prev_gps, prev_heading, heading
			
def gps_callback(data):
	# print('gps', data.header.stamp)
	global CUR_GPS
	global prev_gps
	global heading
	global prev_heading
	global length_deque
	global skip
	global skip_threshold
	global ref_long_lat
	global use_buffer
	if use_buffer:
		global gps_buffer
		global time_buffer
	# time.sleep(2)

	if data.status.status == -1:
		rospy.logdebug("Crosswalk filter node cannot get valid GPS data")
	else:
		CUR_GPS = data
		img = np.array([CUR_GPS.latitude, CUR_GPS.longitude])

		ref_long_lat, skip = skip_img(ref_long_lat, img, str(data.header.stamp.to_sec()), skip_threshold)

		prev_gps, prev_heading, heading = determine_heading(img, length_deque, prev_gps, prev_heading)
		
		if use_buffer:
			gps_buffer.append((CUR_GPS, heading, skip))
			time_buffer.append(int(str(data.header.stamp)))


def run_node(camera_name):
	rospy.init_node(camera_name + "_crosswalk_filter_node")
	rospy.loginfo("Initialized node crosswalk_filter for " + camera_name)      
		
	cam_id = camera_name[-1]
	image_sub = rospy.Subscriber(
		camera_name + "/image_raw/compressed",
		CompressedImage,
		img_callback,
		callback_args=(camera_name,),
		queue_size=1,
		buff_size=2 ** 24,
	)
	gps_sub = rospy.Subscriber("/fix", NavSatFix, gps_callback, queue_size=1)
	rospy.spin()   
	

def publish_to_gabriel(camera_name):
	rospy.init_node("publish_to_gabriel")
	rospy.loginfo("Initialized node to publish to gabriel")      
			
	global stop
	cam_id = camera_name[-1]
	if dont_detect == False:
		model_dir = "./model/crosswalk_detector/model_final.pth"  # running dir: gabriel-BusEdge/client/
		min_score_thresh = 0.5
		model = CrosswalkFilter(model_dir, min_score_thresh)    
	pub = rospy.Publisher("crosswalk_filter" + cam_id, UInt8MultiArray, queue_size=1)
	df = pd.read_csv('gps.csv')
	img_gps_buffer = []
	cw_assign_np_li = np.array([])
	
	try:    
		stat_df = pd.read_csv(datetime.now(tz_NY).strftime("%Y_%m_%d/") + '_stats.csv')
	except:
		stat_df = pd.DataFrame()    
	
	src = "./raw_images" + '/' + datetime.now(tz_NY).strftime("%Y_%m_%d/")
	error_dir = "./raw_images" + '/' + datetime.now(tz_NY).strftime("%Y_%m_%d_") + 'error/'
	os.makedirs(src, exist_ok=True)
	while stop == False:
		image_queue = sorted(os.listdir(src))
		#keep last few images because the deletion is faster than the saving
		if len(image_queue) > 5:
			while len(image_queue) > 5:
				img_name = image_queue.pop(0)
				img_path = os.path.join(src, img_name)
	
				frame_copy = cv2.imread(img_path)
				try:
					CUR_GPS, heading = get_lat_long_heading(img_path)
				except:
					os.makedirs(error_dir, exist_ok=True)
					os.rename(img_path, os.path.join(error_dir, img_name))  
					rospy.loginfo("error reading " + img_path)            
					continue
				if len(image_queue) > 0:
					#get the heading of the next image in cases of turns. 
					try:
						_, heading = get_lat_long_heading(os.path.join(src, image_queue[0]))
					except:
						os.makedirs(error_dir, exist_ok=True)
						os.rename(img_path, os.path.join(error_dir, image_queue[0]))  
						rospy.loginfo("error reading " + img_path) 
						continue
				os.remove(img_path)  
				if dont_detect == False:
					output_dict = model.detect(frame_copy)
					det_flag = output_dict["num_detections"] > 0
				else:
					det_flag = False
						
				# start = time.time()
				# is there one in view. (also used in cognitive engine to query for images in the image bank)
				viewable_gt_flag, cw_assignment = viewable_gt(CUR_GPS, df, heading)
				# end = time.time()
				# print('viewable', end - start)        
		
				image_filename = os.path.basename(img_path)
		
				# start = time.time()
				img_gps_buffer.append(img_object(frame_copy, long = CUR_GPS.longitude, lat = CUR_GPS.latitude, altitude = CUR_GPS.altitude, heading = heading, filename = image_filename, det_flag = det_flag, viewable_gt_flag = viewable_gt_flag, cw_assign = cw_assignment))
				cw_assign_np_li = np.append(cw_assign_np_li, cw_assignment)
				# end = time.time()
				# print('append', end - start)        
				
				# start = time.time()
				send_flag, batch, img_gps_buffer, cw_assign, change, cw_assign_np_li = check_buffer(img_gps_buffer, df = df, cw_assign_np_li = cw_assign_np_li, dont_detect = dont_detect)
				# end = time.time()
				# print('check buffer', end - start)        
					
				
				# start = time.time()
				# if send_flag:
				#     stats_dic = record_frame(image_filename, det_flag, viewable_gt_flag, cw_assignment, CUR_GPS.latitude, CUR_GPS.longitude, heading, send_flag)
				#     stats_dic = record_batch_info(stat_df, batch, stats_dic, cw_assign, change)
				#     stat_df = stat_df.append(pd.DataFrame([stats_dic]))  
				#     stat_df.reset_index(drop=True, inplace=True)            
				#     stat_df.to_csv(datetime.now(tz_NY).strftime("%Y_%m_%d") + '_stats.csv', index=False)
				# elif record_all_frames:
				#     stats_dic = record_frame(image_filename, det_flag, viewable_gt_flag, cw_assignment, CUR_GPS.latitude, CUR_GPS.longitude, heading, send_flag)
				#     stat_df = stat_df.append(pd.DataFrame([stats_dic]))  
				#     stat_df.reset_index(drop=True, inplace=True)            
					
				#     stat_df.to_csv(datetime.now(tz_NY).strftime("%Y_%m_%d") + '_stats.csv', index=False)
				# end = time.time()
				# print('record frame', end - start)        
							
				if send_flag == True:
					# if not on_bus:
					#     extract_batch_metadata(batch, send_flag, stat_df, cw_assign, change)
					input_frame = gabriel_pb2.InputFrame()
					engine_fields = busedge_pb2.EngineFields()
		
					input_frame, engine_fields = encode_batch(batch, input_frame, engine_fields, cw_assign)
		
					if send_to_server == True:
		
			
						input_frame.extras.Pack(engine_fields)
						serialized_message = input_frame.SerializeToString()
				
						rospy.loginfo(
							"Sent image msg with size {:.2f} KB".format(len(serialized_message) / 1024)
						)
				
						pub_data = UInt8MultiArray()
						pub_data.data = serialized_message
						pub.publish(pub_data)
				
						time.sleep(0.1)
					rospy.loginfo("Sent image msg")
				else:
					pass     
		else:

			print(len(img_gps_buffer))
			time.sleep(20)
			rospy.loginfo("no image to read")

			image_queue = sorted(os.listdir(src))
			#if after a long time, and nothing's changed, then release the images in buffer
			if len(image_queue) <= 5 and img_gps_buffer:
				send_flag, batch, img_gps_buffer, cw_assign, change, cw_assign_np_li = check_buffer(img_gps_buffer, df = df, cw_assign_np_li = cw_assign_np_li, dont_detect = dont_detect, release = True)

				if send_flag == True:
					# if not on_bus:
					#     extract_batch_metadata(batch, send_flag, stat_df, cw_assign, change)
					input_frame = gabriel_pb2.InputFrame()
					engine_fields = busedge_pb2.EngineFields()
		
					input_frame, engine_fields = encode_batch(batch, input_frame, engine_fields, cw_assign)
		
					if send_to_server == True:
		
			
						input_frame.extras.Pack(engine_fields)
						serialized_message = input_frame.SerializeToString()
				
						rospy.loginfo(
							"Sent image msg with size {:.2f} KB".format(len(serialized_message) / 1024)
						)
				
						pub_data = UInt8MultiArray()
						pub_data.data = serialized_message
						pub.publish(pub_data)
				
						time.sleep(0.1)
					rospy.loginfo("Sent image msg")





def stop_handler(sig, frame):
	global stop
	stop = True

if __name__ == "__main__":
	# run_node('camera3')

	parser = argparse.ArgumentParser()
	parser.add_argument(
		"-c",
		"--cam-id",
		nargs="+",
		type=int,
		default=[3],
		help="List, camera IDs to run",
	)
	# parser.add_argument("--undistort", default = True,  help="undistort bag images")

	args = parser.parse_args()


	signal.signal(signal.SIGINT, stop_handler)
	# signal.pause()

	if args.cam_id[0] != 0:
		# proc_poll = multiprocessing.Pool(5)
		for cam_id in args.cam_id:
			camera_name = "camera" + str(cam_id)
			

			p1 = multiprocessing.Process(target=run_node, args=(camera_name,))
			p1.start()
			p2 = multiprocessing.Process(target=publish_to_gabriel, args=(camera_name,))
			p2.start()
			
			# spin() simply keeps python from exiting until this node is stopped
