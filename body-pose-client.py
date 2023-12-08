# from this video: 
# https://www.youtube.com/watch?v=-toNMaS4SeQ

import cv2

import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import numpy as np
import time
import socket
import sys

bSocket = True

if (len(sys.argv) > 1):
	print(sys.argv)
	if sys.argv[1] == "--no-socket":
		bSocket = False


if bSocket:
	# open socket to omniverse machine
	mysocket = socket.socket()
	mysocket.connect(('192.168.4.5',12346)) # easybake
	# mysocket.connect(('127.0.0.1',12346))


def close_socket(thissocket):
    try:
        thissocket.shutdown(socket.SHUT_RDWR)
        thissocket.close()
        thissocket = None
    except socket.error as e:
        pass
    print("socket is closed")


def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

# set up pose landmarker from livestream
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a pose landmarker instance with the live stream mode:
def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('pose landmarker result: {}'.format(result))

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

with PoseLandmarker.create_from_options(options) as landmarker:

	cap = cv2.VideoCapture(0)
	# cap = cv2.VideoCapture(1)

	count = 0
	start = time.time()
	try: 
		while cap.isOpened():
			
			success, image = cap.read()
			frame_timestamp_ms = time.time() - start
			if image is None:
				continue

			# image = cv2.cvtColor(cv2.flip(image, -1), cv2.COLOR_BGR2RGB)

			# razer kyo pro
			image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

			# improve performance
			image.flags.writeable = False

			# Detect pose landmarks from the input frame
			mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
			# detection_result = detector.detect(mp_image)
			landmarker.detect_async(mp_image, frame_timestamp_ms)
			# detection_result = detector.detect(image)

			# #Process the detection result. In this case, visualize it.
			# annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
			# cv2.imshow("Body Pose", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

			# # improve performance
			# image.flags.writeable = True

			# cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

			# if bSocket:
			# 	try:
			# 		count += 1
			# 		if count % 3 == 0:
			# 			print("sending:", [x_rot, y_rot, z_rot, xdiff, ydiff, zdiff])
			# 			count = 0
			# 		sendData = str([x_rot, y_rot, z_rot, xdiff, ydiff, zdiff])
			# 		mysocket.send(sendData.encode())
			# 	except:
			# 		pass

			# cv2.imshow('Head Pose Estimation', image)

			if cv2.waitKey(5) & 0xFF == 27:
				break

	except KeyboardInterrupt:
		print("quitting")

	if bSocket:
		close_socket(mysocket)

	cap.release()


