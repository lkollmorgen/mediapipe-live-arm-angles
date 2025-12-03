## Mediapipe tools
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

## utils
from pathlib import Path
import cv2
import numpy as np
import math
import time

ROOT = Path(__file__).resolve().parent.parent
#model_path = ROOT / 'pose_landmarker_lite.task'
model_path = ROOT / 'util' / 'pose_landmarker_full.task'
time_log = ROOT / 'data' / 'time.log'
coords_file = ROOT / 'data' / 'coords.txt'

latest_result = None

#============================================
#Annotation functions
#============================================

def to_pixel_coords(landmark, width, height):
	return int(landmark[0] * width), int(landmark[1] * height)

def process_frame(rgb_image, detection_result, \
				  log_to_output=False, confidence_threshold = 0.0):
	pose_landmarks_list = detection_result.pose_landmarks
	annotated_image = np.copy(rgb_image)
	
	coords = {}
	
	if pose_landmarks_list is None:
		return annotated_image, None

	for pose_landmarks in pose_landmarks_list:
		proto = landmark_pb2.NormalizedLandmarkList()

		marks = pose_landmarks
		for lm in marks:
			proto.landmark.append(
				landmark_pb2.NormalizedLandmark(
					x=lm.x, y=lm.y, z=lm.z
				)
			)

		solutions.drawing_utils.draw_landmarks(
			annotated_image,proto,
			solutions.pose.POSE_CONNECTIONS,
			solutions.drawing_styles.get_default_pose_landmarks_style()
		)
			
		required = [11, 12, 15, 16, 23, 24]
		
		if any(pose_landmarks[idx].visibility < confidence_threshold \
			   for idx in required):
			return annotated_image, None
		
		coords = {
			"left shoulder" : [marks[11].x, marks[11].y, marks[11].z],
			"left wrist" : [marks[15].x, marks[15].y, marks[15].z],
			"left hip" : [marks[23].x, marks[23].y, marks[23].z],
			"right shoulder" : [marks[12].x, marks[12].y, marks[12].z],
			"right wrist" : [marks[16].x, marks[16].y, marks[16].z],
			"right hip" : [marks[24].x, marks[24].y, marks[24].z]
		}	
		
		if log_to_output:
			for k, v in coords.items():
				print(f"{k}: {v}", file=coords_file)
		
		return annotated_image, coords

	return annotated_image, None


start_time = time.perf_counter()

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

end_time = time.perf_counter()
with open(time_log, 'w') as f:
	print(f"Results setup: {end_time - start_time}",file=f)

#============================================
#Util functions
#============================================

def angle_at_shoulder_deg(shoulder, wrist, hip, use_chest=False):
	s = np.array(shoulder, dtype=float)
	w = np.array(wrist, dtype=float)
	h = np.array(hip, dtype=float)

	v_torso = h - s
	v_arm = w - s
	if use_chest:
		v_torso = 0.5 * v_torso
		
	na = np.linalg.norm(v_torso)
	nb = np.linalg.norm(v_arm)
	if na < 1e-6 or nb < 1e-6:
		return None
	
	dot = np.dot(v_torso, v_arm)
	cos_theta = dot / (na * nb)
	cos_theta = max(-1.0, min(1.0, cos_theta))
	theta = math.degrees(math.acos(cos_theta))
	return np.round(theta, decimals=0)

# Create a pose landmarker instance with the live stream mode:
def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
	#print('pose landmarker result: {}'.format(result))
	global latest_result
	latest_result=result

start_time = time.perf_counter()

#video = cv2.VideoCapture(0)
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
video.set(cv2.CAP_PROP_FPS, 30)

end_time = time.perf_counter()
with open(time_log, 'a') as f:
	print(f"Init video: {end_time - start_time}",file=f)

start_time = time.perf_counter()

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

end_time = time.perf_counter()
with open(time_log, 'a') as f:
	print(f"Set options: {end_time - start_time}",file=f)

timestamp_ms = 0
with PoseLandmarker.create_from_options(options) as landmarker:
	start_time = time.perf_counter()

	# Get the default frame width and height
	frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
	frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
	# Define the codec and create VideoWriter object
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	out = cv2.VideoWriter('../output.mp4', fourcc, 20.0, (frame_width, frame_height))
	
	end_time = time.perf_counter()
	with open(time_log, 'a') as f:
		print(f"Define frame size & create videowriter: {end_time - start_time}",file=f)

	try: 
		while video.isOpened():
			ret, frame = video.read()

			if not ret:
				print("Ignoring empty frame")
				break
			
			# Convert the frame received from OpenCV to a MediaPipe’s Image object.
			frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
			
			timestamp_ms = int(time.time() * 1000)
			landmarker.detect_async(mp_image, timestamp_ms)
					
			if latest_result is not None:
				annotated, latest_coords = process_frame(frame_rgb, latest_result)
				if latest_coords is not None:	
					## gather shoulder angles in pixel
					lw_x, lw_y = to_pixel_coords(latest_coords["left shoulder"], frame_width, frame_height)
					ls_x, ls_y = to_pixel_coords(latest_coords["left wrist"], frame_width, frame_height)
					lh_x, lh_y = to_pixel_coords(latest_coords["left hip"], frame_width, frame_height)

					rw_x, rw_y = to_pixel_coords(latest_coords["right shoulder"], frame_width, frame_height)
					rs_x, rs_y = to_pixel_coords(latest_coords["right wrist"], frame_width, frame_height)
					rh_x, rh_y = to_pixel_coords(latest_coords["right hip"], frame_width, frame_height)

					l_angle = angle_at_shoulder_deg((lw_x, lw_y),(ls_x, ls_y), (lh_x, lh_y))
					r_angle = angle_at_shoulder_deg((rw_x, rw_y),(rs_x, rs_y), (rh_x, rh_y))
					
					cv2.putText(annotated, f"L: {l_angle:.1f}",
								(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2,cv2.LINE_AA)
					cv2.putText(annotated, f"R: {r_angle:.1f}",
								(frame_width-140, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2,cv2.LINE_AA)

			else:
				annotated = frame_rgb
			annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)			

			# Display the captured frame
			cv2.imshow('Camera', annotated_bgr)
			#out.write(frame)
			out.write(annotated_bgr)

			# Press 'q' to exit the loop
			if cv2.waitKey(1) == ord('q'):
				break

	finally:
	
		# Release the capture and writer objects
		video.release()
		out.release()
		cv2.destroyAllWindows()
