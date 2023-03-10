from centroid_tracker import CentroidTracker
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import mediapipe as mp
from MP_Gaze_estimation import gaze as Gaze

def predict_gaze(frame, bbox):
	face_img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
	print('face_img.shape', face_img.shape)
	results = face_mesh.process(face_img)
	if results.multi_face_landmarks:
		gaze = Gaze.gaze(face_img, results.multi_face_landmarks[0])
	else:
		gaze = None
	return gaze

def parse_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
	ap.add_argument('-i', '--input', default=0, help='Path to a video or image file. Skip to capture frames from a camera.')
	args = vars(ap.parse_args())
	return args

#. Initial setup					
args = parse_args()
ct = CentroidTracker(maxDisappeared=100, maxDistance=50)
(H, W) = (None, None)
#. Face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=args['confidence'])  
#. Face mesh
mp_face_mesh = mp.solutions.face_mesh  
face_mesh = mp_face_mesh.FaceMesh(
		max_num_faces=1,  # number of faces to track in each frame
		refine_landmarks=True,  # includes iris landmarks in the face mesh model
		min_detection_confidence=0.5,
		min_tracking_confidence=0.5) 

if args['input'] == 0:
	vs = VideoStream(src=0).start()
	time.sleep(2.0)
else:
	vs = cv2.VideoCapture(args['input'])

#. Main loop						
while True:
	ret, frame = vs.read()
	if not ret: break
	start = time.time()
	frame = imutils.resize(frame, width=400)
	frame.flags.writeable = False
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	#. Detect faces					
	results = face_detection.process(frame)
	rects = []
	if results.detections:
		for detection in results.detections:
			box = detection.location_data.relative_bounding_box
			xmin = int(box.xmin * W)
			ymin = int(box.ymin * H)
			xmax = xmin + int(box.width * W)
			ymax = ymin + int(box.height * H)
			rects.append((xmin, ymin, xmax, ymax))
	
	#. update our centroid tracker 	
	objects = ct.update(rects)
	
	#. loop over the tracked objects
	for (objectID, object_data) in objects.items():
		centroid = object_data[0]
		bbox = object_data[1]

		#. Predict gaze
		gaze = predict_gaze(frame, bbox)
		
		#. Draw the bounding box 
		frame.flags.writeable = True
		text = "ID:{}".format(objectID)
		cv2.circle(frame, (centroid[0], centroid[1]), 2, (0, 255, 0), -1)
		cv2.rectangle(frame,(bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
		cv2.rectangle(frame, (bbox[0], bbox[1] - 20), (bbox[2], bbox[1]), (255, 255, 255), -1)
		cv2.putText(frame, text, (bbox[0], bbox[1] ), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

		#. Draw gaze
		if gaze is not None:
			cv2.arrowedLine(frame, (centroid[0], centroid[1]), (centroid[0] + int(gaze[0]), centroid[1] + int(gaze[1])), (0, 0, 255), 2)

	#. show the output frame		
	fps = 1.0 / (time.time() - start)
	cv2.rectangle(frame, (0, 0), (100, 40), (255, 255, 255), -1)
	cv2.putText(frame, "FPS: {:.0f}".format(fps), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(30) & 0xFF
	if key == ord("q"):
		break
	
cv2.destroyAllWindows()
vs.stop()
