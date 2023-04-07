import numpy as np
import argparse
import torch
import time
import cv2
import mediapipe as mp
import imutils
import onnxruntime as ort
import onnx

def enlarge_bbox(bbox, p=0.3):
	#.Enlarge bounding box by p% on each side
	xmin, ymin, xmax, ymax = bbox
	width = xmax - xmin
	height = ymax - ymin
	new_xmin = max(0, xmin - int(width * p))
	new_ymin = max(0, ymin - int(height * p))
	new_xmax = min(W, xmax + int(width * p))
	new_ymax = min(H, ymax + int(height * p))
	return np.array([new_xmin, new_ymin, new_xmax, new_ymax])

def preprocess_input(frame, bbox):
	face_img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
	face_img = cv2.resize(face_img, (args.img_size, args.img_size))
	face_img = (face_img / 255.0 - mean) / std
	face_img = face_img.transpose(2, 0, 1)
	face_img = torch.from_numpy(face_img).float()
	return face_img

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
	parser.add_argument('-i', '--input', default=0, help='Path to a video or image file. Skip to capture frames from a camera.')
	parser.add_argument('--model_path', type=str, default='gaze_estimation/checkpoints/final.onnx', help='model path')
	parser.add_argument('--img_size', type=int, default=64, help='image size')
	args = parser.parse_args()
	return args

#. Initial setup
args = parse_args()
(H, W) = (None, None)
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

#. Face detector
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=args.confidence)  

#. AgeGenderNet model
model_path = args.model_path
onnx_model = onnx.load(model_path)
onnx.checker.check_model(onnx_model)
ort_sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
print(f'Model loaded from {model_path}')

#. Video stream
vs = cv2.VideoCapture(args.input)

#. Main loop						
while True:
	ret, frame = vs.read()
	frame = imutils.resize(frame, width=800)
	if not ret: break
	start = time.time()
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
		bbox = enlarge_bbox(bbox, p=0.3)

		#. Predict gaze
		gaze = predict_gaze(frame, bbox)
		
		#. Draw the bounding box and gaze
		frame.flags.writeable = True
		text = "ID:{}".format(objectID)
		cv2.circle(frame, (centroid[0], centroid[1]), 2, (0, 255, 0), -1)							#centroid
		cv2.rectangle(frame,(bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), 1)			 	#bbox
		cv2.rectangle(frame, (bbox[0], bbox[1] - 20), (bbox[2], bbox[1]), (255, 255, 255), -1)	 	#text background for ID 
		cv2.putText(frame, text, (bbox[0], bbox[1] ), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)	#ID
		if gaze is not None:
			frame = draw_gaze(centroid, frame, gaze, color=(255, 255, 0), thickness=3) 				#gaze

	#. show the output frame		
	fps = 1.0 / (time.time() - start)
	cv2.rectangle(frame, (0, 0), (100, 40), (255, 255, 255), -1)
	cv2.putText(frame, "FPS: {:.0f}".format(fps), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
	frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)	
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(30) & 0xFF
	if key == ord("q"):
		break
	
cv2.destroyAllWindows()
