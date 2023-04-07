import cv2
import numpy as np
import argparse
import torch
import mediapipe as mp

from gaze_estimation.models.fc import FC



def detect_face(image):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	height, width, _ = image.shape
	image.flags.writeable = False
	results = face_detection.process(image)
	image.flags.writeable = True
	if not results.detections:
		return np.array([]), np.array([])
	bboxes, landmarks = [], []
	for detection in results.detections:
		#.get bounding box
		bbox = detection.location_data.relative_bounding_box
		bbox = [bbox.xmin, bbox.ymin, bbox.width, bbox.height]
		xmin = int(bbox[0] * width)
		ymin = int(bbox[1] * height)
		xmax = int(xmin + bbox[2] * width)
		ymax = int(ymin + bbox[3] * height)
		bbox = [xmin, ymin, xmax, ymax]

		#.get landmarks
		landmark = detection.location_data.relative_keypoints
		landmark = [[int(l.x * width) , int(l.y*height)] for l in landmark]

		bboxes.append(bbox)
		landmarks.append(landmark)
	return np.array(bboxes), np.array(landmarks)

def draw_landmarks_and_bbox(frame, bbox, landmarks):
	#.Draw bounding box
	cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
	#.Draw landmarks
	for landmark in landmarks:
		cv2.circle(frame, (landmark[0], landmark[1]), 2, (0, 0, 255), -1)

	return frame

def draw_gaze(xmin, ymin, w, h,image_in, pitchyaw, thickness=2, color=(255, 255, 0),scale=2.0):
	image_out = image_in
	length = w/2
	pos = (int(xmin+w / 2.0), int(ymin+ h / 2.0))
	dx = -length * np.sin(pitchyaw[0]) * np.cos(pitchyaw[1])
	dy = -length * np.sin(pitchyaw[1])
	cv2.arrowedLine(image_out, tuple(np.round(pos).astype(np.int32)),
				   tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
				   thickness, cv2.LINE_AA, tipLength=0.18)
	return image_out    

def enlarge_bbox(bbox):
	#.Enlarge bounding box by 10% on each side
	xmin, ymin, xmax, ymax = bbox
	width = xmax - xmin
	height = ymax - ymin
	new_xmin = max(0, xmin - int(width * 0.2))
	new_ymin = max(0, ymin - int(height * 0.2))
	new_xmax = min(640, xmax + int(width * 0.2))
	new_ymax = min(480, ymax + int(height * 0.2))
	new_width = new_xmax - new_xmin
	new_height = new_ymax - new_ymin
	return np.array([new_xmin, new_ymin, new_width, new_height])

def preprocess_input(frame, bbox, landmarks):
	#.Enlarge bounding box
	enlarged_bbox = enlarge_bbox(bbox)

	#.Crop face
	face = frame[enlarged_bbox[1]:enlarged_bbox[1]+enlarged_bbox[3], enlarged_bbox[0]:enlarged_bbox[0]+enlarged_bbox[2]]
	face_img_h, face_img_w, _ = face.shape
	
	#.Normalize bbox
	bbox = torch.tensor([bbox[0], bbox[1], bbox[2], bbox[3]]) / torch.tensor([face_img_w, face_img_h, face_img_w, face_img_h])

	#.Normalize landmarks
	landmarks = torch.tensor(landmarks) / torch.tensor([face_img_w, face_img_h])
	landmarks = landmarks.view(-1)

	#.Prepare input by combining bbox and landmarks
	bb_lmk = torch.cat((bbox, landmarks), dim=0)
	return bb_lmk, enlarged_bbox

def detect_gaze(lmk):
	inputs = lmk.view(1, -1).to(args.device)
	#.run model    
	model.eval()
	with torch.no_grad():
		pred_gaze = model(inputs)
	
	return np.array(pred_gaze[0].cpu())
	
def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_path', type=str, default='gaze_estimation/checkpoints/model_64_0.05476_0.07085_106680.pth', help='model path')
	parser.add_argument('--device', type=str, default='cuda:0', help='device')
	parser.add_argument('--title', type=str, default='debug', help='title')
	return parser.parse_args()


#.Initial setup									
args = parse_args()
device = torch.device(args.device)
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) 
cap = cv2.VideoCapture('/home/ashish/Videos/sp1.mp4')

#.Prepare model    ​                                	 
model = FC(input_size=16, output_size=2)
model = model.to(args.device)
checkpoint = torch.load(args.model_path)
model.load_state_dict(checkpoint['model'])

#.main loop
while True:
	ret, frame = cap.read()
	if not ret : 
		break
	img_h, img_w, _ = frame.shape
	frame = cv2.resize(frame, (img_w, img_h))

	#.Detect face and landmarks
	bboxes, landmarks = detect_face(frame)
	if (bboxes.size and landmarks.size):	
		for bbox, landmark in zip(bboxes, landmarks):
			#.Draw bounding box and landmarks
			frame = draw_landmarks_and_bbox(frame, bbox, landmark)

			#.Detect gaze
			lmk, enlarged_bbox = preprocess_input(frame, bbox, landmark)
			gaze = detect_gaze(lmk)

			#.Draw gaze
			frame = draw_gaze(enlarged_bbox[0], enlarged_bbox[1], enlarged_bbox[2], enlarged_bbox[3], frame, gaze)

	#.Show frame
	cv2.imshow(args.title, frame)
	if cv2.waitKey(30) == ord('q'):
		break







