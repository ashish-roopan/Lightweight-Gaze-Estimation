import cv2
import numpy as np
import time
import argparse
import torch
import mediapipe as mp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from models.GenderNet import GenderNet


def detect_face(image):
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

# def enlarge_bbox(bbox):
# 	#.Enlarge bounding box by 10% on each side
# 	xmin, ymin, xmax, ymax = bbox
# 	width = xmax - xmin
# 	height = ymax - ymin
# 	new_xmin = max(0, xmin - int(width * 0.2))
# 	new_ymin = max(0, ymin - int(height * 0.2))
# 	new_xmax = min(640, xmax + int(width * 0.2))
# 	new_ymax = min(480, ymax + int(height * 0.2))
	
def enlarge_bbox(bbox, p=0.3):
	#.Enlarge bounding box by p% on each side
	xmin, ymin, xmax, ymax = bbox
	width = xmax - xmin
	height = ymax - ymin
	new_xmin = max(0, xmin - int(width * p))
	new_ymin = max(0, ymin - int(height * p))
	new_xmax = min(W, xmax + int(width * p))
	new_ymax = min(H, ymax + int(height * p))
	new_width = new_xmax - new_xmin
	new_height = new_ymax - new_ymin
	return np.array([new_xmin, new_ymin, new_width, new_height])


def preprocess_input(frame, bbox, landmarks):
	#.Enlarge bounding box
	enlarged_bbox = enlarge_bbox(bbox, p=.5)

	#.Crop face
	face_img = frame[enlarged_bbox[1]:enlarged_bbox[1]+enlarged_bbox[3], enlarged_bbox[0]:enlarged_bbox[0]+enlarged_bbox[2]]
	
	face_img = transform(image=face_img)['image']
	return face_img, enlarged_bbox

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input', default=0, help='Path to a video or image file. Skip to capture frames from a camera.')
	parser.add_argument('--model_path', type=str, default='checkpoints/model_9_0.18582_0.15715_18965.pt', help='model path')
	parser.add_argument('--device', type=str, default='cpu', help='device')
	parser.add_argument('--title', type=str, default='out', help='title')
	return parser.parse_args()


#.Initial setup									
args = parse_args()
device = torch.device(args.device)
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) 
cap = cv2.VideoCapture(args.input)
transform =  A.Compose([
			A.Resize(224, 224),
			A.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],),
			ToTensorV2()])

mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device)
std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device)


#.​‌‍‌𝗣𝗿𝗲𝗽𝗮𝗿𝗲 𝗠𝗼𝗱𝗲𝗹    ​                					
model = GenderNet()
model = model.to(args.device)

# import torch._dynamo as dynamo
# torch._dynamo.config.verbose = True
# torch.backends.cudnn.benchmark = True
# model = torch.compile(model, mode="max-autotune", fullgraph=False)
# print("Model compiled set")

model = torch.compile(model)
checkpoint = torch.load(args.model_path)
model.load_state_dict(checkpoint['model'])
model.eval()


#.main loop
while True:
	ret, frame = cap.read()
	start0 = time.time()
	if not ret : 
		break
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	H, W, _ = frame.shape

	#.Detect face and landmarks
	bboxes, landmarks = detect_face(frame)
	if (bboxes.size and landmarks.size):	
		for bbox, landmark in zip(bboxes, landmarks):
			#.Detect gender
			face_img, enlarged_bbox = preprocess_input(frame, bbox, landmark)
			face_img = face_img.unsqueeze(0).to(device)
			pred_gender = model(face_img)[0]
			print('pred_gender', pred_gender)
			
			#.get image
			image = face_img[0]
			image = image.permute(1, 2, 0)
			image = image * std + mean
			image = image.cpu().numpy()
			image = np.ascontiguousarray(image * 255, dtype=np.uint8)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			h, w = image.shape[:2]
			
			cv2.rectangle(frame, (enlarged_bbox[0], enlarged_bbox[1]), (enlarged_bbox[0]+enlarged_bbox[2], enlarged_bbox[1]+enlarged_bbox[3]), (0, 0, 255), 2)
			if pred_gender[0] > 0.5:
				pred_gender = 'Female'
			else:
				pred_gender = "Male"
			text = f'{str(pred_gender)}'
			cv2.rectangle(frame, (enlarged_bbox[0], enlarged_bbox[1]), (enlarged_bbox[0] + 50, enlarged_bbox[1]+25), (255, 255, 255), -1)
			cv2.putText(frame, text, (enlarged_bbox[0], enlarged_bbox[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0,0), 1)

	#.Show frame
	frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
	cv2.imshow(args.title, frame)
	if cv2.waitKey(30) == ord('q'):
		break







