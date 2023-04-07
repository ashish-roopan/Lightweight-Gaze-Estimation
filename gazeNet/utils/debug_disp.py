import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


def draw_gaze(a,b,c,d,image, pitchyaw, thickness=2, color=(255, 255, 0),sclae=2.0):
	"""Draw gaze angle on given image with a given eye positions."""
	(h, w) = image.shape[:2]
	length = w/2
	pos = (int(a+c / 2.0), int(b+d / 2.0))
	dx = -length * np.sin(pitchyaw[0]) * np.cos(pitchyaw[1])
	dy = -length * np.sin(pitchyaw[1])
	cv2.arrowedLine(image, tuple(np.round(pos).astype(np.int32)),
				   tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
				   thickness, cv2.LINE_AA, tipLength=0.18)
	return image    

mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device='cuda')
std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device='cuda')
softmax = nn.Softmax(dim=1).cuda()
idx_tensor = [idx for idx in range(90)]
idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda()

def debug_disp(model, dataloader, device):
	images, bin_gaze, gt_gaze,name = next(iter(dataloader))
	images = images.to(device)
	model.eval()
	with torch.no_grad():
		yaw, pitch = model(images)

	pitch_predicted = softmax(pitch)
	yaw_predicted = softmax(yaw)

	pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1) * 4 - 180
	yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1) * 4 - 180

	pitch = pitch_predicted[0].cpu().numpy() * np.pi / 180
	yaw = yaw_predicted[0].cpu().numpy() * np.pi / 180


	
	#get image
	image = images[0]
	image = image.permute(1, 2, 0)
	image = image * std + mean
	image = image.cpu().numpy()
	image = np.ascontiguousarray(image * 255, dtype=np.uint8)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	h, w = image.shape[:2]

	
	gt_pitch = float(gt_gaze[0][0]) * np.pi / 180
	gt_yaw = float(gt_gaze[0][1]) * np.pi / 180

	gt_gaze = [gt_pitch, gt_yaw]
	# pred_gaze = [float(pitch), float(yaw)]
	pred_gaze = [float(yaw), float(pitch)]

	#draw gaze
	image = draw_gaze(0,0,w,h,image, gt_gaze, color=(0, 255, 0))
	image = draw_gaze(0,0,w,h,image, pred_gaze, color=(0, 0, 255))

	image = cv2.resize(image, (512, 512))
	return image

