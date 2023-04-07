import cv2
import numpy as np
import torch

def postprocess(images):
	#.unnormalize
	image = images[0].cpu()
	image = (image * std + mean) * 255
	image = torch.clamp(image, 0, 255)
	image = np.array(image.permute(1,2,0).numpy(), dtype=np.uint8)
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	return image

mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3,1,1)
std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3,1,1)

def debug_disp(model, dataloader, device):
	img_names, images, gt_genders, gt_ages = next(iter(dataloader))
	images = images.to(device) 
	model.eval()
	with torch.no_grad():
		pred_genders, pred_age = model(images)

	#.postprocess
	image = postprocess(images)
	if pred_genders[0] > 0.5:
		pred_gender = 1
	else:
		pred_gender = 0

	age_diff = int(abs(gt_ages[0] - pred_age[0].detach().cpu().numpy()) * 100)
	
	text = f'{str(int(gt_genders[0]))} : {str(pred_gender)} : {str(age_diff)}'
	cv2.rectangle(image, (0, 0), (224, 50), (255, 255, 255), -1)
	cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0,0), 2)
	return image


