import json
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2



class custom_dataset(Dataset):

	def __init__(self, root_dir, split, transform=None, index=None):
		self.root_dir = root_dir
		self.transform = transform
		
		#load train list 
		with open(f'{root_dir}/lmk_{split}.json') as f:
			self.data = json.load(f)

		self.images = []
		for key, value in self.data.items():
			image_path = key
			self.images.append(image_path)

		if index is not None:
			self.images = self.images[index[0]:index[1]]


	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		#load data
		if torch.is_tensor(idx):
			idx = idx.tolist()
		img_name = self.images[idx]
		values = self.data[img_name]
		bbox = values['bbox'] #[x1, y1, x2, y2]
		landmarks = values['landmarks']
		gaze = values['gaze']

		#load image
		image = cv2.imread(self.root_dir + img_name)
		img_h, img_w, _ = image.shape
		# transform image
		# image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
		if self.transform:
			image = self.transform(image=image)['image']

		#normalize bbox
		bbox_h = bbox[3] - bbox[1]
		bbox_w = bbox[2] - bbox[0]
		bbox = torch.tensor([bbox[0], bbox[1], bbox_w, bbox_h]) / torch.tensor([img_w, img_h, img_w, img_h])

		#normalize landmarks
		landmarks = torch.tensor(landmarks) / torch.tensor([img_w, img_h])
		landmarks = landmarks.view(-1)

		#gaze
		gaze = torch.tensor([float(gaze[0]), float(gaze[1])])
		
		#prepare input by combining bbox and landmarks
		bb_lmk = torch.cat((bbox, landmarks), dim=0)

		

		return image, bb_lmk, gaze
	
def get_transforms():
	image_transforms = { 
		'train': A.Compose([
			A.Resize(224, 224),
			A.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],),
			ToTensorV2()
		]),

		'valid': A.Compose([
			A.Resize(224, 224),
			A.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],),
			ToTensorV2()
			]),


		'test': A.Compose([
			A.Resize(224, 224),
			A.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],),
			ToTensorV2()
			])
	}

	return image_transforms

def get_dataloader(data_dir, batch_size, split, num_images=None, num_workers=1):
	# Load the Data
	image_transforms = get_transforms()

	#select how many images to train with. To overfit on 1 image, set num_images = 1
	if num_images is not None: 
		index = [0, int(num_images)]
	else:
		index = None

	dataset = custom_dataset(root_dir=data_dir,split=split, transform=image_transforms[split], index=index)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

	return dataloader, dataset
