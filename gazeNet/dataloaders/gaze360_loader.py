import cv2
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2


class custom_dataset(Dataset):

	def __init__(self, label_file, root, split, transform=None, index=None):
		self.transform = transform
		self.root = root
		self.orig_list_len = 0
		self.angle = 70
		self.binwidth = 4
		self.lines = []
	
		with open(label_file) as f:
			lines = f.readlines()
			lines.pop(0)
			self.orig_list_len = len(lines)
			for line in lines:
				gaze2d = line.strip().split(" ")[1:]
				label = np.array(gaze2d).astype("float")
				if abs((label[0]*180/np.pi)) <= self.angle and abs((label[1]*180/np.pi)) <= self.angle:
					self.lines.append(line)
						
		print("{} items removed from dataset that have an angle > {}".format(self.orig_list_len-len(self.lines), self.angle))

		if index is not None:
			self.lines = self.lines[index[0]:index[1]]
			
	def __len__(self):
		return len(self.lines)

	def __getitem__(self, idx):
		line = self.lines[idx]
		line = line.strip().split(" ")

		face = line[0]
		name = face.split("/")[0]
		gaze2d = line[1:]
		label = np.array(gaze2d).astype("float")
		label = torch.from_numpy(label).type(torch.FloatTensor)

		pitch = label[0]* 180 / np.pi
		yaw = label[1]* 180 / np.pi

		img = cv2.imread(os.path.join(self.root, face))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		if self.transform:
			img = self.transform(image=img)['image']

		# Bin values
		bins = np.array(range(-1*self.angle, self.angle, self.binwidth))
		binned_pose = np.digitize([pitch, yaw], bins) - 1

		labels = binned_pose
		cont_labels = torch.FloatTensor([pitch, yaw])
		return img, labels, cont_labels, name
	
def get_transforms():
	image_transforms = { 
		'train': A.Compose([
			A.Resize(64, 64),
			A.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],),
			ToTensorV2()
		]),

		'valid': A.Compose([
			A.Resize(64, 64),
			A.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],),
			ToTensorV2()
			]),


		'test': A.Compose([
			A.Resize(64, 64),
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

	label_file = os.path.join(data_dir, f'../{split}.txt')
	dataset = custom_dataset(label_file, data_dir, split, transform=image_transforms[split], index=index)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

	return dataloader, dataset
