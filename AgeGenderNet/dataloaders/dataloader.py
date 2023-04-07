import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class custom_dataset(Dataset):
	def __init__(self, root_dir, split, transform=None, index=None):
		self.root_dir = root_dir
		self.transform = transform

		self.img_dir = os.path.join(root_dir, f'{split}_set/')
		self.images = os.listdir(self.img_dir) 

		#. filter out ages below 5
		self.images = [img for img in self.images if int(img.split('_')[0]) >= 5]  

		if index is not None:
			self.images = self.images[index[0]:index[1]]

	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		img_name = self.images[idx]

		#load image
		image = cv2.imread(self.img_dir + img_name)
		if self.transform:
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			image = self.transform(image=image)['image']

		gender = int(img_name.split('_')[1])
		age = int(img_name.split('_')[0]) / 100 

		gender = torch.tensor(gender, dtype=torch.float32).view(1)
		age = torch.tensor(age, dtype=torch.float32).view(1)
		return img_name, image, gender, age
	
def get_transforms():
	image_transforms = { 
		'train': A.Compose([
			A.Resize(224, 224),
			A.GaussNoise(p=0.5),
			A.HorizontalFlip(p=0.5),
			A.ShiftScaleRotate(p=0.5),
			A.HueSaturationValue(p=0.5),
			A.Blur(p=0.5),
			A.MotionBlur(p=0.5),
			A.RandomBrightnessContrast(p=0.5),
			A.RandomFog(p=0.5),
			A.RandomShadow(p=0.5),
			A.RandomSunFlare(p=0.5, src_radius=150),
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
	#.Load the Data
	image_transforms = get_transforms()

	#.Select how many images to train with. To overfit on 1 image, set num_images = 1
	if num_images is not None: 
		index = [0, int(num_images)]
	else:
		index = None

	dataset = custom_dataset(root_dir=data_dir,split=split, transform=image_transforms[split], index=index)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
	return dataloader, dataset
