import os
import cv2
import random
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models.AgeGenderNet import AgeGenderNet
from dataloaders.dataloader import get_dataloader
from scripts.train import train_epoch
from scripts.validate import valid_epoch
from utils.debug_disp import debug_disp

import wandb
wandb.init(project="Gender classification")

def log_results():
	#. Log  			
	avg_train_loss = sum(train_losses) / len(train_losses)
	avg_valid_loss = sum(valid_losses) / len(valid_losses)
	avg_test_loss =  sum(test_losses) / len(test_losses)

	print(f'Epoch:{start_epoch + epoch :03d} |\
	Training Loss:{avg_train_loss:.6f} |\
	valid Loss:{avg_valid_loss:.6f} |\
	test Loss:{avg_test_loss:.6f} |\
	LR : {scheduler.get_last_lr()[0]:.6f}')

	wandb.log({'train_loss': avg_train_loss,\
			'valid_loss': avg_valid_loss,\
			'test_loss': avg_test_loss})

	#. debug display 	
	if args.debug and epoch % 1 == 0:
		train_debug_disp = debug_disp(model, train_dataloader, device)
		valid_debug_disp = debug_disp(model, valid_dataloader, device)
		test_debug_disp = debug_disp(model, test_dataloader, device)

		out_img = np.concatenate((train_debug_disp, valid_debug_disp, test_debug_disp), axis=0)
		out_img = cv2.resize(out_img, (0, 0), fx=0.7, fy=0.7)

		cv2.imshow(args.title, out_img)
		cv2.waitKey(1)

	#. save model
	if args.save_model:
		save_model(avg_train_loss, avg_valid_loss)
def save_model(avg_train_loss, avg_valid_loss):
	global best_valid_loss
	global ckpt_flag
	if avg_valid_loss < best_valid_loss:
		best_valid_loss = avg_valid_loss
		save_path = os.path.join('./checkpoints/', f'model_{epoch}_{avg_train_loss:.5f}_{avg_valid_loss:.5f}_{len(train_dataset)}.pth')
		torch.save({
			'model': model.state_dict(),
			'optimizer': optimizer.state_dict(),
			# 'lr_sched': scheduler,
			'epoch': start_epoch + epoch,
			'best_valid_loss': best_valid_loss,
			'args': args
		}, save_path)
		print(f'Best model saved to {save_path}')

	if epoch == epochs - 1:
		save_path = os.path.join('./checkpoints/', f'model_{epoch}_{avg_train_loss:.5f}_{avg_valid_loss:.5f}_{len(train_dataset)}.pth')
		torch.save({
			'model': model.state_dict(),
			'optimizer': optimizer.state_dict(),
			'lr_sched': scheduler,
			'epoch': start_epoch + epoch,
			'best_valid_loss': best_valid_loss,
			'args': args
		}, save_path)
		print(f'Last model saved to {save_path}')
def deterministic(rank):
	torch.manual_seed(rank)
	torch.cuda.manual_seed(rank)
	np.random.seed(rank)
	random.seed(rank)
	cudnn.deterministic = True
	cudnn.benchmark = False
def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='datasets/UTKFace/', help='data directory')
	parser.add_argument('--batch_size', type=int, default=1, help='batch size')
	parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs')
	parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
	parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
	parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
	parser.add_argument('--model_path', type=str, help='model path')
	parser.add_argument('--seed', type=int, default=3407, help='random seed')
	parser.add_argument('--device', type=str, default='cuda:0', help='device')
	parser.add_argument('--save_model', type=bool, default=True, help='save model')
	parser.add_argument('--debug', type=bool, default=True, help='debug')
	parser.add_argument('--num_images', type=int, default=320000, help='number of images to train')
	parser.add_argument('--title', type=str, default='out', help='title of the project')
	return parser.parse_args()


#> 				Initial setup									
args = parse_args()
deterministic(args.seed)
device = torch.device(args.device)

#>                ​‌‍‌‍𝗣𝗿𝗲𝗽𝗮𝗿𝗲 𝗗𝗮𝘁𝗮​                		  			
data_dir = args.data_dir
train_dataloader, train_dataset = get_dataloader(data_dir  , batch_size=args.batch_size, num_images=args.num_images, split='train', num_workers=10)
valid_dataloader, valid_dataset = get_dataloader(data_dir , batch_size=args.batch_size, num_images=args.num_images, split='valid', num_workers=10)
test_dataloader, test_dataset = get_dataloader(data_dir , batch_size=args.batch_size,num_images=args.num_images, split='test', num_workers=10)
print('len(train_dataset): ', len(train_dataset))
print('len(valid_dataset): ', len(valid_dataset))
print('len(test_dataset): ', len(test_dataset))

#>           ​‌‍‌𝗣𝗿𝗲𝗽𝗮𝗿𝗲 𝗠𝗼𝗱𝗲𝗹    ​                					
model = AgeGenderNet()
model = model.to(args.device)
# model = torch.compile(model)

#>         ​‌‍‌𝗛𝘆𝗽𝗲𝗿𝗽𝗮𝗿𝗮𝗺𝗲𝘁𝗲𝗿𝘀​​                   					
start_epoch = 0
best_valid_loss = 2550
lr = args.lr
momentum = 0.9
weight_decay = args.weight_decay
epochs = args.num_epochs
criterion = nn.BCELoss()
# reg_criterion = nn.MSELoss()
reg_criterion = nn.L1Loss()

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500], gamma=0.1)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_dataloader), epochs=epochs)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3,
					       step_size_up=32000, step_size_down=32000, mode='triangular',
					       cycle_momentum=False, base_momentum=0.85, max_momentum=0.95, last_epoch=-1, verbose=False)

#>         𝗟𝗼𝗮𝗱 𝗠𝗼𝗱𝗲⁡𝗹	​											
if args.model_path:
	checkpoint = torch.load(args.model_path)
	model.load_state_dict(checkpoint['model'])
	# optimizer.load_state_dict(checkpoint['optimizer'])
	# scheduler = checkpoint['lr_sched']
	# start_epoch = checkpoint['epoch']
	# best_valid_loss = checkpoint['best_valid_loss']
	# lr = scheduler.get_last_lr()
	print(f'Model loaded from {args.model_path} at epoch {start_epoch} with best valid loss {best_valid_loss}')

wandb.config = {
	"learning_rate": lr,
	"epochs": epochs,
	"batch_size": args.batch_size,
	"weight_decay" : 0.01,
	"momentum" : 0.9,
	"optimizer" : optimizer,
	"scheduler" : scheduler
}

#>         Train Model										
for epoch in range(epochs):
	train_losses = train_epoch(model, train_dataloader, criterion, reg_criterion, optimizer, scheduler, args.device, wandb)
	valid_losses = valid_epoch(model, valid_dataloader, criterion, reg_criterion, args.device, wandb)
	test_losses =  valid_epoch(model, test_dataloader, criterion, reg_criterion, args.device, wandb)

	#.log and save model
	log_results()