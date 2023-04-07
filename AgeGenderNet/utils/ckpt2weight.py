import os
import torch
import argparse
import sys
sys.path.insert(0, '.')
from models.AgeGenderNet import GenderNet

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='gaze_estimation/checkpoints/model_47_4.37111_212.16002_126927.pth', help='model path')
    parser.add_argument('--device', type=str, default='cpu', help='device')
    args = parser.parse_args()
    return args

args = parse_args()
device = args.device
model_path = args.model_path
save_path = model_path.replace('.pth', '.pt')
model = GenderNet()
model = torch.compile(model)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

print('model.device: ', next(model.parameters()).device)

torch.save({
    'model': model.state_dict(),
}, save_path)

print(f'Model saved to {save_path}')