import torch
import argparse
import sys
import numpy as np

sys.path.append('.')
from models.AgeGenderNet import GenderNet

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='checkpoints/model_47.pth', help='model path')
    parser.add_argument('--device', type=str, default='cpu', help='device')
    parser.add_argument('--img_size', type=int, default=64, help='image size')
    return parser.parse_args()

args = parse_args()
device = torch.device(args.device)
onnx_file = args.model_path.replace('.pth', '.onnx')
H, W = args.img_size, args.img_size

#.Prepare model    ​
model = GenderNet()
checkpoint = torch.load(args.model_path, map_location=device)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

#.Convert to ONNX
dummy_input = torch.zeros(1, 3, H, W)
output = model(dummy_input)
torch.onnx.export(model, dummy_input, onnx_file, verbose=True, input_names=['input'], output_names=['output'], opset_version=14,\
                   do_constant_folding=True, export_params=True)

print('ONNX conversion done : {}'.format(onnx_file))