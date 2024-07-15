#!/usr/bin/env python3

""" Inference script for neural network
Uses a specified weights file to perform inference on a single input image
and prints the top-1 and top-5 predictions.

author baiyu
"""

import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
from utils import get_network
from conf import settings

def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('L')
    if transform is not None:
        image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def infer_image(image_path, net, device):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    image = load_image(image_path, transform)
    image = image.to(device)

    net.eval()
    with torch.no_grad():
        output = net(image)
        _, top1_pred = output.topk(1, 1, largest=True, sorted=True)
        _, top5_pred = output.topk(5, 1, largest=True, sorted=True)

        return top1_pred, top5_pred

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='network type')
    parser.add_argument('-weights', type=str, required=True, help='path to the weights file')
    parser.add_argument('-image', type=str, required=True, help='path to the input image')
    parser.add_argument('-gpu', action='store_true', default=False, help='use GPU for inference')
    args = parser.parse_args()

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    net = get_network(args).to(device)
    net.load_state_dict(torch.load(args.weights, map_location=device))

    top1_pred, top5_pred = infer_image(args.image, net, device)
    print(f'Top 1 Prediction: {top1_pred.item()}')
    print(f'Top 5 Predictions: {top5_pred.cpu().numpy().flatten()}')

if __name__ == '__main__':
    main()
