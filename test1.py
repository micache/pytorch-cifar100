#!/usr/bin/env python3

""" Inference script for neural network
Uses a specified weights file to perform inference on a single input image
and returns the predicted sequence.

author baiyu
"""

import argparse
import torch
from PIL import Image
import torchvision.transforms as transforms
from utils import get_network
from conf import settings
from CombinedModel import CombinedModel

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
        class_output, seq_output = net(image)
        _, class_pred = class_output.max(1)
        
        batch_size = seq_output.size(0)
        seq_output = seq_output.view(batch_size, 5, 27)  # Adjust based on actual sequence length and vocab size
        seq_preds = seq_output.argmax(dim=2)

        pred_seq = ''.join([chr(char + 97) if char != 26 else '!' for char in seq_preds[0].cpu().numpy()])

        return pred_seq, class_pred

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='network type')
    parser.add_argument('-weights', type=str, required=True, help='path to the weights file')
    parser.add_argument('-image', type=str, required=True, help='path to the input image')
    parser.add_argument('-gpu', action='store_true', default=False, help='use GPU for inference')
    args = parser.parse_args()

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    from models.vgg import vgg16_bn
    net = CombinedModel(vgg16_bn(num_classes=952).to(device), vgg16_bn(num_classes=135).to(device))

    net.load_state_dict(torch.load(args.weights, map_location=device))

    pred_seq, pred_class = infer_image(args.image, net, device)
    print(f'Predicted Sequence: {pred_seq}')
    print(pred_class)

if __name__ == '__main__':
    main()
