#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""

import argparse

from matplotlib import pyplot as plt

import editdistance
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from conf import settings
from utils import get_network, get_test_dataloader

from CombinedModel import CombinedModel

def levi_distance(pred_seq, true_seq, class_diff, pad_idx=26):
    pred_seq = ''.join([chr(char + 97) if char != pad_idx else '!' for char in pred_seq.cpu().numpy()])
    true_seq = ''.join([chr(char + 97) if char != pad_idx else '!' for char in true_seq.cpu().numpy()])
            
    if (pred_seq == 'zc!!!' and true_seq == 'zc!!!'):   #both are hiragana
        return 1 - class_diff
    elif (pred_seq == 'zc!!!' or true_seq == 'zc!!!'):  #either pred or true is hiragana
        return 5
    else: 
        return editdistance.eval(pred_seq, true_seq)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    args = parser.parse_args()

    from models.vgg import vgg16_bn
    net = CombinedModel(vgg16_bn(num_classes=952).cuda(), vgg16_bn(num_classes=135).cuda())

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
    )

    net.load_state_dict(torch.load(args.weights))
    print(net)
    net.eval()

    total_levenshtein_distance = 0.0
    total_length = 0

    with torch.no_grad():
        for n_iter, (image, class_label, seq_label) in enumerate(cifar100_test_loader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))

            if args.gpu:
                image = image.cuda()
                class_label = class_label.cuda()
                seq_label = seq_label.cuda()

            class_output, seq_output = net(image)
            _, class_pred = class_output.max(1)
            batch_size = seq_output.size(0)
            seq_output = seq_output.view(batch_size, 5, 27)  # Adjust based on actual sequence length and vocab size
            seq_preds = seq_output.argmax(dim=2)

            for i in range(seq_label.size(0)):
                levenshtein_dist = levi_distance(seq_preds[i], seq_label[i], class_pred[i].eq(class_label[i]).sum())
                total_levenshtein_distance += levenshtein_dist
                total_length += 5

    avg_levenshtein_distance = float(total_levenshtein_distance) / total_length

    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')

    print()
    print("Accuracy: ", 1 - avg_levenshtein_distance)
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))