# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import sys
import argparse
import time
from datetime import datetime

import editdistance
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights

from CombinedModel import CombinedModel

def train(epoch):

    start = time.time()
    net.train()
    for batch_index, (images, class_labels, seq_labels) in enumerate(cifar100_training_loader):

        if args.gpu:
            class_labels = class_labels.cuda()
            seq_labels = seq_labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        class_output, seq_output = net(images)
        loss = loss_function(class_output, class_labels, seq_output, seq_labels, class_criterion, seq_criterion)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(cifar100_training_loader.dataset)
        ))

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= args.warm:
            warmup_scheduler.step()

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(epoch=0, tb=True):

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0
    total_levenshtein_distance= 0
    total_length = 0

    for (images, class_labels, seq_labels) in cifar100_test_loader:

        if args.gpu:
            class_labels = class_labels.cuda()
            seq_labels = seq_labels.cuda()
            images = images.cuda()

        class_output, seq_output = net(images)
        loss = loss_function(class_output, class_labels, seq_output, seq_labels, class_criterion, seq_criterion)

        test_loss += loss.item()
        # Class prediction accuracy
        _, class_preds = class_output.max(1)
        correct += class_preds.eq(class_labels).sum().item()
        
        # Reshape seq_output from (batch_size, 135) to (batch_size, 5, 27)
        batch_size = seq_output.size(0)
        seq_output = seq_output.view(batch_size, 5, 27)
        seq_preds = seq_output.argmax(dim=2)

        pad_idx = 26 # character '!'
        for i in range(seq_labels.size(0)):
            pred_seq = ''.join([chr(char + 97) if char != pad_idx else '!' for char in seq_preds[i].cpu().numpy()])
            true_seq = ''.join([chr(char + 97) if char != pad_idx else '!' for char in seq_labels[i].cpu().numpy()])
            
            levenshtein_dist = 0
            if (pred_seq == 'zc!!!' and true_seq == 'zc!!!'):   #both are hiragana
                levenshtein_dist = 1 - (class_preds[i].eq(class_labels[i]).sum())
            elif (pred_seq == 'zc!!!' and true_seq != 'zc!!!'):  #pred is hiragana
                levenshtein_dist = len(true_seq.replace('!', ''))
            elif (pred_seq != 'zc!!!' and true_seq == 'zc!!!'):  #true is hiragana
                levenshtein_dist = len(pred_seq.replace('!', ''))
            else: 
                levenshtein_dist = editdistance.eval(pred_seq, true_seq)
            
            total_levenshtein_distance += levenshtein_dist
            total_length += len(true_seq.replace('!', ''))

    seq_accuracy = 1 - (float(total_levenshtein_distance) / total_length)

    finish = time.time()
    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(cifar100_test_loader.dataset),
        seq_accuracy,
        finish - start
    ))
    print()

    #add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', seq_accuracy, epoch)

    return seq_accuracy

def loss_function(class_output, class_target, seq_output, seq_target, class_criterion, seq_criterion, alpha=0.5):
    class_loss = class_criterion(class_output, class_target)
    
    # Reshape seq_output from (batch_size, 135) to (batch_size, 5, 27)
    batch_size = seq_output.size(0)
    seq_output = seq_output.view(batch_size, 5, 27)
    
    # Reshape seq_output and seq_target for the loss function
    seq_output = seq_output.view(-1, seq_output.size(-1))
    seq_target = seq_target.view(-1)
    
    seq_loss = seq_criterion(seq_output, seq_target)
    return alpha * class_loss + (1 - alpha) * seq_loss

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    args = parser.parse_args()

    from models.vgg import vgg16_bn
    net = CombinedModel(vgg16_bn(num_classes=952).cuda(), vgg16_bn(num_classes=135).cuda())

    #data preprocessing:
    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )
    
    # Forward pass
    class_criterion = nn.CrossEntropyLoss()
    seq_criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)

    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    #since tensorboard can't overwrite old values
    #so the only way is to create a new tensorboard log
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))
    input_tensor = torch.Tensor(1, 1, 64, 64)
    if args.gpu:
        input_tensor = input_tensor.cuda()
    writer.add_graph(net, input_tensor)
    
    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    if args.resume:
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))


    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue

        train(epoch)
        acc = eval_training(epoch)

        #start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)

    writer.close()
