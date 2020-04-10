# encoding: utf-8
"""
@author: Jiahao LU
@contact: lujiahao8146@gmail.com
@file: Train.py
@time: 2020/4/8
@desc:
"""
import numpy as np
import torch

from torch.utils.data import random_split, DataLoader
from DataSet import MPIIDataset
from ImgUtil import calc_bbox_size
import os
from datetime import datetime
from copy import deepcopy
from Models import ResNet

def adjust_learning_rate(optimizer, epoch, drop_stride, frac):
    lrfac = frac if epoch % drop_stride == 0 else 1
    for i, param_group in enumerate(optimizer.param_groups):
        if lrfac != 1:
            print("Reducing learning rate of group %d from %f to %f" %
                  (i, param_group['lr'], param_group['lr'] * lrfac))
        param_group['lr'] *= lrfac


def run_train(model, optimizer, data_loader, criterion, device, log_interval=10):
    """
    train the model using backward propagation
    :param model: the model to be trained. instance of subclass of nn.Module
    :param optimizer: torch optimiser with learning rate
    :param data_loader: torch DataLoader of training data set
    :param criterion: nn.BCELoss
    :param device: CUDA GPU or CPU
    :param log_interval: interval for showing training loss
    :return: none
    """
    model.train()
    total_loss = 0
    loss = criterion
    for i, (fields, target), _ in enumerate(data_loader):
        fields, target = fields.to(device), target.to(device)
        y = model(fields)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i % log_interval == 0:
            print('    - loss:', total_loss / log_interval)
            total_loss = 0
    return loss.item()


def run_test(model, data_loader, device, criterion, alpha=0.5):
    """
    evaluate / test the model
    Percentage of Correct Key-points wrt head (PCKh)
    :param model: the model to be evaluated/tested. instance of subclass of nn.Module
    :param data_loader: torch DataLoader of eval/test data set
    :param device: CUDA GPU or CPU
    :return: auc score, accuracy of prediction
    """
    model.eval()
    njoints = 0
    correct = 0
    with torch.no_grad():
        for quiz, target, headsize in data_loader:
            target_cpu = deepcopy(target)
            quiz, target = quiz.to(device), target.to(device)
            y = model(quiz)
            loss = criterion(y, target.float())
            njoints += 1
            if np.linalg.norm(y.cpu() - target_cpu) <= alpha * headsize:
                correct += 1
    return float(correct) / float(njoints), njoints, loss.item()


def save_checkpoint(model, path, epoch, optimizer, acc, loss):
    if os.path.exists(path):
        timestamp = str(datetime.timestamp(datetime.now()))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'acc': acc,
            'loss': loss,
        }, './default_output_' + timestamp)
        raise FileExistsError('Path already exists. save to ./default_output_' + timestamp)
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'acc': acc,
                'loss': loss,
                }, path)


def main_process(_model, dataset, sav_dir, epoch, learning_rate, batch_size,
                 is_reload=False, load_dir='', do_train=True, momentum=0):
    """
    Main process for train/evaluate/test the model, determine the hyper parameters here.
    :param load_dir:
    :param is_reload:
    :param sav_dir:
    :param dataset:
    :param _model:
    :param epoch: number of epochs
    :param learning_rate: learning rate of gradient descent
    :param batch_size: size of batches
    :param momentum: L2 regularisation
    :return: the trained model
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))

    # Prepare the data
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = random_split(
        dataset, (train_length, valid_length, test_length))
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Prepare the model and loss function

    if is_reload and os.path.exists(load_dir):
        checkpoint = torch.load(load_dir)
        _model.load_state_dict(checkpoint['model_state_dict'])
        _model = _model.to(device)
        _optimizer = torch.optim.RMSprop(params=_model.parameters(), lr=learning_rate, momentum=momentum)
        _optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('Reload model from last epoch %d, loss = %.5f, acc = %.5f%%' %
              (checkpoint['epoch'], checkpoint['loss'], checkpoint['acc'] * 100))
    else:
        if is_reload and not os.path.exists(load_dir):
            print('Model path not found. Start with new model.')
        _model = _model.to(device)
        _optimizer = torch.optim.RMSprop(params=_model.parameters(), lr=learning_rate, momentum=momentum)
    _criterion = torch.nn.MSELoss().to(device)

    # train
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')
    if do_train:
        for epoch_i in range(epoch):
            train_loss = run_train(_model, _optimizer, train_data_loader, _criterion, device)
            adjust_learning_rate(_optimizer, epoch_i, drop_stride=2, frac=0.7)
            val_acc, val_total, val_loss = run_test(_model, valid_data_loader, device, _criterion)

            print('epoch:', epoch_i, 'train loss:', train_loss)
            print('validation on %d examples: --- acc: %.5f\n' % (val_total, val_acc))

    test_acc, test_total, test_loss = run_test(_model, test_data_loader, device, _criterion)
    print('test loss: %.5f' % test_loss)
    print('test acc: %.5f on %d examples' % (test_acc, test_total))
    save_checkpoint(_model, sav_dir, -1, _optimizer, test_acc, test_loss)
    return _model


if __name__ == '__main__':
    bellepose = ResNet(256)
    ds = MPIIDataset(256)
    main_process(bellepose, sav_dir='./resnet50_model',
                 dataset=ds, epoch=3, learning_rate=2.5e-4, batch_size=20)