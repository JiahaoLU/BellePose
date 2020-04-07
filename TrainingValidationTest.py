import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from DataSet import *
from torchvision import datasets, transforms, models
import torch.optim as optim
from sklearn.model_selection import train_test_split
import threading


def fun_timer():
    global boom
    boom = True


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
    for i, (fields, target) in enumerate(data_loader):
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


def run_test(model, data_loader, device, criterion):
    """
    evaluate / test the model
    :param model: the model to be evaluated/tested. instance of subclass of nn.Module
    :param data_loader: torch DataLoader of eval/test data set
    :param device: CUDA GPU or CPU
    :return: auc score, accuracy of prediction
    """
    model.eval()
    targets, predicts = list(), list()
    correct = 0
    with torch.no_grad():
        for fields, target in data_loader:
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            loss = criterion(y, target.float())
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
            predict_click = torch.round(y.data)
            correct += (predict_click == target).sum().item()
    return roc_auc_score(targets, predicts), correct / len(targets) * 100, loss.item()


def main_process(dataset_path, epoch, learning_rate, batch_size, weight_decay, embeddim, boomtime = 60):
    """
    Main process for train/evaluate/test the model, determine the hyper parameters here.
    :param boomtime: time boom for adjusting hyperparameters to control the same training time cost
    :param dataset_path: path of the original csv file
    :param epoch: number of epochs
    :param learning_rate: learning rate of gradient descent
    :param batch_size: size of batches
    :param weight_decay: L2 regularisation
    :param embeddim: dimension of latent vector
    :return: the trained model
    """
    global boom
    boom = False
    timer = threading.Timer(boomtime, fun_timer)
    timer.start()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))

    # Prepare the data
    dataset = MPIIDataset(dataset_path)
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length))
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    field_dims = dataset.get_anno_dims()

    # Prepare the model and loss function
    model = FieldAwareFactorizationMachineModel(field_dims, embed_dim=embeddim).to(device)
    criterion = torch.nn.BCELoss().to(device)  # binary cross entropy loss
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    epoch_list = []
    train_loss_list = []
    val_auc_list = []
    val_acc_list = []
    val_loss_list = []
    best_model = None
    best_val_auc = 0
    best_val_acc = 0
    best_val_loss = 999999

    # train
    for epoch_i in range(epoch):
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')
        train_loss = run_train(model, optimizer, train_data_loader, criterion, device)
        val_auc, val_acc,val_loss = run_test(model, valid_data_loader, device, criterion)
        epoch_list.append(epoch_i)
        train_loss_list.append(train_loss)
        val_auc_list.append(val_auc)
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)
        print('epoch:', epoch_i, 'train loss:', train_loss, 'validation: auc:',\
              val_auc, '--- acc:', val_acc, '--- loss:', val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            best_val_auc = val_auc
            best_val_acc = val_acc
        if boom:
            print('time up, break')
            break
        else:
            print(timer)
    test_auc, test_acc, test_loss = run_test(model, test_data_loader, device, criterion)
    print('test auc:', test_auc, 'test acc:', test_acc, 'test loss', test_loss)
    return model, epoch_list, train_loss_list, val_auc_list, val_acc_list, val_loss_list, \
           test_auc, test_acc, test_loss, best_model, best_val_auc, best_val_acc, best_val_loss


def main(save_model=True):
    DATASET_PATH = "./Data/train20k.csv"
    EPOCH = 1000
    LEARNING_RATE = 0.001
    BATCH_SIZE = 3200
    WEIGHT_DECAY = 1e-6
    EMBED_DIM = 10
    trained_model,epoch_list,train_loss_list,val_auc_list,val_acc_list,val_loss_list,auc,acc,loss,\
    best_model,best_val_auc,best_val_acc,best_val_loss = \
        main_process(DATASET_PATH, EPOCH, LEARNING_RATE, BATCH_SIZE, WEIGHT_DECAY, EMBED_DIM)
    print(trained_model,epoch_list,train_loss_list,val_auc_list,val_acc_list,val_loss_list,auc,acc,loss,\
    best_model,best_val_auc,best_val_acc,best_val_loss)
    # if save_model:
    #     model_name = "FFM"
    #     torch.save(trained_model, f'{model_name}.pt')


if __name__ == "__main__":
    main()
