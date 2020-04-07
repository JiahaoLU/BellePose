from TrainingValidationTest import *
import json


class HyperFinder():
    def __init__(self):
        self.DATASET_PATH = "./Data/train.csv"
        self.EPOCH = 1000
        self.LEARNING_RATE = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
        self.BATCH_SIZE = [40 * i for i in range(1, 11)]
        self.WEIGHT_DECAY = [pow(10, -6 + i) for i in range(5)]
        self.EMBED_DIM = [pow(2, i) for i in range(5)]
        self.save_best = True

        self.best_learning_rate = None
        self.model_best = None
        self.epoch_list = []
        self.train_loss_list = []
        self.val_acc_list = []
        self.val_acc_list = []
        self.val_loss_list = []
        self.auc_best = 0
        self.acc_best = 1
        self.loss_best = 999999
        self.Total_Epoch = 1
        self.data_save = {}
        self.data_saveE = {}
        self.data_saveW = {}

    def save_model(self,trained_model,*args):
        model_name = ''
        for i in args:
            print(i)
            model_name += (str(i)[:7] + ' ')
        torch.save(trained_model, f'{model_name}.pt')

    def obj_2_json(self, hpname, load_dict):
        with open("./" + hpname + ".json", "w") as dump_f:
            json.dump(load_dict, dump_f)

    def find_best_Learning_Rate(self):
        BATCH_SIZE = 12800
        WEIGHT_DECAY = 1e-6
        EMBED_DIM = 8
        for LEARNING_RATE in self.LEARNING_RATE:
            self.data_save[str(LEARNING_RATE)] = {}
            trained_model, epoch_list, train_loss_list, val_auc_list, val_acc_list, val_loss_list, test_auc, test_acc, test_loss, \
            best_model, best_val_auc, best_val_acc, best_val_loss = \
                main_process(self.DATASET_PATH, self.EPOCH, LEARNING_RATE, BATCH_SIZE, WEIGHT_DECAY, EMBED_DIM,
                             5 * 60)
            self.data_save[str(LEARNING_RATE)]['EPOCH'] = epoch_list
            self.data_save[str(LEARNING_RATE)]['train_loss_list'] = train_loss_list
            self.data_save[str(LEARNING_RATE)]['val_auc_list'] = val_auc_list
            self.data_save[str(LEARNING_RATE)]['val_acc_list'] = val_acc_list
            self.data_save[str(LEARNING_RATE)]['val_loss_list'] = val_loss_list
            self.data_save[str(LEARNING_RATE)]['test_auc'] = test_auc
            self.data_save[str(LEARNING_RATE)]['test_acc'] = test_acc
            self.data_save[str(LEARNING_RATE)]['test_loss'] = test_loss
            self.data_save[str(LEARNING_RATE)]['best_val_auc'] = best_val_auc
            self.data_save[str(LEARNING_RATE)]['best_val_acc'] = best_val_acc
            self.data_save[str(LEARNING_RATE)]['best_val_loss'] = best_val_loss
            if test_loss < self.loss_best:
                self.best_learning_rate = LEARNING_RATE
                self.epoch_list = epoch_list
                self.auc_best = test_auc
                self.acc_best = test_acc
                self.loss_best = test_loss
                self.Total_Epoch = len(epoch_list)
                self.model_best = trained_model

        if self.save_best:
            self.save_model(self.model_best, self.best_learning_rate, self.auc_best, self.acc_best, self.loss_best)

    def find_best_Weight_decay(self):
        BATCH_SIZE = 12800
        LEARNING_RATE = 0.01
        EMBED_DIM = 8
        for WEIGHT_DECAY in self.WEIGHT_DECAY:
            self.data_saveW[str(WEIGHT_DECAY)] = {}
            trained_model, epoch_list, train_loss_list, val_auc_list, val_acc_list, val_loss_list, test_auc, test_acc, test_loss, \
            best_model, best_val_auc, best_val_acc, best_val_loss = \
                main_process(self.DATASET_PATH, self.EPOCH, LEARNING_RATE, BATCH_SIZE, WEIGHT_DECAY, EMBED_DIM,
                             5 * 60)
            self.data_saveW[str(WEIGHT_DECAY)]['EPOCH'] = epoch_list
            self.data_saveW[str(WEIGHT_DECAY)]['train_loss_list'] = train_loss_list
            self.data_saveW[str(WEIGHT_DECAY)]['val_auc_list'] = val_auc_list
            self.data_saveW[str(WEIGHT_DECAY)]['val_acc_list'] = val_acc_list
            self.data_saveW[str(WEIGHT_DECAY)]['val_loss_list'] = val_loss_list
            self.data_saveW[str(WEIGHT_DECAY)]['test_auc'] = test_auc
            self.data_saveW[str(WEIGHT_DECAY)]['test_acc'] = test_acc
            self.data_saveW[str(WEIGHT_DECAY)]['test_loss'] = test_loss
            self.data_saveW[str(WEIGHT_DECAY)]['best_val_auc'] = best_val_auc
            self.data_saveW[str(WEIGHT_DECAY)]['best_val_acc'] = best_val_acc
            self.data_saveW[str(WEIGHT_DECAY)]['best_val_loss'] = best_val_loss
            if test_loss < self.loss_best:
                self.best_learning_rate = LEARNING_RATE
                self.epoch_list = epoch_list
                self.auc_best = test_auc
                self.acc_best = test_acc
                self.loss_best = test_loss
                self.Total_Epoch = len(epoch_list)
                self.model_best = trained_model

        if self.save_best:
            self.save_model(self.model_best, self.best_learning_rate, self.auc_best, self.acc_best, self.loss_best)

    def find_best_k(self):
        BATCH_SIZE = 12800
        WEIGHT_DECAY = 1e-6
        LEARNING_RATE = 0.001
        for EMBED_DIM in self.EMBED_DIM:
            self.data_saveE[str(EMBED_DIM)] = {}
            trained_model, epoch_list, train_loss_list, val_auc_list, val_acc_list, val_loss_list, test_auc, test_acc, test_loss, \
            best_model, best_val_auc, best_val_acc, best_val_loss = \
                main_process(self.DATASET_PATH, self.EPOCH, LEARNING_RATE, BATCH_SIZE, WEIGHT_DECAY, EMBED_DIM,
                             5 * 60)
            self.data_saveE[str(EMBED_DIM)]['EPOCH'] = epoch_list
            self.data_saveE[str(EMBED_DIM)]['train_loss_list'] = train_loss_list
            self.data_saveE[str(EMBED_DIM)]['val_auc_list'] = val_auc_list
            self.data_saveE[str(EMBED_DIM)]['val_acc_list'] = val_acc_list
            self.data_saveE[str(EMBED_DIM)]['val_loss_list'] = val_loss_list
            self.data_saveE[str(EMBED_DIM)]['test_auc'] = test_auc
            self.data_saveE[str(EMBED_DIM)]['test_acc'] = test_acc
            self.data_saveE[str(EMBED_DIM)]['test_loss'] = test_loss
            self.data_saveE[str(EMBED_DIM)]['best_val_auc'] = best_val_auc
            self.data_saveE[str(EMBED_DIM)]['best_val_acc'] = best_val_acc
            self.data_saveE[str(EMBED_DIM)]['best_val_loss'] = best_val_loss
            if test_loss < self.loss_best:
                self.best_learning_rate = LEARNING_RATE
                self.epoch_list = epoch_list
                self.auc_best = test_auc
                self.acc_best = test_acc
                self.loss_best = test_loss
                self.Total_Epoch = len(epoch_list)
                self.model_best = trained_model

        if self.save_best:
            self.save_model(self.model_best, self.best_learning_rate, self.auc_best, self.acc_best, self.loss_best)

def main():
    hyperFinder = HyperFinder()
    hyperFinder.find_best_Learning_Rate()
    hyperFinder.obj_2_json('learning_rate', hyperFinder.data_save)
    hyperFinder.find_best_Weight_decay()
    hyperFinder.obj_2_json('Weight_decay', hyperFinder.data_saveW)
    hyperFinder.find_best_k()
    hyperFinder.obj_2_json('EMBED_DIM', hyperFinder.data_saveE)


if __name__ == "__main__":
    main()

