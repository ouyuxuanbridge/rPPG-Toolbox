"""Trainer for TSCAN."""

import logging
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from metrics.metrics import calculate_metrics
from neural_methods.loss.NegPearsonLoss import Neg_Pearson
from neural_methods.model.TS_CAN import TSCAN
from neural_methods.trainer.BaseTrainer import BaseTrainer
from tqdm import tqdm
from neural_methods.trainer.TrivialAugmentTemporal import TrivialAugmentTemporal
from neural_methods.trainer.TrivialAugmentTemporal import *
from neural_methods.trainer.TrivialAugment   import TrivialAugment
from neural_methods.trainer.TrivialAugment   import *
opn=np.zeros((11,5))
class TscanTrainer(BaseTrainer):

    def __init__(self, config):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.frame_depth = config.MODEL.TSCAN.FRAME_DEPTH
        self.model = TSCAN(frame_depth=self.frame_depth, img_size=config.TRAIN.DATA.PREPROCESS.H).to(self.device)
        self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
        self.criterion = torch.nn.MSELoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=config.TRAIN.LR, weight_decay=0)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.base_len = self.num_of_gpu * self.frame_depth
        self.chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
        self.config = config
        self.best_epoch = 0

    def train(self, data_loader):
        """ TODO:Docstring"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")
        min_valid_loss = 1
        for epoch in range(self.max_epoch_num):
            print(f"====Training Epoch: {epoch}====")
            running_loss = 0.0
            train_loss = []
            self.model.train()
            # Model Training
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                data, labels = batch[0].to(
                    self.device), batch[1].to(self.device)
                N, D, C, H, W = data.shape
                data, labels =self.augmentation(data,labels)
                

                C=C*2
                data = data.view(N * D, C, H, W)
                labels = labels.view(-1, 1)
                data = data[:(N * D) // self.base_len * self.base_len]
                labels = labels[:(N * D) // self.base_len * self.base_len]
                self.optimizer.zero_grad()
                pred_ppg = self.model(data)
                loss = self.criterion(pred_ppg, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if idx % 100 == 99:  # print every 100 mini-batches
                    print(
                        f'[{epoch + 1}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                train_loss.append(loss.item())
                tbar.set_postfix(loss=loss.item())
            valid_loss = self.valid(data_loader)
            self.save_model(epoch)
            print('validation loss: ', valid_loss)
            if (valid_loss < min_valid_loss) or (valid_loss < 0):
                min_valid_loss = valid_loss
                self.best_epoch = epoch
                print("Update best model! Best epoch: {}".format(self.best_epoch))
                self.save_model(epoch)
        print("best trained epoch:{}, min_val_loss:{}".format(self.best_epoch, min_valid_loss))

    def valid(self, data_loader):
        """ Model evaluation on the validation dataset."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")
        print("===Validating===")
        valid_loss = []
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                data_valid, labels_valid = valid_batch[0].to(
                    self.device), valid_batch[1].to(self.device)
                N, D, C, H, W = data_valid.shape
                data_valid = data_valid.view(N * D, C, H, W)
                labels_valid = labels_valid.view(-1, 1)
                data_valid = data_valid[:(N * D) // self.base_len * self.base_len]
                labels_valid = labels_valid[:(N * D) // self.base_len * self.base_len]
                pred_ppg_valid = self.model(data_valid)
                loss = self.criterion(pred_ppg_valid, labels_valid)
                valid_loss.append(loss.item())
                valid_step += 1
                vbar.set_postfix(loss=loss.item())
            valid_loss = np.asarray(valid_loss)
        return np.mean(valid_loss)

    def test(self, data_loader):
        """ Model evaluation on the testing dataset."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        print("===Testing===")
        np.save("Trivialaugmentv1opn.npy",opn)
        predictions = dict()
        labels = dict()
        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
        else:
            best_model_path = os.path.join(
                self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
            print("Testing uses non-pretrained model!")
            print(best_model_path)
            self.model.load_state_dict(torch.load(best_model_path))
        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        with torch.no_grad():
            for _, test_batch in enumerate(data_loader['test']):
                batch_size = test_batch[0].shape[0]
                data_test, labels_test = test_batch[0].to(
                    self.config.DEVICE), test_batch[1].to(self.config.DEVICE)
                N, D, C, H, W = data_test.shape
                data_test = data_test.view(N * D, C, H, W)
                labels_test = labels_test.view(-1, 1)
                data_test = data_test[:(N * D) // self.base_len * self.base_len]
                labels_test = labels_test[:(N * D) // self.base_len * self.base_len]
                pred_ppg_test = self.model(data_test)
                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]
                    labels[subj_index][sort_index] = labels_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]

        calculate_metrics(predictions, labels, self.config)

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)


    def augmentation(self, data, labels):
        N, D, C, H, W = data.shape
        C=2*C
        data_numpy = data.detach().cpu().permute(0,1,3,4,2).numpy()/255
        label_numpy = labels.detach().cpu().numpy()
        augmenter = TrivialAugmentTemporal()
        #np.save("beforee.npy",data_numpy[0,0,:,:,:])
        #np.save("labelbef.npy",label_numpy[0,:])
        data_numpy, labels, op, level = augmenter(data_numpy, label_numpy)
        #np.save("aftere.npy",data_numpy[0,0,:,:,:])
        #np.save("labelaft.npy",labels[0,:])
        #import sys
        #sys.exit()

        self.collect(op, level)
        labels = torch.from_numpy(np.float32(labels).copy()).to(self.device)    
        data_stack_list = []
        for batch_idx in range(N):
            
            video_aug = data_numpy[batch_idx,:,:,:,:]
            diff_normalize_data_part = self.diff_normalize_data(video_aug)
            standardized_data_part = self.standardized_data(video_aug)
            cat_data = np.concatenate((diff_normalize_data_part,standardized_data_part),axis=3)
            data_stack_list.append(cat_data)
        data_stack = np.asarray(data_stack_list)
        data_stack_tensor = torch.zeros([N, D, C, H, W], dtype = torch.float).to(self.device)
        for batch_idx in range(N):
            data_stack_tensor[batch_idx] = torch.from_numpy(data_stack[batch_idx]).permute(0,3,1,2).to(self.device)
        data = data_stack_tensor
        

        return data, labels

    def diff_normalize_data(self, data):
        """Difference frames and normalization data"""
        normalized_len = len(data)
        h, w, c = data[0].shape
        normalized_data = np.zeros((normalized_len, h, w, c), dtype=np.float32)
        normalized_data[normalized_len-1] = (data[normalized_len-1] - data[normalized_len-2]) / (
                    data[normalized_len-1] + data[normalized_len-2] + 1e-7)
        for j in range(normalized_len - 1):
            normalized_data[j] = (data[j + 1] - data[j]) / (
                    data[j + 1] + data[j] + 1e-7)
        normalized_data = normalized_data / np.std(normalized_data)
        normalized_data[np.isnan(normalized_data)] = 0
        return normalized_data

    def standardized_data(self, data):
        """Difference frames and normalization data"""
        data = np.asarray(data)
        data = data - np.mean(data)
        data = data / np.std(data)
        data[np.isnan(data)] = 0
        return data

    def collect(self,op,level):

        #opn = np.zeros((2, 5), int)
        if op==gaussianlabel:
            if level==0:
                opn[0][0]=opn[0][0]+1
            elif level==1:
                opn[0][1]=opn[0][1]+1
            elif level==2:
                opn[0][2]=opn[0][2]+1
            elif level==3:
                opn[0][3]=opn[0][3]+1
            elif level==4:
                opn[0][4]=opn[0][4]+1
            else:
                print("error")

        elif op==labelwarping:
            if level==0:
                opn[1][0]=opn[1][0]+1
            elif level==1:
                opn[1][1]=opn[1][1]+1
            elif level==2:
                opn[1][2]=opn[1][2]+1
            elif level==3:
                opn[1][3]=opn[1][3]+1
            elif level==4:
                opn[1][4]=opn[1][4]+1
            else:
                print("error")

        elif op == scaling:
            if level == 0:
                opn[2][0] = opn[2][0] + 1
            elif level == 1:
                opn[2][1] = opn[2][1] + 1
            elif level == 2:
                opn[2][2] = opn[2][2] + 1
            elif level == 3:
                opn[2][3] = opn[2][3] + 1
            elif level == 4:
                opn[2][4] = opn[2][4] + 1
            else:
                print("error")

        elif op==translate_y:
            if level==0:
                opn[3][0]=opn[3][0]+1
            elif level==1:
                opn[3][1]=opn[3][1]+1
            elif level==2:
                opn[3][2]=opn[3][2]+1
            elif level==3:
                opn[3][3]=opn[3][3]+1
            elif level==4:
                opn[3][4]=opn[3][4]+1
            else:
                print("error")

        elif op==rotate:
            if level==0:
                opn[4][0]=opn[4][0]+1
            elif level==1:
                opn[4][1]=opn[4][1]+1
            elif level==2:
                opn[4][2]=opn[4][2]+1
            elif level==3:
                opn[4][3]=opn[4][3]+1
            elif level==4:
                opn[4][4]=opn[4][4]+1
            else:
                print("error")
        elif op==flip_lr:
            if level==0:
                opn[5][0]=opn[5][0]+1
            elif level==1:
                opn[5][1]=opn[5][1]+1
            elif level==2:
                opn[5][2]=opn[5][2]+1
            elif level==3:
                opn[5][3]=opn[5][3]+1
            elif level==4:
                opn[6][4]=opn[6][4]+1
            else:
                print("error")
        
        elif op==lowfreqnoise:
            if level==0:
                opn[6][0]=opn[6][0]+1
            elif level==1:
                opn[6][1]=opn[6][1]+1
            elif level==2:
                opn[6][2]=opn[6][2]+1
            elif level==3:
                opn[6][3]=opn[6][3]+1
            elif level==4:
                opn[6][4]=opn[6][4]+1
            else:
                print("error")

        elif op==videonoise:
            if level==0:
                opn[7][0]=opn[7][0]+1
            elif level==1:
                opn[7][1]=opn[7][1]+1
            elif level==2:
                opn[7][2]=opn[7][2]+1
            elif level==3:
                opn[7][3]=opn[7][3]+1
            elif level==4:
                opn[7][4]=opn[7][4]+1
            else:
                print("error")
        elif op==videonoise:
            if level==0:
                opn[7][0]=opn[7][0]+1
            elif level==1:
                opn[7][1]=opn[7][1]+1
            elif level==2:
                opn[7][2]=opn[7][2]+1
            elif level==3:
                opn[7][3]=opn[7][3]+1
            elif level==4:
                opn[7][4]=opn[7][4]+1
            else:
                print("error")

        elif op==shear_x:
            if level==0:
                opn[8][0]=opn[8][0]+1
            elif level==1:
                opn[8][1]=opn[8][1]+1
            elif level==2:
                opn[8][2]=opn[8][2]+1
            elif level==3:
                opn[8][3]=opn[8][3]+1
            elif level==4:
                opn[8][4]=opn[8][4]+1
            else:
                print("error")
        elif op==shear_y:
            if level==0:
                opn[9][0]=opn[9][0]+1
            elif level==1:
                opn[9][1]=opn[9][1]+1
            elif level==2:
                opn[9][2]=opn[9][2]+1
            elif level==3:
                opn[9][3]=opn[9][3]+1
            elif level==4:
                opn[9][4]=opn[9][4]+1
            else:
                print("error")
        elif op==shear_y:
            if level==0:
                opn[10][0]=opn[10][0]+1
            elif level==1:
                opn[10][1]=opn[10][1]+1
            elif level==2:
                opn[10][2]=opn[10][2]+1
            elif level==3:
                opn[10][3]=opn[10][3]+1
            elif level==4:
                opn[10][4]=opn[10][4]+1
            else:
                print("error")




