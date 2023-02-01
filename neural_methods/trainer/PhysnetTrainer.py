"""PhysNet Trainer."""
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from metrics.metrics import calculate_metrics
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from neural_methods.trainer.BaseTrainer import BaseTrainer
from torch.autograd import Variable
from tqdm import tqdm
from neural_methods.trainer.TrivialAugmentTemporal128 import TrivialAugmentTemporal128
from neural_methods.trainer.TrivialAugmentTemporal128 import *
from neural_methods.trainer.TrivialAugment128   import TrivialAugment128
from neural_methods.trainer.TrivialAugment128   import *

class PhysnetTrainer(BaseTrainer):

    def __init__(self, config):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.model = PhysNet_padding_Encoder_Decoder_MAX(
            frames=config.MODEL.PHYSNET.FRAME_NUM).to(self.device)  # [3, T, 128,128]
        self.loss_model = Neg_Pearson()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=config.TRAIN.LR)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.base_len = self.num_of_gpu
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
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                #print("batch0")
                #print(batch[0].shape)
                #print(batch[1].shape)
                #print(type(batch[0]))
                #np.save(batch[0][0,:,:,:,:],"physnetvideo.npy")
                #import sys
                #sys.exit()
                batch[0], batch[1] =self.augmentation(batch[0],batch[1])
                rPPG, x_visual, x_visual3232, x_visual1616 = self.model(
                    batch[0].to(torch.float32).to(self.device))

                BVP_label = batch[1].to(
                    torch.float32).to(self.device)
                rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
                BVP_label = (BVP_label - torch.mean(BVP_label)) / \
                            torch.std(BVP_label)  # normalize
                loss = self.loss_model(rPPG, BVP_label)
                loss.backward()
                running_loss += loss.item()
                if idx % 100 == 99:  # print every 100 mini-batches
                    print(
                        f'[{epoch + 1}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                train_loss.append(loss.item())
                self.optimizer.step()
                self.optimizer.zero_grad()
                tbar.set_postfix(loss=loss.item())
            valid_loss = self.valid(data_loader)
            self.save_model(epoch)
            print('validation loss: ', valid_loss)
            if (valid_loss < min_valid_loss) or (valid_loss < 0):
                min_valid_loss = valid_loss
                self.best_epoch = epoch
                print("Update best model! Best epoch: {}".format(self.best_epoch))
                self.save_model(epoch)
        print("best trained epoch:{}, min_val_loss:{}".format(
            self.best_epoch, min_valid_loss))

    def valid(self, data_loader):
        """ Runs the model on valid sets."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")
        print(" ====Validing===")
        valid_loss = []
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                BVP_label = valid_batch[1].to(
                    torch.float32).to(self.device)
                rPPG, x_visual, x_visual3232, x_visual1616 = self.model(
                    valid_batch[0].to(torch.float32).to(self.device))
                rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
                BVP_label = (BVP_label - torch.mean(BVP_label)) / \
                            torch.std(BVP_label)  # normalize
                loss_ecg = self.loss_model(rPPG, BVP_label)
                valid_loss.append(loss_ecg.item())
                valid_step += 1
                vbar.set_postfix(loss=loss_ecg.item())
            valid_loss = np.asarray(valid_loss)
        return np.mean(valid_loss)

    def test(self, data_loader):
        """ Runs the model on test sets."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        print("===Testing===")
        predictions = dict()
        labels = dict()
        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
            print(self.config.INFERENCE.MODEL_PATH)
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
                data, label = test_batch[0].to(
                    self.config.DEVICE), test_batch[1].to(self.config.DEVICE)
                pred_ppg_test, _, _, _ = self.model(data)
                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx]
                    labels[subj_index][sort_index] = label[idx]
        calculate_metrics(predictions, labels, self.config)

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)

    def augmentation(self, data, labels):
        #print("datashape")
        #print(data.shape)
        N, C, D, H, W = data.shape

        data_numpy = data.detach().cpu().permute(0, 2, 3, 4, 1).numpy() / 255
        label_numpy = labels.detach().cpu().numpy()
        augmenter = TrivialAugmentTemporal128()
        #np.save("beforee.npy",data_numpy[0,0,:,:,:])
        #np.save("labelbef.npy",label_numpy[0,:])
        #print("data_numpy")
        #print(data_numpy.shape)
        data_numpy, labels, op, level = augmenter(data_numpy, label_numpy)
        #np.save("aftere.npy",data_numpy[0,0,:,:,:])
        #np.save("labelaft.npy",labels[0,:])
        #import sys
        #sys.exit()

        #self.collect(op, level)
        labels = torch.from_numpy(np.float32(labels).copy()).to(self.device)
        data_stack_list = []
        for batch_idx in range(N):
            video_aug = data_numpy[batch_idx, :, :, :, :]
            diff_normalize_data_part = self.diff_normalize_data(video_aug)
            #cat_data = np.concatenate((diff_normalize_data_part, standardized_data_part), axis=3)
            data_stack_list.append(diff_normalize_data_part)
            #print(diff_normalize_data_part.shape)
        data_stack = np.asarray(data_stack_list)
        data_stack_tensor = torch.zeros([N, C, D, H, W], dtype=torch.float).to(self.device)
        for batch_idx in range(N):
            #print(torch.from_numpy(data_stack[batch_idx]).shape)
            data_stack_tensor[batch_idx] = torch.from_numpy(data_stack[batch_idx]).permute(3,0,1,2).to(self.device)
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
