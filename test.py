import json
import os
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch
import pylab as pl
#from torch import CNNMEAE
from sklearn import preprocessing
from torch.utils.data import DataLoader
from preprocess import trainprepro
from matplotlib import pyplot as plt
#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#from dataset import BatchCollator
#from util import UnNormalize

#torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_tensor_type(torch.DoubleTensor)
class Tester():
    def __init__(self, cfg, dataloader, model, device):
        self.cfg = cfg
        self.sensor_length = cfg.sensor_length
        self.sensor_channel_size = cfg.sensor_channel_size
        #self.image_height = cfg.image_height
        #self.image_width = cfg.image_width
        self.image_channel_size = cfg.image_channel_size


        self.num_dataloaders = cfg.num_dataloaders
        self.device = device

        self.batch_size = cfg.batch_size

        self.model = model

        self.cls_loss_coef = cfg.cls_loss_coef
        self.entropy_loss_coef = cfg.entropy_loss_coef
        self.condi_loss_coef = cfg.condi_loss_coef
        self.addressing = cfg.addressing
        self.num_memories = cfg.num_memories

        #self.writer = SummaryWriter(self.log_dir + '/logs')

        self.cls_criterion = nn.CrossEntropyLoss(reduction='mean')
        self.rec_criterion = nn.MSELoss(reduction='sum')
        self.condi_criterion = nn.BCELoss(reduction='sum')
        #self.instances_train = dataloader.instance_idx_train
        #self.instances_test = dataloader.instance_idx_test
        self.instances_Test = dataloader.instance_idx_Test
        self.data_dir = 'G:\Project\CNN-1DMEAE'
        #self.label_hat1400 = []
        #self.label_hat1350 = []
        #self.label_hat1300 = []

        #self.label = []

        if cfg.test_set == 'train':
            self.test_set = dataloader.data_test
        else:
            self.test_set = dataloader.data_Test



    def test(self):
        records = dict(loss=[],
                       rec_loss=[],
                       entropy_loss=[],
                       condi_loss=[],
                       rec_error=[],
                       cls_loss=[],
                       cls_acc=[],
                      )

        self.testloader = DataLoader(dataset=self.test_set,
                                     batch_size=1,
                                     shuffle=False,
                                     #collate_fn=self.collator,
                                     #num_workers=self.num_dataloaders,
                                     num_workers=0,
                                    )
        labels = []
        per_loss = []
        per_loss_new = []
        encoding = []
        encodingme = []
        memw = []
        count = 0
        for i, batch in tqdm(enumerate(self.testloader), total=len(self.testloader), desc='Test'):
            count +=1
            self.model.eval()

            batch = [b.to(self.device) for b in batch]
            imgs, label, instances = batch[0],batch[1],self.instances_Test
            batch_size = imgs.size(0)
            print(batch_size)

            with torch.no_grad():
                result = self.model(imgs)

            rec_imgs = result['rec_x']
            cls_logit = result['logit_x']
            mem_weight = result['mem_weight']
            mem  = result['mem']
            z = result['encode']
            #z_hat = result['encodeme']
            encoding.append(z.cpu().numpy())
            #encodingme.append(z_hat.cpu().numpy())
            memw.append(mem_weight)


            if self.condi_loss_coef > 0.0:
                onehot_c = torch.FloatTensor(batch_size, self.num_memories).to(self.device)
                onehot_c.zero_()
                onehot_c.scatter_(1, instances.unsqueeze(1), 1.0)
                condi_loss = self.condi_criterion(mem_weight, onehot_c)
                condi_loss /= batch_size
                condi_loss *= self.condi_loss_coef
            else:
                condi_loss = torch.zeros(1).to(self.device)

            if self.addressing == 'sparse':
                mask = (mem_weight == 0).float()
                maksed_weight = mem_weight + mask
                entropy_loss = -mem_weight * torch.log(maksed_weight)
                entropy_loss = entropy_loss.sum() / batch_size
                entropy_loss *= self.entropy_loss_coef
            else:
                entropy_loss = torch.zeros(1).to(self.device)

            rec_loss = self.rec_criterion(rec_imgs, imgs)
            #rec_loss /= batch_size
            rec_error = (rec_imgs - imgs).pow(2).sum(1).sum(1).sum(1)
            #rec_error = (rec_imgs - imgs).pow(2).sum(1).sum(1)
            labels.append(label.item())
            per_loss.append(rec_loss.item())
            per_loss_new.append(rec_loss.item())



            if self.cls_loss_coef > 0.0:
                cls_loss = self.cls_criterion(cls_logit, instances)
                cls_loss *= self.cls_loss_coef
            else:
                cls_loss = torch.zeros(1).to(self.device)
            cls_pred = cls_logit.max(1)[1]
            cls_acc =(cls_logit.max(1)[1] == instances).float()

            loss = rec_loss + cls_loss + entropy_loss + condi_loss
            #self.writer.add_scalar("Valid_loss", loss, self.step)

            records['loss'] += [loss.cpu().item()]
            records['rec_loss'] += [rec_loss.cpu().item()]
            records['entropy_loss'] += [entropy_loss.cpu().item()]
            records['condi_loss'] += [condi_loss.cpu().item()]
            records['rec_error'] += rec_error.cpu().tolist()
            records['cls_loss'] += [cls_loss.cpu().item()]
            records['cls_acc'] += cls_acc.cpu().tolist()

        acc_list = []
        f1_list = []
        recall_list = []
        pre_list = []
        fpr_list = []



        #print(per_loss)
        #per_loss_new = per_loss
        per_loss.sort()



        #print(per_loss)
        for t in per_loss:
            #print(t)
            label_hat = []

            TP1 = 0
            TN1 = 0
            FP1 = 0
            FN1 = 0
            for i in range(len(per_loss)):
                #print("new:",per_loss_new[i])
                if per_loss_new[i] <= t:
                    (label_hat).append(1)
                else:
                    (label_hat).append(0)
            #print(label_hat)
            label_hat = np.array(label_hat)
            labels = np.array(labels)
            length = len(label_hat)
            for j in range(length):
                # if label_hat[j] == self.label[j]:
                #   count = count + 1
                if int(label_hat[j]) == 0 and int(labels[j]) == 0:
                    TP1 = TP1 + 1
                if int(label_hat[j]) == 1 and int(labels[j]) == 0:
                    FN1 = FN1 + 1
                if int(label_hat[j]) == 1 and int(labels[j]) == 1:
                    TN1 = TN1 + 1
                if int(label_hat[j]) == 0 and int(labels[j]) == 1:
                    FP1 = FP1 + 1

            acc1 = (TP1 + TN1) / (TP1 + TN1 + FP1 + FN1+1e-10)
            per1 = TP1 / (TP1 + FP1 + 1e-10)
            recall1 = TP1 / (TP1 + FN1 + 1e-10)
            F1 = 2*per1*recall1/(per1+recall1+1e-10)
            FPR = FP1 / (FP1 + TN1 + 1e-10)


            #print(per1)
            #print(recall1)
            recall_list.append(recall1)
            pre_list.append(per1)
            acc_list.append(acc1)
            f1_list.append(F1)
            fpr_list.append(FPR)

        recall_list = np.array(recall_list)
        pre_list = np.array(pre_list)
        f1_list = np.array(f1_list)
        acc_list = np.array(acc_list)
        max_f1 = max(f1_list)
        max_fi_index = np.where(f1_list == max_f1 )
        max_acc =max(acc_list)
        max_acc_index = np.where(acc_list == max_acc)

        pl.ylabel("p")
        pl.xlabel("r")
        pl.title("ROC")
        print(TP1+FP1+TN1+FN1)

        pl.plot(fpr_list,recall_list )
        pl.show()
        auc = -np.trapz(recall_list,fpr_list)
        print('auc=',auc)
        for k, v in records.items():
            records[k] = sum(records[k]) / len(records[k])

        print(per_loss_new)
        per_loss_new = np.array(per_loss_new)
        #print(per_loss.shape)
        #per_loss = np.expand_dims(per_loss, axis=1)
        per_loss_new = np.expand_dims(per_loss_new, axis=1)

        label_hat = []

        TP1 = 0
        TN1 = 0
        FP1 = 0
        FN1 = 0
        for i in range(len(per_loss)):
            # print("new:",per_loss_new[i])
            #if per_loss_new[i] <= 0.16114:
            if per_loss_new[i] <=5.677:
                (label_hat).append(1)
            else:
                (label_hat).append(0)
        # print(label_hat)
        label_hat = np.array(label_hat)
        labels = np.array(labels)
        length = len(label_hat)
        for j in range(length):
            # if label_hat[j] == self.label[j]:
            #   count = count + 1
            if int(label_hat[j]) == 0 and int(labels[j]) == 0:
                TP1 = TP1 + 1
            if int(label_hat[j]) == 1 and int(labels[j]) == 0:
                FN1 = FN1 + 1
            if int(label_hat[j]) == 1 and int(labels[j]) == 1:
                TN1 = TN1 + 1
            if int(label_hat[j]) == 0 and int(labels[j]) == 1:
                FP1 = FP1 + 1

        acc1 = (TP1 + TN1) / (TP1 + TN1 + FP1 + FN1 + 1e-10)
        per1 = TP1 / (TP1 + FP1 + 1e-10)
        recall1 = TP1 / (TP1 + FN1 + 1e-10)
        F1 = 2 * per1 * recall1 / (per1 + recall1 + 1e-10)
        FPR = FP1 / (FP1 + TN1 + 1e-10)
        print("TP=",TP1)
        print("TN=", TN1)
        print("FP=", FP1)
        print("FN=", FN1)
        print("acc=",acc1)
        print("precision=", per1)
        print("recall=", recall1)
        print("F1=", F1)
        print("FPR=",FPR)


        loss = records['loss']
        rec_loss = records['rec_loss']
        rec_error = records['rec_error']
        entropy_loss = records['entropy_loss']
        condi_loss = records['condi_loss']
        cls_loss = records['cls_loss']
        cls_acc = records['cls_acc']


        print('='*100)
        print('Test')
        print('Reconst error: {rec_error:.4f}'.format(rec_error=rec_error, end=' '))
        print('Loss: {loss:.4f}, Reconst loss: {rec_loss:.4f}' \
                  .format(loss=loss, rec_loss=rec_loss, end=' '))
        print('Entropy loss: {entropy_loss:8f}' \
                  .format(entropy_loss=entropy_loss, end=' '))
        print('Condition loss: {condi_loss:4f}' \
                  .format(condi_loss=condi_loss, end=' '))
        print('Cls_loss: {cls_loss:.4f}, Cls acc: {cls_acc:.4f}' \
                  .format(cls_loss=cls_loss, cls_acc=cls_acc, end=' '))

        #per_loss_new = 1 - per_loss_new
        #print(per_loss_new)
        encoding=np.array(encoding)
        encodingme = np.array(encodingme)
        #np.savez(os.path.join('anomaly_20memnoskip_trainlong.npz'), encoding=encoding, mem=mem)
        #np.savez(os.path.join('anomaly_100memskip.npz'), encoding=encoding, mem=mem)
        """
        q1, q3 = np.pecentile(per_loss_new, (25, 75), interpolation='midpoint')
        IQR = q3 - q1
        threshold2 = q3 + 1.5 * IQR
        print(threshold2)

        """
        #np.savez(os.path.join('loss-test-e1t027.npz'),loss=per_loss_new)
        #self.writer.close()

