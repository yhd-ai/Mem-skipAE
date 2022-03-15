import json
import os
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch
#from torch import CNNMEAE
from numpy import *
from torch.utils.data import DataLoader
from preprocess import trainprepro
#from tensorboard_logger import Logger


#logger = Logger(logdir="./tensorboard_logs", flush_secs=10)
#from dataset import BatchCollator
#from util import UnNormalize

#torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_tensor_type(torch.DoubleTensor)
class Trainer():
    def __init__(self, cfg, dataloader, model, optimizer, device, step=0):
        self.cfg = cfg
        self.sensor_length = cfg.sensor_length
        self.sensor_channel_size = cfg.sensor_channel_size

        self.image_channel_size = cfg.image_channel_size


        self.log_dir = cfg.log_dir
        self.writer = SummaryWriter(self.log_dir+'/logs')
        self.num_dataloaders = cfg.num_dataloaders
        self.device = device

        self.train_writer = SummaryWriter(logdir=os.path.join(cfg.log_dir, 'train'))
        self.valid_writer = SummaryWriter(logdir=os.path.join(cfg.log_dir, 'valid'))

        self.train_writer.add_text('cfg', json.dumps(cfg.__dict__))
        self.valid_writer.add_text('cfg', json.dumps(cfg.__dict__))

        self.batch_size = cfg.batch_size
        self.num_epochs = cfg.num_epochs

        self.model = model
        self.optimizer = optimizer

        self.cls_loss_coef = cfg.cls_loss_coef
        self.entropy_loss_coef = cfg.entropy_loss_coef
        self.condi_loss_coef = cfg.condi_loss_coef
        self.addressing = cfg.addressing
        self.num_memories = cfg.num_memories

        self.cls_criterion = nn.CrossEntropyLoss(reduction='mean')
        self.rec_criterion = nn.MSELoss(reduction='sum')
        self.condi_criterion = nn.BCELoss(reduction='sum')
        self.instances_train = dataloader.instance_idx_train
        self.instances_test = dataloader.instance_idx_test


        self.train_set = dataloader.data_train
        if cfg.test_set == 'train':
            self.test_set = dataloader.data_test
        else:
            self.test_set = dataloader.data_test

        #self.collator = BatchCollator(image_height=self.image_height,
        #                              image_width=self.image_width,
        #                              image_channel_size=self.image_channel_size,
        #                             )

        self.step = step
        self.best_cls_acc = 0.0
        self.best_rec_error = 1000000.0

    def valid(self):
        records = dict(loss=[],
                       rec_loss=[],
                       entropy_loss=[],
                       condi_loss=[],
                       rec_error=[],
                       cls_loss=[],
                       cls_acc=[],
                       compact_loss = [],
                       spreading_loss=[],
                      )

        self.testloader = DataLoader(dataset=self.test_set,
                                     batch_size=self.batch_size,
                                     shuffle=False,
                                     #collate_fn=self.collator,
                                     #num_workers=self.num_dataloaders,
                                     num_workers=0,
                                    )
        #for i, batch in tqdm(enumerate(self.testloader), total=len(self.testloader), desc='Valid'):#tqdm：进度条 enumerate（）：同时列出数据和数据索引
        for i, batch in tqdm(enumerate(self.testloader)):
            self.model.eval()#评估模式：不更新权重

            batch = [b.to(self.device) for b in batch]

            #imgs, labels, instances = batch[0], batch[1], batch[2]
            #rint(np.array(batch).shape)
            imgs, instances = batch[0], self.instances_test
            batch_size = imgs.size(0)

            with torch.no_grad():
                result = self.model(imgs)

            rec_imgs = result['rec_x']
            cls_logit = result['logit_x']
            mem_weight = result['mem_weight']
            compact_loss = result['compact_loss']
            spreading_loss = result['spreading_loss']

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
            rec_loss /= batch_size
            #rec_error = (rec_imgs - imgs).pow(2).sum(1).sum(1).sum(1)
            rec_error = (rec_imgs - imgs).pow(2).sum(1).sum(1)
            compact_loss /= batch_size
            spreading_loss /= batch_size


            if self.cls_loss_coef > 0.0:
                cls_loss = self.cls_criterion(cls_logit, instances)
                cls_loss *= self.cls_loss_coef
            else:
                cls_loss = torch.zeros(1).to(self.device)
            cls_pred = cls_logit.max(1)[1]
            cls_acc = (cls_logit.max(1)[1] == instances).float()


            loss = rec_loss + cls_loss + entropy_loss + condi_loss + 0.1*compact_loss + 0.1*spreading_loss
            self.writer.add_scalar("Valid_loss", loss,self.step )
            #print(loss)
            #print("loss=", loss)
            records['loss'] += [loss.cpu().item()]
            records['rec_loss'] += [rec_loss.cpu().item()]
            records['entropy_loss'] += [entropy_loss.cpu().item()]
            records['condi_loss'] += [condi_loss.cpu().item()]
            records['rec_error'] += rec_error.cpu().tolist()
            records['cls_loss'] += [cls_loss.cpu().item()]
            records['cls_acc'] += cls_acc.cpu().tolist()
            records['compact_loss'] += [compact_loss.cpu().item()]
            records['spreading_loss'] += [spreading_loss.cpu().item()]

        for k, v in records.items():
            records[k] = sum(records[k]) / len(records[k])

        loss = records['loss']
        rec_loss = records['rec_loss']
        rec_error = records['rec_error']
        entropy_loss = records['entropy_loss']
        condi_loss = records['condi_loss']
        cls_loss = records['cls_loss']
        cls_acc = records['cls_acc']
        c_loss = records['compact_loss']
        s_loss = records['spreading_loss']

        self._save_checkpoint(rec_error, cls_acc)
        print('='*100)
        print('Valid')
        print('Reconst error: {rec_error:.4f}'.format(rec_error=rec_error, end=' '))
        print('Loss: {loss:.4f}, Reconst loss: {rec_loss:.4f}' \
                  .format(loss=loss, rec_loss=rec_loss, end=' '))
        print('Entropy loss: {entropy_loss:8f}' \
                  .format(entropy_loss=entropy_loss, end=' '))
        print('Condition loss: {condi_loss:4f}' \
                  .format(condi_loss=condi_loss, end=' '))
        print('Cls_loss: {cls_loss:.4f}, Cls acc: {cls_acc:.4f}' \
                  .format(cls_loss=cls_loss, cls_acc=cls_acc, end=' '))
        print('Compact_loss: {c_loss:.4f}, Cls acc: {cls_acc:.4f}' \
              .format(c_loss=c_loss, cls_acc=cls_acc, end=' '))
        print('spreading_loss: {s_loss:.4f}, Cls acc: {cls_acc:.4f}' \
              .format(s_loss=s_loss, cls_acc=cls_acc, end=' '))
        print()
        self.valid_writer.add_scalar('01._Reconst_error', rec_error, self.step)
        self.valid_writer.add_scalar('02._Loss', loss, self.step)
        self.valid_writer.add_scalar('03._Reconst_loss', rec_loss, self.step)
        self.valid_writer.add_scalar('04._Entropy_loss', entropy_loss, self.step)
        self.valid_writer.add_scalar('05._Condition_loss', condi_loss, self.step)
        self.valid_writer.add_scalar('06._Cls_loss', cls_loss, self.step)
        self.valid_writer.add_scalar('07._Cls_acc', cls_acc, self.step)
        self.valid_writer.add_scalar('08._Com_loss', c_loss, self.step)
        self.valid_writer.add_scalar('09._spreading_loss', s_loss, self.step)

    def train(self):
        self.valid()
        for epoch in tqdm(range(self.num_epochs), desc='Train'):
            records = dict(loss=[],
                           rec_loss=[],
                           entropy_loss=[],
                           condi_loss=[],
                           rec_error=[],
                           cls_loss=[],
                           cls_acc=[],
                           compact_loss=[],
                           spreading_loss=[],
                          )

            self.trainloader = DataLoader(dataset=self.train_set,
                                          batch_size=self.batch_size,
                                          shuffle=True,
                                          #collate_fn=self.collator,
                                          num_workers=0,
                                          #num_workers=self.num_dataloaders,
                                         )
            #for i, batch in tqdm(enumerate(self.trainloader), total=len(self.trainloader), desc='Epoch %d' % epoch):
            for i, batch in enumerate(self.trainloader):
                self.model.train()#训练模式：更新权重
                self.optimizer.zero_grad()

                batch = [b.to(self.device) for b in batch]

                #imgs, labels, instances = batch[0], batch[1], batch[2]
                imgs,instances  = batch[0],self.instances_train
                batch_size = imgs.size(0)

                result = self.model(imgs)

                rec_imgs = result['rec_x']
                cls_logit = result['logit_x']
                mem_weight = result['mem_weight']
                compact_loss = result['compact_loss']
                spreading_loss = result['spreading_loss']

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
                    masked_weight = mem_weight + mask
                    entropy_loss = -mem_weight * torch.log(masked_weight)
                    entropy_loss = entropy_loss.sum() / batch_size
                    entropy_loss *= self.entropy_loss_coef
                else:
                    entropy_loss = torch.zeros(1).to(self.device)

                rec_loss = self.rec_criterion(rec_imgs, imgs)
                rec_loss /= batch_size
                #rec_error = (rec_imgs - imgs).pow(2).sum(1).sum(1).sum(1)
                rec_error = (rec_imgs - imgs).pow(2).sum(1).sum(1)
                compact_loss /= batch_size
                spreading_loss /= batch_size

                if self.cls_loss_coef > 0.0:
                    cls_loss = self.cls_criterion(cls_logit, instances)
                    cls_loss *= self.cls_loss_coef
                else:
                    cls_loss = torch.zeros(1).to(self.device)
                cls_pred = cls_logit.max(1)[1]
                cls_acc = (cls_logit.max(1)[1] == instances).float()

                loss = rec_loss + cls_loss + entropy_loss + condi_loss + 0.1*compact_loss + 0.1*spreading_loss
                self.writer.add_scalar("train_loss", loss, self.step)
                #torch.set_default_tensor_type(torch.DoubleTensor)
                torch.set_default_tensor_type(torch.DoubleTensor)
                loss = loss.double()
                #print("loss=",loss)
                loss.backward()
                self.optimizer.step()

                self._update_tensorboard(loss=loss.item(),
                                         rec_loss=rec_loss.item(),
                                         entropy_loss=entropy_loss.item(),
                                         condi_loss=condi_loss.item(),
                                         rec_error=rec_error.sum().item() / batch_size,
                                         cls_loss=cls_loss.item(),
                                         cls_acc=(cls_acc.sum() / batch_size).item(),
                                         c_loss = compact_loss.item(),
                                         s_loss = spreading_loss.item(),
                                        )

                # self._print_progress(loss=loss.item(),
                #                      rec_loss=rec_loss.item(),
                #                      entropy_loss=entropy_loss.item(),
                #                      condi_loss=condi_loss.item(),
                #                      rec_error=rec_error.sum().item() / batch_size,
                #                      cls_loss=cls_loss.item(),
                #                      cls_acc=(cls_acc.sum() / batch_size).item(),
                #                     )

                records['loss'] += [loss.cpu().item()]
                records['rec_loss'] += [rec_loss.cpu().item()]
                records['entropy_loss'] += [entropy_loss.cpu().item()]
                records['condi_loss'] += [condi_loss.cpu().item()]
                records['rec_error'] += rec_error.cpu().tolist()
                records['cls_loss'] += [cls_loss.cpu().item()]
                records['cls_acc'] += cls_acc.cpu().tolist()
                records['compact_loss'] += [compact_loss.cpu().item()]
                records['spreading_loss'] += [spreading_loss.cpu().item()]


                self.step += 1

            for k, v in records.items():
                records[k] = sum(records[k]) / len(records[k])

            loss = records['loss']
            rec_loss = records['rec_loss']
            rec_error = records['rec_error']
            entropy_loss = records['entropy_loss']
            condi_loss = records['condi_loss']
            cls_loss = records['cls_loss']
            cls_acc = records['cls_acc']
            s_loss = records['spreading_loss']
            c_loss = records['compact_loss']

            print('='*100)
            print('Train')
            print('Reconst error: {rec_error:.4f}'.format(rec_error=rec_error, end=' '))
            print('Loss: {loss:.4f}, Reconst loss: {rec_loss:.4f}' \
                      .format(loss=loss, rec_loss=rec_loss, end=' '))
            print('Entropy loss: {entropy_loss:8f}' \
                      .format(entropy_loss=entropy_loss, end=' '))
            print('Condition loss: {condi_loss:4f}' \
                      .format(condi_loss=condi_loss, end=' '))
            print('Cls_loss: {cls_loss:.4f}, Cls acc: {cls_acc:.4f}' \
                      .format(cls_loss=cls_loss, cls_acc=cls_acc, end=' '))
            print('Compact_loss: {c_loss:.4f}, Cls acc: {cls_acc:.4f}' \
                  .format(c_loss=c_loss, cls_acc=cls_acc, end=' '))
            print('spreading_loss: {s_loss:.4f}, Cls acc: {cls_acc:.4f}' \
                  .format(s_loss=s_loss, cls_acc=cls_acc, end=' '))
            print()
            self.valid()
        self.writer.close()

    def _update_tensorboard(self, loss, rec_loss, entropy_loss, condi_loss, rec_error, cls_loss, cls_acc,c_loss,s_loss):
        self.train_writer.add_scalar('01._Reconst_error', rec_error, self.step)
        self.train_writer.add_scalar('02._Loss', loss, self.step)
        self.train_writer.add_scalar('03._Reconst_loss', rec_loss, self.step)
        self.train_writer.add_scalar('04._Entropy_loss', entropy_loss, self.step)
        self.train_writer.add_scalar('05._Condition_loss', condi_loss, self.step)
        self.train_writer.add_scalar('06._CLS_loss', cls_loss, self.step)
        self.train_writer.add_scalar('07._CLS_acc', cls_acc, self.step)
        self.train_writer.add_scalar('08._Com_loss', c_loss, self.step)
        self.train_writer.add_scalar('09._spreading_loss', s_loss, self.step)

    def _print_progress(self, loss, rec_loss, entropy_loss, condi_loss, rec_error, cls_loss, cls_acc):
        print('='*100)
        print('Reconst error: {rec_error:.4f}'.format(rec_error=rec_error, end=' '))
        print('Loss: {loss:.4f}, Reconst loss: {rec_loss:.4f}' \
                  .format(loss=loss, rec_loss=rec_loss, end=' '))
        print('Entropy loss: {entropy_loss:.8f}' \
                  .format(entropy_loss=entropy_loss, end=' '))
        print('Condition loss: {condi_loss:4f}' \
                  .format(condi_loss=condi_loss, end=' '))
        print('Cls_loss: {cls_loss:.4f}, Cls acc: {cls_acc:.4f}' \
                  .format(cls_loss=cls_loss, cls_acc=cls_acc, end=' '))
        print()

    def _save_checkpoint(self, rec_error, cls_acc):
        last_ckpt_path = os.path.join(self.log_dir, 'ckpt', 'model-last.ckpt')
        torch.save(dict(
            cfg=self.cfg,
            step=self.step,
            model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(),
        ), last_ckpt_path)



        if self.cls_loss_coef != 0.0:
            if cls_acc > self.best_cls_acc:
                best_ckpt_path = os.path.join(self.log_dir, 'ckpt', 'model-best.ckpt')
                torch.save(dict(
                    cfg=self.cfg,
                    step=self.step,
                    model=self.model.state_dict(),
                    optimizer=self.optimizer.state_dict(),
                ), best_ckpt_path)
                self.best_cls_acc = cls_acc
        else:
            if rec_error < self.best_rec_error:
                best_ckpt_path = os.path.join(self.log_dir, 'ckpt', 'model-best.ckpt')
                torch.save(dict(
                    cfg=self.cfg,
                    step=self.step,
                    model=self.model.state_dict(),
                    optimizer=self.optimizer.state_dict(),
                ), best_ckpt_path)
                self.best_rec_error = rec_error
