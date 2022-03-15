import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def weights_inint(mod):
    # classname = mod.__class__.__name__
    nn.init.kaiming_uniform_(mod.weight.data)


class CNNMEAE(nn.Module):
    def __init__(self, cfg, device):
        super(CNNMEAE, self).__init__()
        self.device = device

        self.cls_loss_coef = cfg.cls_loss_coef
        self.sensor_channel_size = cfg.sensor_channel_size
        self.sensor_length = cfg.sensor_length
        self.num_instances = cfg.num_instances
        self.num_classes = cfg.num_classes
        self.num_memories = cfg.num_memories

        # self.sensor_length = cfg.sensor_length
        # elf.image_height = cfg.image_height
        # self.image_width = cfg.image_width
        self.image_channel_size = cfg.image_channel_size

        self.addressing = cfg.addressing
        self.conv_channel_size = cfg.conv_channel_size

        self.feature_size = int(7872 / 8)
        self.drop_rate = cfg.drop_rate

        self.encoder = Encoder(image_channel_size=self.image_channel_size,
                               conv_channel_size=self.conv_channel_size,
                               )

        """data = np.load('normal_nomem.npz')
        self.memory = data['encoding']
        self.memory = self.memory.squeeze(1)
        self.memory = torch.from_numpy(self.memory)
        self.memory = nn.Parameter(self.memory)
        print(self.memory.shape)


        """
        init_mem = torch.zeros(self.num_memories, self.feature_size)
        nn.init.kaiming_uniform_(init_mem)
        nn.init.sparse_(init_mem, 0.2)

        self.memory = nn.Parameter(init_mem)

        self.cosine_similarity = nn.CosineSimilarity(dim=2, )
        #        self.defusion = deFusion(self.sensor_channel_size)
        self.decoder = Decoder(sensor_channel_size=self.sensor_channel_size,
                               sensor_length=self.sensor_length,
                               image_channel_size=self.image_channel_size,
                               conv_channel_size=self.conv_channel_size,
                               )

        self.relu = nn.ReLU(inplace=True)

        # if self.cls_loss_coef > 0.0:
        #     self.classifier = Classifier(image_channel_size=self.image_channel_size,
        #                                  conv_channel_size=self.conv_channel_size,
        #                                  num_classes=self.num_classes,
        #                                  drop_rate=self.drop_rate,
        #                                 )

        if self.addressing == 'sparse':
            self.threshold = 1 / self.memory.size(0)  #
            self.epsilon = 1e-15

    def forward(self, x):
        batch, channel, height, width = x.size()  # batch ,1 ,18, 3000
        # z1 = self.fusion(x)                       #batich,1, 1, 3000
        z = self.encoder(x)

        #nonlocal z_hat
        ex_mem = self.memory.unsqueeze(0).repeat(batch, 1, 1)
        ex_z = z.unsqueeze(1).repeat(1, self.num_memories, 1)
        mem_logit = self.cosine_similarity(ex_z, ex_mem)


        mem_weight = F.softmax(mem_logit, dim=1)

        if self.addressing == 'soft':
            z_hat = torch.matmul(mem_weight, self.memory)
        elif self.addressing == 'sparse':
            mem_weight = (self.relu(mem_weight - self.threshold) * mem_weight) \
                            / (torch.abs(mem_weight - self.threshold) + self.epsilon)
            mem_weight = mem_weight / mem_weight.norm(p=1, dim=1) \
                            .unsqueeze(1).expand(batch, self.num_memories)

            z_hat = torch.mm(mem_weight, self.memory)



        #mem_weight = 0

        _, a = torch.topk(mem_weight, 1,dim=1)

        print(a.shape)
        _, gathering_indices = torch.topk(mem_weight, 2,dim=1)
        print(gathering_indices.shape)


        loss_mse = nn.MSELoss(reduction='sum')

        for i in range(mem_weight.shape(0)):
            compact_loss = loss_mse(z[i],self.memory[i][a[i]])
            sum_compact_loss  = sum_compact_loss + compact_loss

            pos = self.memory[i][gathering_indices[:, 0]]
            neg = self.memory[i][gathering_indices[:, 1]]




        rec_x = self.decoder(z_hat)
        # rec_x = self.defusion(rec_x1)
        # if self.cls_loss_coef > 0.0:
        #    logit_x = self.classifier(rec_x)
        # else:
        logit_x = torch.zeros(batch, self.num_classes).to(self.device)
        # mem_weight=0
        return dict(rec_x=rec_x, logit_x=logit_x, mem_weight=mem_weight, mem=self.memory.cpu().detach().numpy(),
                    encode=z)
        # return dict(rec_x=rec_x, logit_x=logit_x)

    def generate_from_memory(self, idx):
        z_hat = self.memory[idx]
        batch, _ = z_hat.size()

        rec_x = self.decoder(z_hat)

        # if self.cls_loss_coef > 0.0:
        #    logit_x = self.classifier(rec_x)
        # else:
        logit_x = torch.zeros(batch, self.num_classes).to(self.device)
        return dict(rec_x=rec_x, logit_x=logit_x)


class Encoder(nn.Module):
    def __init__(self, image_channel_size, conv_channel_size):
        super(Encoder, self).__init__()
        self.image_channel_size = image_channel_size
        self.conv_channel_size = conv_channel_size

        self.conv1 = nn.Conv1d(in_channels=self.image_channel_size,  # batch , 4,402
                               out_channels=self.conv_channel_size,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               )

        self.bn1 = nn.BatchNorm1d(num_features=self.conv_channel_size, )  # batch , 4 ,402

        self.conv2 = nn.Conv1d(in_channels=self.conv_channel_size,  # batch , 8 ,101
                               out_channels=self.conv_channel_size * 8,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               )

        self.bn2 = nn.BatchNorm1d(num_features=self.conv_channel_size * 8, )

        self.conv3 = nn.Conv1d(in_channels=self.conv_channel_size * 8,  # batch, 16, 50
                               out_channels=self.conv_channel_size * 8,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               )

        self.bn3 = nn.BatchNorm1d(num_features=self.conv_channel_size * 8, )
        self.conv4 = nn.Conv1d(in_channels=self.conv_channel_size * 8,  # batch, 16, 50
                               out_channels=self.conv_channel_size,
                               kernel_size=3,
                               stride=1,
                               padding=0,
                               )

        self.bn4 = nn.BatchNorm1d(num_features=self.conv_channel_size, )

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        # x = x.squeeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # print(x.shape)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        # print(x.shape)
        batch, _, _, _ = x.size()
        x = x.view(batch, -1)
        return x


class Decoder(nn.Module):
    def __init__(self, sensor_channel_size, sensor_length, image_channel_size, conv_channel_size):
        super(Decoder, self).__init__()
        self.sensor_channel_size = sensor_channel_size
        self.sensor_length = sensor_length
        self.image_channel_size = image_channel_size
        self.conv_channel_size = conv_channel_size

        self.deconv1 = nn.ConvTranspose1d(in_channels=self.conv_channel_size,
                                          out_channels=self.conv_channel_size * 8,  # 3
                                          kernel_size=3,
                                          stride=1,
                                          padding=0,
                                          # output_padding=1,
                                          )

        self.bn1 = nn.BatchNorm1d(num_features=self.conv_channel_size * 8, )

        self.deconv2 = nn.ConvTranspose1d(in_channels=self.conv_channel_size * 8,
                                          out_channels=self.conv_channel_size * 8,  # 5
                                          kernel_size=(2, 3),
                                          stride=2,
                                          padding=1,
                                          output_padding=1,
                                          )

        self.bn2 = nn.BatchNorm1d(num_features=self.conv_channel_size * 8, )

        self.deconv3 = nn.ConvTranspose1d(in_channels=self.conv_channel_size * 8,
                                          out_channels=self.conv_channel_size,  # 9
                                          kernel_size=(2, 3),
                                          stride=2,
                                          padding=1,
                                          output_padding=1,
                                          )
        self.bn3 = nn.BatchNorm1d(num_features=self.conv_channel_size, )
        self.deconv4 = nn.ConvTranspose1d(in_channels=self.conv_channel_size,  #
                                          out_channels=self.image_channel_size,
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          output_padding=1,
                                          )
        self.bn4 = nn.BatchNorm1d(num_features=self.image_channel_size, )

        self.relu = nn.LeakyReLU(inplace=True)
        self.tan = nn.Tanh()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 8, 1, 123)

        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.deconv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.deconv4(x)
        # x = self.bn4(x)
        # x = self.relu(x)
        x = self.sig(x)
        # x = self.tan(x)
        # print(x.shape)
        return x

