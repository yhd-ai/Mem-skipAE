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
        self.device = device  # 运行的设备cpu/gpu

        self.cls_loss_coef = cfg.cls_loss_coef  # default=0 作用？
        self.sensor_channel_size = cfg.sensor_channel_size
        self.sensor_length = cfg.sensor_length
        self.num_instances = cfg.num_instances
        self.num_classes = cfg.num_classes
        self.num_memories = cfg.num_memories
        self.batch = cfg.batch_size
        # self.sensor_length = cfg.sensor_length
        # elf.image_height = cfg.image_height
        # self.image_width = cfg.image_width
        self.image_channel_size = cfg.image_channel_size

        self.addressing = cfg.addressing
        self.conv_channel_size = cfg.conv_channel_size

        self.feature_size = int(7872 / 8)  # 不知道为什么要乘以4*4*4
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
        init_mem = torch.zeros(self.num_memories, self.feature_size)  # 记忆单元的权重矩阵清零
        nn.init.kaiming_uniform_(init_mem)
        nn.init.sparse_(init_mem, 0.2)  # 初始化权重参数

        self.memory = nn.Parameter(init_mem)  # 将不可训练的tensor 转化成可以训练的parameter

        self.cosine_similarity = nn.CosineSimilarity(dim=2, )  # 第二个维度余弦相似度
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

        # nonlocal z_hat
        ex_mem = self.memory.unsqueeze(0).repeat(batch, 1,
                                                 1)  # 给mem增加第一个维度，且第一个维度的size为batch（batch，num_memories,feature_size）
        ex_z = z.unsqueeze(1).repeat(1, self.num_memories,
                                     1)  # 给mem增加第二个维度，第二个维度的size为记忆单元数量(batch，num_memories,feature_size）

        mem_logit = self.cosine_similarity(ex_z, ex_mem)
        # print(mem_logit.shape)#计算记忆单元和encoder后第二个维度的余弦相似度

        mem_weight = F.softmax(mem_logit, dim=1)
        # print(mem_weight)#对张量的第二个维度进行softmax分类
        # print(self.memory.shape)
        if self.addressing == 'soft':  # soft类
            z_hat = torch.matmul(mem_weight, self.memory)  # 给decoder的embedding是记忆权重矩阵和记忆矩阵的乘积
        elif self.addressing == 'sparse':  # 稀疏类
            mem_weight = (self.relu(mem_weight - self.threshold) * mem_weight) \
                         / (torch.abs(mem_weight - self.threshold) + self.epsilon)
            mem_weight = mem_weight / mem_weight.norm(p=1, dim=1) \
                .unsqueeze(1).expand(batch, self.num_memories)

            z_hat = torch.mm(mem_weight, self.memory)

        """_, a = torch.topk(mem_weight, 1, dim=1)
        # print(mem_weight.shape)

        # print(a.shape)
        _, gathering_indices = torch.topk(mem_weight, 2, dim=1)
        # print(gathering_indices)

        loss_mse = nn.MSELoss(reduction='sum')
        loss_tri = torch.nn.TripletMarginLoss(margin=1.0, reduction='sum')
        sum_compact_loss = 0
        sum_spread_loss = 0
        for i in range(batch):
            # print(i)
            compact_loss = loss_mse(z[i], self.memory[a[i, 0]])
            # print(z.shape)
            # print(self.memory[a[i,0]].shape)


            sum_compact_loss = sum_compact_loss + compact_loss

            pos = self.memory[gathering_indices[i, 0]]

            neg = self.memory[gathering_indices[i, 1]]
            # print(pos.shape)
            # print(neg.shape)
            # print(z[i].shape)


            # print(self.memory[a[i,0]].eq(pos))
            spreading_loss = loss_tri(z[i].unsqueeze(0), pos.unsqueeze(0), neg.unsqueeze(0))

            sum_spread_loss = sum_spread_loss + spreading_loss

        # print(sum_spread_loss)
        # print(sum_compact_loss)
        """



        #mem_weight = 0
        sum_compact_loss = 0
        sum_spread_loss = 0

        rec_x = self.decoder(z)  # rec_x是重构的原图
        # rec_x = self.defusion(rec_x1)
        # if self.cls_loss_coef > 0.0:
        #    logit_x = self.classifier(rec_x)
        # else:
        logit_x = torch.zeros(batch, self.num_classes).to(self.device)
        # mem_weight=0
        return dict(rec_x=rec_x, logit_x=logit_x, mem_weight=mem_weight, mem=self.memory.cpu().detach().numpy(),
                    encode=z, compact_loss=sum_compact_loss, spreading_loss=sum_spread_loss)
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


"""class Fusion(nn.Module):
    def __init__(self,sensor_channel_size,device):
        super(Fusion, self).__init__()
        self.sensor_channel_size = sensor_channel_size
        self.conv1 = nn.Conv2d(in_channels = 1,
                               out_channels = 1,
                               kernel_size = (self.sensor_channel_size,1),
                               stride=1,
                               padding=0,
                                )
        self.rule = nn.LeakyReLU(inplace=True)
        self.device = device
    def forward(self,x):
        x = x.type(torch.DoubleTensor).cuda(device=self.device)
        x = self.conv1(x)
        x = self.rule(x)
        return x
"""

"""class deFusion(nn.Module):
    def __init__(self,sensor_channel_size):
        super(deFusion, self).__init__()
        self.sensor_channel_size = sensor_channel_size
        self.conv_1 = nn.ConvTranspose2d(in_channels = 1,
                                         out_channels = 1,
                                         kernel_size = (self.sensor_channel_size,1),
                                         stride=1,
                                         padding=0,

        )
        self.rule = nn.Sigmoid()
        #self.rule = nn.LeakyReLU()
    def forward(self,x):
        x = x.unsqueeze(1)
        x = self.conv_1(x)
        x = self.rule(x)
        #x = print(x.shape)
        #x = x[:,:,:,:3000]
        #print(x.shape)
        return x
"""


class Encoder(nn.Module):
    def __init__(self, image_channel_size, conv_channel_size):
        super(Encoder, self).__init__()
        self.image_channel_size = image_channel_size
        self.conv_channel_size = conv_channel_size

        self.conv1 = nn.Conv2d(in_channels=self.image_channel_size,  # batch , 4,402
                               out_channels=self.conv_channel_size,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               )

        self.bn1 = nn.BatchNorm2d(num_features=self.conv_channel_size, )  # batch , 4 ,402

        self.conv2 = nn.Conv2d(in_channels=self.conv_channel_size,  # batch , 8 ,101
                               out_channels=self.conv_channel_size * 8,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               )

        self.bn2 = nn.BatchNorm2d(num_features=self.conv_channel_size * 8, )

        self.conv3 = nn.Conv2d(in_channels=self.conv_channel_size * 8,  # batch, 16, 50
                               out_channels=self.conv_channel_size * 8,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               )

        self.bn3 = nn.BatchNorm2d(num_features=self.conv_channel_size * 8, )
        self.conv4 = nn.Conv2d(in_channels=self.conv_channel_size * 8,  # batch, 16, 50
                               out_channels=self.conv_channel_size,
                               kernel_size=3,
                               stride=1,
                               padding=0,
                               )

        self.bn4 = nn.BatchNorm2d(num_features=self.conv_channel_size, )

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

        self.deconv1 = nn.ConvTranspose2d(in_channels=self.conv_channel_size,
                                          out_channels=self.conv_channel_size * 8,  # 3
                                          kernel_size=3,
                                          stride=1,
                                          padding=0,
                                          # output_padding=1,
                                          )

        self.bn1 = nn.BatchNorm2d(num_features=self.conv_channel_size * 8, )

        self.deconv2 = nn.ConvTranspose2d(in_channels=self.conv_channel_size * 8,
                                          out_channels=self.conv_channel_size * 8,  # 5
                                          kernel_size=(2, 3),
                                          stride=2,
                                          padding=1,
                                          output_padding=1,
                                          )

        self.bn2 = nn.BatchNorm2d(num_features=self.conv_channel_size * 8, )

        self.deconv3 = nn.ConvTranspose2d(in_channels=self.conv_channel_size * 8,
                                          out_channels=self.conv_channel_size,  # 9
                                          kernel_size=(2, 3),
                                          stride=2,
                                          padding=1,
                                          output_padding=1,
                                          )
        self.bn3 = nn.BatchNorm2d(num_features=self.conv_channel_size, )
        self.deconv4 = nn.ConvTranspose2d(in_channels=self.conv_channel_size,  #
                                          out_channels=self.image_channel_size,
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          output_padding=1,
                                          )
        self.bn4 = nn.BatchNorm2d(num_features=self.image_channel_size, )

        self.relu = nn.LeakyReLU(inplace=True)
        self.tan = nn.Tanh()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 8, 1, 123)

        x = self.deconv1(x )
        x = self.bn1(x)
        x = self.relu(x)

        x = self.deconv2(x )
        x = self.bn2(x)
        x = self.relu(x)

        x = self.deconv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.deconv4(x )
        # x = self.bn4(x)
        # x = self.relu(x)
        x = self.sig(x)
        # x = self.tan(x)
        # print(x.shape)
        return x


"""class Classifier(nn.Module):
    def __init__(self, image_channel_size, conv_channel_size, num_classes, drop_rate):
        super(Classifier, self).__init__()
        self.image_channel_size = image_channel_size
        self.conv_channel_size = conv_channel_size
        self.num_classes = num_classes
        self.drop_rate = drop_rate

        self.conv1 = nn.Conv1d(in_channels=self.image_channel_size,
                               out_channels=6,
                               kernel_size=5,
                              )

        self.conv2 = nn.Conv1d(in_channels=6,
                               out_channels=16,
                               kernel_size=5,
                              )

        self.fc1 = nn.Linear(in_features=16*4*4, out_features=256,)
        self.fc2 = nn.Linear(in_features=256, out_features=128,)
        self.fc3 = nn.Linear(in_features=128, out_features=self.num_classes,)

        self.maxpool2d = nn.MaxPool1d(kernel_size=2,)
        self.relu = nn.ReLU(inplace=True)
        # self.dropout = nn.Dropout(p=self.drop_rate)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool2d(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.maxpool2d(x)
        x = self.relu(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu(x)
        # x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        # x = self.dropout(x)

        x = self.fc3(x)
        return x"""""
