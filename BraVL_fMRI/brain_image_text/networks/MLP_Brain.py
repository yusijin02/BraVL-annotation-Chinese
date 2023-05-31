
import torch
import torch.nn as nn


class EncoderBrain(nn.Module):
    def __init__(self, flags):
        super(EncoderBrain, self).__init__()
        self.flags = flags
        self.hidden_dim = 512

        modules = []
        modules.append(nn.Sequential(nn.Linear(flags.m1_dim, self.hidden_dim), nn.ReLU(True)))
        # 隐藏层个数默认为2, flags.num_hidden_layers = 2
        modules.extend([nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(True))
                        for _ in range(flags.num_hidden_layers - 1)])
        self.enc = nn.Sequential(*modules)
        # enc (Encoder): X ---> (m1_dim, 512) ---> ReLU ---> (512, 512) --> ReLU ---> h
        self.relu = nn.ReLU()

        self.hidden_mu = nn.Linear(in_features=self.hidden_dim, out_features=flags.class_dim, bias=True)  # 计算均值
        # class_dim 默认为 32 (论文里的 z 的维度)
        # hidden_mu (均值): h ---> (512, 32) ---> mean
        self.hidden_logvar = nn.Linear(in_features=self.hidden_dim, out_features=flags.class_dim, bias=True)  # 计算对数方差
        # hidden_logvar (对数方差): h ---> (512, 32) ---> logvar


    def forward(self, x):
        # 前向计算
        h = self.enc(x)  # 计算隐藏层h
        h = h.view(h.size(0), -1)  # 将h变成(batch_size, hidden_dim)的形状
        latent_space_mu = self.hidden_mu(h)  # 计算均值
        latent_space_logvar = self.hidden_logvar(h)  # 计算对数方差
        # 将均值和方差都变成(batch_size, class_dim)的形状
        latent_space_mu = latent_space_mu.view(latent_space_mu.size(0), -1)
        latent_space_logvar = latent_space_logvar.view(latent_space_logvar.size(0), -1)

        # (batch_size, m1_dim)                                    (batch_size, 512)              (batch_size, 32)
        #  |                                                             |                               |
        #  |                                                             |                               |
        # x_b ---> (m1_dim, 512) ---> ReLU ---> (512, 512) --> ReLU ---> h ------> (512, 32) ---> latent_space_mu
        #                                                                |_______> (512, 32) ---> latent_space_logvar

        return None, None, latent_space_mu, latent_space_logvar



class DecoderBrain(nn.Module):
    def __init__(self, flags):
        super(DecoderBrain, self).__init__()
        self.flags = flags
        self.hidden_dim = 512
        modules = []

        modules.append(nn.Sequential(nn.Linear(flags.class_dim, self.hidden_dim), nn.ReLU(True)))

        modules.extend([nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(True))
                        for _ in range(flags.num_hidden_layers - 1)])
        self.dec = nn.Sequential(*modules)
        self.fc3 = nn.Linear(self.hidden_dim, flags.m1_dim)
        self.relu = nn.ReLU()


    def forward(self, style_latent_space, class_latent_space):
        z = class_latent_space
        x_hat = self.dec(z)
        x_hat = self.fc3(x_hat)

        # (batch_size, 32)                                    (batch_size, 512)         (batch_size, 32)
        #  |                                                        |                          |
        #  |                                                        |                          |
        #  z ---> (32, 512) ---> ReLU ---> (512, 512) --> ReLU ---> h ------> (512, 32) ---> x_hat

        return x_hat, torch.tensor(0.75).to(z.device)
