
from abc import ABC, abstractmethod
import os

import torch
import torch.distributions as dist

class Modality(ABC):
    # ABC是Python内置的抽象类
    def __init__(self, name, enc, dec, class_dim, style_dim, lhood_name):
        self.name = name    # 模态名
        self.encoder = enc  # Encoder网络, nn.Module类的派生类对象
        self.decoder = dec  # Decoder网络, nn.Module类的派生类对象
        self.class_dim = class_dim  # 输出维度, 三个模态共同隐藏factor(论文里Fig.4的z)的维度
        self.style_dim = style_dim  # 输入维度, 三个模态各自的输入维度
        self.likelihood_name = lhood_name  # 似然分布
        self.likelihood = self.get_likelihood(lhood_name)  # 一个torch.distributions里的似然分布对象


    def get_likelihood(self, name):
        # 根据输入的分布名, 获得PyTorch中的分布对象
        if name == 'laplace':
            pz = dist.Laplace
        elif name == 'bernoulli':
            pz = dist.Bernoulli
        elif name == 'normal':
            pz = dist.Normal
        elif name == 'categorical':
            pz = dist.OneHotCategorical
        else:
            print('likelihood not implemented')
            pz = None
        return pz





    def calc_log_prob(self, out_dist, target, norm_value):
        log_prob = out_dist.log_prob(target).sum()
        mean_val_logprob = log_prob/norm_value
        return mean_val_logprob



    def save_networks(self, dir_checkpoints):
        torch.save(self.encoder.state_dict(), os.path.join(dir_checkpoints,
                                                           'enc_' + self.name))
        torch.save(self.decoder.state_dict(), os.path.join(dir_checkpoints,
                                                           'dec_' + self.name))

