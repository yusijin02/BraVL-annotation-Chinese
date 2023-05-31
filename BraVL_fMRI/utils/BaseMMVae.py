from abc import ABC, abstractmethod

import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.distributions as dist
from divergence_measures.mm_div import calc_alphaJSD_modalities
from divergence_measures.mm_div import calc_group_divergence_moe
from divergence_measures.mm_div import poe

from utils import utils


class BaseMMVae(ABC, nn.Module):
    def __init__(self, flags, modalities, subsets):
        super(BaseMMVae, self).__init__()  # 调用父类nn.Module的构造函数
        self.num_modalities = len(modalities.keys())  # 模态数
        self.flags = flags  # 状态字典
        self.modalities = modalities  # 模态字典
        self.subsets = subsets  # 子集字典
        self.set_fusion_functions()

        encoders = nn.ModuleDict()
        decoders = nn.ModuleDict()
        lhoods = dict()
        for m, m_key in enumerate(sorted(modalities.keys())):
            encoders[m_key] = modalities[m_key].encoder
            decoders[m_key] = modalities[m_key].decoder
            lhoods[m_key] = modalities[m_key].likelihood
        self.encoders = encoders  # 编码器网络
        # self.encoders = {
        #     "brain": EncoderBrain对象,
        #     "image": EncoderImage对象,
        #     "text" : EncoderText 对象,
        # }
        self.decoders = decoders
        # self.encoders = {
        #     "brain": DecoderBrain对象,
        #     "image": DecoderImage对象,
        #     "text" : DecoderText 对象,
        # }
        self.lhoods = lhoods
        # self.encoders和self.decoders都是EncoderBrain, EncoderImage, EncoderText类对象之一
        # self.lhoods是torch.distributions里的似然分布对象


    def reparameterize(self, mu, logvar):
        # 生成一个均值为mu, 对数方差为logvar的分布
        # 这一步的目的是让采样的过程变得可导
        std = logvar.mul(0.5).exp_()  # 计算标准差
        eps = Variable(std.data.new(std.size()).normal_())  # eps是和原来的形状一样的标准正态分布
        return eps.mul(std).add_(mu)  # 对标准正态分布乘标准差, 然后加上均值, 则得到一个原来的高斯分布的样本, 并且这个采样过程是可导的


    def set_fusion_functions(self):
        weights = utils.reweight_weights(torch.Tensor(self.flags.alpha_modalities))
        # alpha_modalities = [权重正则, 脑模态的权重正则, 视觉模态的权重正则, 文本模态的权重正则]
        # utils.reweight_weights() 确保它们的和为 1
        self.weights = weights.to(self.flags.device)  # 将四个正则项的权重放入GPU
        if self.flags.modality_moe:
            self.modality_fusion = self.moe_fusion
            self.fusion_condition = self.fusion_condition_moe
            self.calc_joint_divergence = self.divergence_static_prior
        elif self.flags.modality_jsd:
            self.modality_fusion = self.moe_fusion
            self.fusion_condition = self.fusion_condition_moe
            self.calc_joint_divergence = self.divergence_dynamic_prior
        elif self.flags.modality_poe:
            self.modality_fusion = self.poe_fusion
            self.fusion_condition = self.fusion_condition_poe
            self.calc_joint_divergence = self.divergence_static_prior
        elif self.flags.joint_elbo:
            #### 默认情况
            self.modality_fusion = self.poe_fusion
            self.fusion_condition = self.fusion_condition_joint
            self.calc_joint_divergence = self.divergence_static_prior


    def divergence_static_prior(self, mus, logvars, weights=None):
        # 计算KL散度
        # 输入: mus和logvars形状都是(n, batch_size, 32), n取决于模态数, 模态数=1,2 ===> n=3, 模态数=3 ===> n=1
        # weights的形状是(模态数,)
        if weights is None:
            # 加载默认权重
            weights=self.weights
        weights = weights.clone()
        weights = utils.reweight_weights(weights) # 将权重归一化为和1
        div_measures = calc_group_divergence_moe(self.flags,
                                                 mus,
                                                 logvars,
                                                 weights,
                                                 normalization=self.flags.batch_size)
        # div_measures = [(batch_size,), (n,)], 分别是n组模态的平均KL散度和, 和n组模态分别的KL散度
        divs = dict()
        divs['joint_divergence'] = div_measures[0]; divs['individual_divs'] = div_measures[1]; divs['dyn_prior'] = None
        # divs = {
        #     'joint_divergence': (batch_size,), n组模态的平均KL散度和
        #     'individual_divs': (n,), n组模态分别的KL散度,
        #     'dyn_prior': None,
        # }
        return divs


    def divergence_dynamic_prior(self, mus, logvars, weights=None):
        if weights is None:
            weights = self.weights;
        div_measures = calc_alphaJSD_modalities(self.flags,
                                                mus,
                                                logvars,
                                                weights,
                                                normalization=self.flags.batch_size);
        divs = dict();
        divs['joint_divergence'] = div_measures[0];
        divs['individual_divs'] = div_measures[1];
        divs['dyn_prior'] = div_measures[2];
        return divs;


    def moe_fusion(self, mus, logvars, weights=None):
        # 输入:
        # mus和logvars是(模态数, batch_size, 32)形状的
        # weights是(模态数, )形状的
        if weights is None:
            weights = self.weights
        weights = utils.reweight_weights(weights)  # 将权重归一化为和1
        #mus = torch.cat(mus, dim=0);
        #logvars = torch.cat(logvars, dim=0);
        mu_moe, logvar_moe = utils.mixture_component_selection(self.flags,
                                                               mus,
                                                               logvars,
                                                               weights)
        # 按给定权重将均值和对数方差划分给不同的模态
        return [mu_moe, logvar_moe]  # 两个的形状都是(模态数, w * batch_size, 32)


    def poe_fusion(self, mus, logvars, weights=None):
        # 输入:
        # mus和logvars是(模态数, batch_size, 32)形状的
        # weights是(模态数, )形状的
        if (self.flags.modality_poe or mus.shape[0] == len(self.modalities.keys())):
            num_samples = mus[0].shape[0]  # batch_size
            mus = torch.cat((mus, torch.zeros(1, num_samples,
                             self.flags.class_dim).to(self.flags.device)),
                             dim=0)
            # mus变成(模态数+1, batch_size, 32)的形状, 且最后一个是全0的(1, batch_size, 32)
            logvars = torch.cat((logvars, torch.zeros(1, num_samples,
                                 self.flags.class_dim).to(self.flags.device)),
                                 dim=0)
            # logvars变成(模态数+1, batch_size, 32)的形状, 且最后一个是全0的(1, batch_size, 32)
            # 加入的模态是标准正态分布, 均值0, 方差1
        # mus = torch.cat(mus, dim=0)
        # logvars = torch.cat(logvars, dim=0)
        mu_poe, logvar_poe = poe(mus, logvars)  # 定义在 ./divergence_measures/mm_div.py
        # 返回融合后的高斯分布的均值和方差, 形状都是(batch_size, 32)
        return [mu_poe, logvar_poe]


    def fusion_condition_moe(self, subset, input_batch=None):
        if len(subset) == 1:
            return True
        else:
            return False


    def fusion_condition_poe(self, subset, input_batch=None):
        if len(subset) == len(input_batch.keys()):
            return True
        else:
            return False


    def fusion_condition_joint(self, subset, input_batch=None):
        return True


    def forward(self, input_batch, K=1):
        # 前向计算
        # 输入: input_batch = {'image': image数据 (Variable对象), 'text': text数据 (Variable对象)}
        latents = self.inference(input_batch)  # 返回一个字典, 计算了各种均值和对数方差
        # latents = {
        #     'mus': (n, batch_size, 32),  # n取决于这里的input_batch的模态数, 模态数=1,2 ===> n=3, 模态数=3 ===> n=1
        #     'logvars': (n, batch_size, 32),  # n取决于这里的input_batch的模态数, 模态数=1,2 ===> n=3, 模态数=3 ===> n=1
        #     'weights': (模态数,),  # 是平均的, 和为1的权重向量
        #     'joint': [(模态数, w * batch_size, 32), (模态数, w * batch_size, 32)],  # w取决于weight, 在这里其实就是平均分割
        #     'subsets': {
        #         'brain': [融合后s_mu, 融合后s_logvar],
        #         'image': [融合后s_mu, 融合后s_logvar],
        #         'text': [融合后s_mu, 融合后s_logvar],
        #         'brain_image': [融合后s_mu, 融合后s_logvar],
        #         'brain_text': [融合后s_mu, 融合后s_logvar],
        #         'image_text': [融合后s_mu, 融合后s_logvar],
        #         'brain_image_text': [融合后s_mu, 融合后s_logvar]
        #     }
        # }
        results = dict()
        results['latents'] = latents
        results['group_distr'] = latents['joint']
        class_embeddings = self.reparameterize(latents['joint'][0], latents['joint'][1])  # 对这个高斯分布可导采样, 得到一个样本
        #### For CUBO ####
        qz_x = dist.Normal(latents['joint'][0],latents['joint'][1].mul(0.5).exp_())  # qz_x是一个高斯分布对象
        zss = qz_x.rsample(torch.Size([K]))  # 采样K个样本, 形状是(K, 模态数, w * batch_size, 32)

        div = self.calc_joint_divergence(latents['mus'],
                                         latents['logvars'],
                                         latents['weights'])
        # div = {
        #     'joint_divergence': (batch_size,), n组模态的平均KL散度和
        #     'individual_divs': (n,), n组模态分别的KL散度,
        #     'dyn_prior': None,
        # }

        for k, key in enumerate(div.keys()):
            results[key] = div[key]

        results_rec = dict()
        px_zs = dict()
        enc_mods = latents['modalities']
        for m, m_key in enumerate(self.modalities.keys()):
            if m_key in input_batch.keys():
                m_s_mu, m_s_logvar = enc_mods[m_key + '_style']
                if self.flags.factorized_representation:  # 默认为False
                    m_s_embeddings = self.reparameterize(mu=m_s_mu, logvar=m_s_logvar)
                else:
                    m_s_embeddings = None
                # self.lhoods[m_key]默认都是dist.Normal

                # self.decoders[m_key](m_s_embeddings, class_embeddings) 输出: x_hat, tensor(0.75)
                # 下面两行: 前面那个传参是完全没用的
                m_rec = self.lhoods[m_key](*self.decoders[m_key](m_s_embeddings, class_embeddings))
                px_z = self.lhoods[m_key](*self.decoders[m_key](m_s_embeddings, zss))
                # m_rec, px_z 都是均值为x_hat, 对数方差为0.75的正态分布
                results_rec[m_key] = m_rec
                px_zs[m_key] = px_z
        results['rec'] = results_rec
        results['class_embeddings'] = class_embeddings
        results['qz_x'] = qz_x
        results['zss'] = zss
        results['px_zs'] = px_zs
        # result = {
        #     'latents': {
        #         'modalities': {
        #              "brain": [latent_space_mu, latent_space_logvar],
        #              "brain_style": [None, None],
        #              "image": [latent_space_mu, latent_space_logvar],
        #              "image_style": [None, None],
        #              "text": [latent_space_mu, latent_space_logvar],
        #              "text_style": [None, None]
        #         }
        #         'mus': (n, batch_size, 32),  # n取决于这里的input_batch的模态数, 模态数=1,2 ===> n=3, 模态数=3 ===> n=1
        #         'logvars': (n, batch_size, 32),  # n取决于这里的input_batch的模态数, 模态数=1,2 ===> n=3, 模态数=3 ===> n=1
        #         'weights': (模态数,),  # 是平均的, 和为1的权重向量
        #         'joint': [(模态数, w * batch_size, 32), (模态数, w * batch_size, 32)],  # w取决于weight, 在这里其实就是平均分割
        #         'subsets': {
        #             'brain': [融合后s_mu, 融合后s_logvar],
        #             'image': [融合后s_mu, 融合后s_logvar],
        #             'text': [融合后s_mu, 融合后s_logvar],
        #             'brain_image': [融合后s_mu, 融合后s_logvar],
        #             'brain_text': [融合后s_mu, 融合后s_logvar],
        #             'image_text': [融合后s_mu, 融合后s_logvar],
        #             'brain_image_text': [融合后s_mu, 融合后s_logvar]
        #         }
        #     }
        #     'group_distr': [(模态数, w * batch_size, 32), (模态数, w * batch_size, 32)],  # w取决于weight, 在这里其实就是平均分割
        #     'joint_divergence': (batch_size,), n组模态的平均KL散度和
        #     'individual_divs': (n,), n组模态分别的KL散度,
        #     'dyn_prior': None,
        #     'rec': {
        #         'brain': 一个dist.Normal对象, 均值为DecoderBrain网络输入为class_embeddings时输出的x_hat, 对数方差为0.75,
        #         'image': 一个dist.Normal对象, 均值为DecoderImage网络输入为class_embeddings时输出的x_hat, 对数方差为0.75,
        #         'text' : 一个dist.Normal对象, 均值为DecoderText 网络输入为class_embeddings时输出的x_hat, 对数方差为0.75,
        #     }
        #     'class_embeddings': 子集之间做MOE之后融合得到的高斯分布的一个样本(可导采样),
        #     'qz_x': dist.Normal对象, 子集之间做MOE之后融合得到的高斯分布,
        #     'zss': (K, 模态数, w * batch_size, 32), 对qz_x采样了K次的样本, K默认为1,
        #     'px_zs': {
        #         'brain': 一个dist.Normal对象, 均值为DecoderBrain网络输入为zss时输出的x_hat, 对数方差为0.75,
        #         'image': 一个dist.Normal对象, 均值为DecoderImage网络输入为zss时输出的x_hat, 对数方差为0.75,
        #         'text' : 一个dist.Normal对象, 均值为DecoderText 网络输入为zss时输出的x_hat, 对数方差为0.75,
        #     }
        # }
        return results

    def encode(self, input_batch):
        latents = dict()
        for m, m_key in enumerate(self.modalities.keys()):
            if m_key in input_batch.keys():
                i_m = input_batch[m_key]  # 数据 (Variable对象)
                l = self.encoders[m_key](i_m)  # 使用对应模态的Encoder网络前向计算, 返回: None, None, latent_space_mu, latent_space_logvar
                latents[m_key + '_style'] = l[:2]  # None, None
                latents[m_key] = l[2:]  # latent_space_mu, latent_space_logvar
            else:
                latents[m_key + '_style'] = [None, None]
                latents[m_key] = [None, None]
        # latents = {
        #     "brain": latent_space_mu, latent_space_logvar,
        #     "brain_style": None, None,
        #     "image": latent_space_mu, latent_space_logvar,
        #     "image_style": None, None,
        #     "text": latent_space_mu, latent_space_logvar,
        #     "text_style": None, None,
        # }
        return latents  # 返回一个字典


    def inference(self, input_batch, num_samples=None):
        # 输入: input_batch = {'image': image数据 (Variable对象), 'text': text数据 (Variable对象)}
        #      num_samples: batch_size
        if num_samples is None:
            num_samples = self.flags.batch_size
        latents = dict()
        enc_mods = self.encode(input_batch)  # 计算encoder前向, 分别计算涉及模态的latent space的均值和对数方差
        # enc_mods是一个字典
        latents['modalities'] = enc_mods
        # latents = {'modalities': {
        #     "brain": [latent_space_mu, latent_space_logvar],
        #     "brain_style": [None, None],
        #     "image": [latent_space_mu, latent_space_logvar],
        #     "image_style": [None, None],
        #     "text": [latent_space_mu, latent_space_logvar],
        #     "text_style": [None, None],
        # }}
        # 定义两个未初始化的张量: mus和logvars
        mus = torch.Tensor().to(self.flags.device)
        logvars = torch.Tensor().to(self.flags.device)
        distr_subsets = dict()
        for k, s_key in enumerate(self.subsets.keys()):
            # subsets = {
            #     '': [],
            #     'brain': [brain的Modality对象],
            #     'image': [image的Modality对象],
            #     'text':  [text 的Modality对象],
            #     'brain_image': [brain的Modality对象, image的Modality对象],
            #     'brain_text' : [brain的Modality对象, text 的Modality对象],
            #     'image_text' : [image的Modality对象, text 的Modality对象],
            #     'brain_image_text': [brain的Modality对象, image的Modality对象, text的Modality对象]
            # }
            if s_key != '':
                mods = self.subsets[s_key]  # mods是一个list of Modality对象
                # 定义两个未初始化的张量: mus_subset和logvars_subset
                mus_subset = torch.Tensor().to(self.flags.device)
                logvars_subset = torch.Tensor().to(self.flags.device)
                mods_avail = True
                for m, mod in enumerate(mods):
                    # m=0,1,2,...    mod=Modality对象,Modality对象,Modality对象,...
                    if mod.name in input_batch.keys():  # "brain", "image", "text", 且在输入的模态内
                        mus_subset = torch.cat((mus_subset,
                                                enc_mods[mod.name][0].unsqueeze(0)),
                                                dim=0)
                        logvars_subset = torch.cat((logvars_subset,
                                                    enc_mods[mod.name][1].unsqueeze(0)),
                                                    dim=0)
                        # enc_mods[mod.name][0]是latent_space_mu (batch_size, 32), enc_mods[mod.name][1]是latent_space_logvar (batch_size, 32)
                        # unsqueeze(0)后都变成(1, batch_size, 32)
                        # 因此, 循环结束后, mus_subset和logvars_subset最多变成(3, batch_size, 32), 取决于这一次模型的输入有多少个模态
                    else:
                        # 当缺失了模态时(不够三个模态)会触发
                        mods_avail = False
                if mods_avail:  # 没有缺失模态
                    # mus_subset是(3, batch_size, 32)
                    weights_subset = ((1/float(len(mus_subset)))*
                                      torch.ones(len(mus_subset)).to(self.flags.device))
                    # subsets的权重
                    # weights_subset = [0.33, 0.33, 0.33]

                    ### 默认情况
                    # self.modality_fusion = self.poe_fusion
                    # self.fusion_condition = self.fusion_condition_joint
                    # self.calc_joint_divergence = self.divergence_static_prior

                    s_mu, s_logvar = self.modality_fusion(mus_subset,
                                                          logvars_subset,
                                                          weights_subset) #子集内部POE#
                    # 对一个子集进行POE
                    # 返回是融合后的高斯分布的均值和对数方差, 形状都是(batch_size, 32)
                    distr_subsets[s_key] = [s_mu, s_logvar]
                    # distr_subsets = {
                    #     'brain': [融合后s_mu, 融合后s_logvar],
                    #     'image': [融合后s_mu, 融合后s_logvar],
                    #     'text':  [融合后s_mu, 融合后s_logvar],
                    #     'brain_image': [融合后s_mu, 融合后s_logvar],
                    #     'brain_text' : [融合后s_mu, 融合后s_logvar],
                    #     'image_text' : [融合后s_mu, 融合后s_logvar],
                    #     'brain_image_text': [融合后s_mu, 融合后s_logvar]
                    # }
                    if self.fusion_condition(mods, input_batch):
                        # 如果len(mods) == len(input_batch.keys()) 就是 True
                        mus = torch.cat((mus, s_mu.unsqueeze(0)), dim=0)
                        logvars = torch.cat((logvars, s_logvar.unsqueeze(0)), dim=0)
                        # 更新融合后的均值和对数方差张量
        # 此时, 获得两个张量: mus和logvars, 分别表示融合后的均值和对数方差, 形状是(n, batch_size, 32)
        # n取决于这里的input_batch的模态数
        # 当模态数为1, 2时 ===> n=3
        # 当模态数为3时    ===> n=1
        if self.flags.modality_jsd:  # 默认情况下不会触发
            num_samples = mus[0].shape[0]
            mus = torch.cat((mus, torch.zeros(1, num_samples,
                                      self.flags.class_dim).to(self.flags.device)), dim=0)
            logvars = torch.cat((logvars, torch.zeros(1, num_samples,
                                          self.flags.class_dim).to(self.flags.device)), dim=0)
            # 再添加一次全0模态
        #weights = (1/float(len(mus)))*torch.ones(len(mus)).to(self.flags.device);
        weights = (1/float(mus.shape[0]))*torch.ones(mus.shape[0]).to(self.flags.device)
        joint_mu, joint_logvar = self.moe_fusion(mus, logvars, weights) #子集之间MOE#
        # 两个的形状都是(模态数, w * batch_size, 32)
        # mus = torch.cat(mus, dim=0);
        # logvars = torch.cat(logvars, dim=0);
        latents['mus'] = mus
        latents['logvars'] = logvars
        latents['weights'] = weights
        latents['joint'] = [joint_mu, joint_logvar]
        latents['subsets'] = distr_subsets
        # latents = {
        #     'mus': (n, batch_size, 32),  # n取决于这里的input_batch的模态数, 模态数=1,2 ===> n=3, 模态数=3 ===> n=1
        #     'logvars': (n, batch_size, 32),  # n取决于这里的input_batch的模态数, 模态数=1,2 ===> n=3, 模态数=3 ===> n=1
        #     'weights': (模态数,),  # 是平均的, 和为1的权重向量
        #     'joint': [(模态数, w * batch_size, 32), (模态数, w * batch_size, 32)],  # w取决于weight, 在这里其实就是平均分割
        #     'subsets': {
        #         'brain': [融合后s_mu, 融合后s_logvar],
        #         'image': [融合后s_mu, 融合后s_logvar],
        #         'text': [融合后s_mu, 融合后s_logvar],
        #         'brain_image': [融合后s_mu, 融合后s_logvar],
        #         'brain_text': [融合后s_mu, 融合后s_logvar],
        #         'image_text': [融合后s_mu, 融合后s_logvar],
        #         'brain_image_text': [融合后s_mu, 融合后s_logvar]
        #     }
        # }
        return latents  # 返回一个字典



    def generate(self, num_samples=None):
        if num_samples is None:
            num_samples = self.flags.batch_size;

        mu = torch.zeros(num_samples,
                         self.flags.class_dim).to(self.flags.device);
        logvar = torch.zeros(num_samples,
                             self.flags.class_dim).to(self.flags.device);
        z_class = self.reparameterize(mu, logvar);
        z_styles = self.get_random_styles(num_samples);
        random_latents = {'content': z_class, 'style': z_styles};
        random_samples = self.generate_from_latents(random_latents);
        return random_samples;


    def generate_sufficient_statistics_from_latents(self, latents):
        suff_stats = dict();
        content = latents['content']
        for m, m_key in enumerate(self.modalities.keys()):
            s = latents['style'][m_key];
            cg = self.lhoods[m_key](*self.decoders[m_key](s, content));
            suff_stats[m_key] = cg;
        return suff_stats;


    def generate_from_latents(self, latents):
        suff_stats = self.generate_sufficient_statistics_from_latents(latents);
        cond_gen = dict();
        for m, m_key in enumerate(latents['style'].keys()):
            cond_gen_m = suff_stats[m_key].mean;
            cond_gen[m_key] = cond_gen_m;
        return cond_gen;


    def cond_generation(self, latent_distributions, num_samples=None):
        if num_samples is None:
            num_samples = self.flags.batch_size;

        style_latents = self.get_random_styles(num_samples);
        cond_gen_samples = dict();
        for k, key in enumerate(latent_distributions.keys()):
            [mu, logvar] = latent_distributions[key];
            content_rep = self.reparameterize(mu=mu, logvar=logvar);
            latents = {'content': content_rep, 'style': style_latents}
            cond_gen_samples[key] = self.generate_from_latents(latents);
        return cond_gen_samples;


    def get_random_style_dists(self, num_samples):
        styles = dict();
        for k, m_key in enumerate(self.modalities.keys()):
            mod = self.modalities[m_key];
            s_mu = torch.zeros(num_samples,
                               mod.style_dim).to(self.flags.device)
            s_logvar = torch.zeros(num_samples,
                                   mod.style_dim).to(self.flags.device);
            styles[m_key] = [s_mu, s_logvar];
        return styles;


    def get_random_styles(self, num_samples):
        styles = dict();
        for k, m_key in enumerate(self.modalities.keys()):
            if self.flags.factorized_representation:
                mod = self.modalities[m_key];
                z_style = torch.randn(num_samples, mod.style_dim);
                z_style = z_style.to(self.flags.device);
            else:
                z_style = None;
            styles[m_key] = z_style;
        return styles;


    def save_networks(self):
        for k, m_key in enumerate(self.modalities.keys()):
            torch.save(self.encoders[m_key].state_dict(),
                       os.path.join(self.flags.dir_checkpoints, 'enc_' +
                                    self.modalities[m_key].name))
            torch.save(self.decoders[m_key].state_dict(),
                       os.path.join(self.flags.dir_checkpoints, 'dec_' +
                                    self.modalities[m_key].name))




