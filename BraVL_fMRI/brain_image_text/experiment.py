import os
import numpy as np 
import itertools
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from modalities.Modality import Modality
from brain_image_text.networks.VAEtrimodal import VAEtrimodal,VAEbimodal
from brain_image_text.networks.QNET import QNet
from brain_image_text.networks.MLP_Brain import EncoderBrain, DecoderBrain
from brain_image_text.networks.MLP_Image import EncoderImage, DecoderImage
from brain_image_text.networks.MLP_Text import EncoderText, DecoderText
from utils.BaseExperiment import BaseExperiment


class BrainImageText(BaseExperiment):
    def __init__(self, flags, alphabet):
        ### alphabet是干嘛的??
        super().__init__(flags)

        self.modalities = self.set_modalities()  # 设置模态, 有三个模态: ['brain', 'image', 'text']
        # self.modalities = {
        #     'brain': Modality对象,
        #     'image': Modality对象,
        #     'text':  Modality对象,
        # }
        self.num_modalities = len(self.modalities.keys())  # 模态数量, 这里是3
        self.subsets = self.set_subsets()  # 定义在基类里, ../utils/BaseExperiment.py
        # 返回一个字典. key是字符串: "brain", "brain_image"这种, value是对应的list of Modality对象

        # 设置数据集
        self.dataset_train = None  # 训练集
        self.dataset_test = None  # 测试集
        self.set_dataset()
        # 至少得到两个: self.dataset_train, self.dataset_test
        # 至多得到四个: self.dataset_train, self.dataset_test, self.dataset_aug, self.dataset_val
        # 它们都是torch的可迭代对象
        self.mm_vae = self.set_model()  # 设置VAE模型
        
        self.optimizer = None  # 优化器
        self.rec_weights = self.set_rec_weights()  # 设置重构损失权重
        # self.rec_weights = {"brain": 1.0, "image": 1.0, "text": 1.0}
        self.style_weights = self.set_style_weights()  # 设置style权重
        # self.style_weights = {"brian": 1.0, "image": 1.0, "text": 1.0}
        self.Q1,self.Q2,self.Q3 = self.set_Qmodel()
        # Q1, Q2, Q3都是: 输入维度 -> 512 -> 输出维度
        self.eval_metric = accuracy_score  # sklearn, 计算分类的准确度

        self.labels = ['digit']  # ?????????????


    def set_model(self):
        # 设置三模态的VAE模型
        model = VAEtrimodal(self.flags, self.modalities, self.subsets)
        # model = model.to(self.flags.device)
        model = nn.DataParallel(model, device_ids=self.flags.device_ids)
        return model  # 返回一个nn.Module的派生类对象(放进GPU里)

    def set_modalities(self):
        # 设置当前对象的模态
        # mod1, mod2, mod3 都是Modality类的对象, /modalities/Modality.py Line 8
        # Modality(模态名, Encoder, Decoder, 共同隐藏factor(论文里Fig.4的z)的维度, 模态的输入维度, 似然分布)
        mod1 = Modality('brain', EncoderBrain(self.flags), DecoderBrain(self.flags),
                    self.flags.class_dim, self.flags.style_m1_dim, 'normal')
        mod2 = Modality('image', EncoderImage(self.flags), DecoderImage(self.flags),
                    self.flags.class_dim, self.flags.style_m2_dim, 'normal')
        mod3 = Modality('text', EncoderText(self.flags), DecoderText(self.flags),
                    self.flags.class_dim, self.flags.style_m3_dim, 'normal')
        mods = {mod1.name: mod1, mod2.name: mod2, mod3.name: mod3}
        return mods

    def set_dataset(self):
        # 设置训练集, 更新该类的属性, 执行后得到:
        # self.dataset_train (训练集), self.dataset_test (测试集)
        # 如果有相关的增强, 则还得到:
        # self.dataset_aug (增强后的数据集)
        # 如果 test_type为'normal', 则还得到:
        # self.dataset_val (验证集)
        # 所有的: self.dataset_train, self.dataset_test, self.dataset_aug, self.dataset_val都是torch的可迭代数据
        ############################################################################################################################
        ## 加载文件路径
        ############################################################################################################################
        # load data 加载数据
        data_dir_root = self.flags.data_dir_root  # 数据集根目录 data_dir_root = "./data/{dataname}" = "./data/DIR-Wiki", ../utils/BaseFlags.py
        sbj = self.flags.sbj  # fMRI的被试对象, ['sub-01', 'sub-02', 'sub-03', 'sub-04']
        stability_ratio = self.flags.stability_ratio  # 看起来没啥乱用 文件命名而已
        image_model = self.flags.image_model  # 图像的embedding模型, 'pytorch/repvgg_b3g4'
        text_model = self.flags.text_model  # 文本的embedding模型, 'GPTNeo'
        roi = self.flags.roi  # 关注的区域(论文里的Region of Interesting), 'LVC_HVC_IT'
        brain_dir = os.path.join(data_dir_root, 'brain_feature', roi, sbj)
        # brain_dir = "./data/DIR-Wiki/brain_feature/LVC_HVC_IT/sub-03"
        image_dir_train = os.path.join(data_dir_root, 'visual_feature/ImageNetTraining', image_model + '-PCA', sbj)
        # image_dir_train = "./data/DIR-Wiki/visual_feature/ImageNetTraining/pytorch/repvgg_b3g4-PCA/sub-03"
        image_dir_test = os.path.join(data_dir_root, 'visual_feature/ImageNetTest', image_model + '-PCA', sbj)
        # image_dir_test = "./data/DIR-Wiki/visual_feature/ImageNetTest/pytorch/repvgg_b3g4-PCA/sub-03"
        text_dir_train = os.path.join(data_dir_root, 'textual_feature/ImageNetTraining/text', text_model, sbj)
        # text_dir_train = "./data/DIR-Wiki/text_feature/ImageNetTraining/text/GPTNeo/sub-03"
        text_dir_test = os.path.join(data_dir_root, 'textual_feature/ImageNetTest/text', text_model, sbj)
        # text_dir_test = "./data/DIR-Wiki/textual_feature/ImageNetTest/text/GPTNeo/sub-03"
        ############################################################################################################################


        ############################################################################################################################
        ## 加载数据集
        ############################################################################################################################
        # 下面是加载训练集
        train_brain = sio.loadmat(os.path.join(brain_dir, 'fmri_train_data'+stability_ratio+'.mat'))['data'].astype('double')
        train_image = sio.loadmat(os.path.join(image_dir_train, 'feat_pca_train.mat'))['data'].astype('double')
        train_text = sio.loadmat(os.path.join(text_dir_train, 'text_feat_train.mat'))['data'].astype('double')
        train_label = sio.loadmat(os.path.join(brain_dir, 'fmri_train_data'+stability_ratio+'.mat'))['class_idx'].T.astype('int')
        # train_brain, train_image, train_text, train_label 都是Numpy数组

        # test_brain = sio.loadmat(os.path.join(brain_dir, 'fmri_test_data_unique.mat'))['data'].astype('double')
        # test_image = sio.loadmat(os.path.join(image_dir_test, 'feat_pca_test_unique.mat'))['data'].astype('double')
        # test_text = sio.loadmat(os.path.join(text_dir_test, 'text_feat_test_unique.mat'))['data'].astype('double')
        # test_label = sio.loadmat(os.path.join(brain_dir, 'fmri_test_data_unique.mat'))['class_idx'].T.astype('int')

        # 下面是加载测试集
        test_brain = sio.loadmat(os.path.join(brain_dir, 'fmri_test_data'+stability_ratio+'.mat'))['data'].astype('double')
        test_image = sio.loadmat(os.path.join(image_dir_test, 'feat_pca_test.mat'))['data'].astype('double')
        test_text = sio.loadmat(os.path.join(text_dir_test, 'text_feat_test.mat'))['data'].astype('double')
        test_label = sio.loadmat(os.path.join(brain_dir, 'fmri_test_data'+stability_ratio+'.mat'))['class_idx'].T.astype('int')
        # test_brain, test_image, test_text, test_label 都是Numpy数组
        ############################################################################################################################


        ############################################################################################################################
        ## 数据增强
        ############################################################################################################################
        # 下面是做数据增强
        if self.flags.aug_type == 'image_text':  # 默认为'image_text', 默认执行
            # image和text都做增强
            image_dir_aug = os.path.join(data_dir_root, 'visual_feature/Aug_1000', image_model + '-PCA', sbj)
            text_dir_aug = os.path.join(data_dir_root, 'textual_feature/Aug_1000/text', text_model, sbj)
            aug_image = sio.loadmat(os.path.join(image_dir_aug, 'feat_pca_aug.mat'))['data'].astype('double')
            aug_text = sio.loadmat(os.path.join(text_dir_aug, 'text_feat_aug.mat'))['data'].astype('double')
            aug_image = torch.from_numpy(aug_image)
            aug_text = torch.from_numpy(aug_text)
            print('aug_image=', aug_image.shape)
            print('aug_text=', aug_text.shape)
        elif self.flags.aug_type == 'text_only':  # 默认为'image_text', 默认不执行
            # 只对text做增强
            text_dir_aug = os.path.join(data_dir_root, 'textual_feature/Aug_1000/text', text_model, sbj)
            aug_text = sio.loadmat(os.path.join(text_dir_aug, 'text_feat_aug.mat'))['data'].astype('double')
            aug_text = aug_text  # 这句话貌似没意义
            aug_text = torch.from_numpy(aug_text)
            print('aug_text=', aug_text.shape)

        elif self.flags.aug_type == 'image_only':  # 默认为'image_text', 默认不执行
            # 只对image做增强
            image_dir_aug = os.path.join(data_dir_root, 'visual_feature/Aug_1000', image_model + '-PCA', sbj)
            aug_image = sio.loadmat(os.path.join(image_dir_aug, 'feat_pca_aug.mat'))['data'].astype('double')
            aug_image = torch.from_numpy(aug_image)
            print('aug_image=', aug_image.shape)
        elif self.flags.aug_type == 'no_aug':  # 默认为'image_text', 默认不执行
            # 不做任何数据增强
            print('no augmentation')
        # 最多获得两个torch变量: aug_image, aug_text
        ############################################################################################################################


        if self.flags.test_type == 'normal':  # 默认为'zsl', 默认不执行
            # 将读入的训练集 划分为: 训练集和验证集
            train_label_stratify = train_label
            # train_test_split():
            # 训练X, 验证X, 训练Y, 验证Y = train_test_split(X, Y, test_size=验证集所占比例, stratify=对哪个变量进行分层 使得训练和验证各类样本比例保持一致)
            train_brain, val_brain, train_label, val_label = train_test_split(train_brain, train_label_stratify, test_size=0.2, stratify=train_label_stratify)
            train_image, val_image, train_label, val_label = train_test_split(train_image, train_label_stratify, test_size=0.2, stratify=train_label_stratify)
            train_text, val_text, train_label, val_label = train_test_split(train_text, train_label_stratify, test_size=0.2, stratify=train_label_stratify)

            val_brain = torch.from_numpy(val_brain)
            val_image = torch.from_numpy(val_image)
            val_text = torch.from_numpy(val_text)
            val_label = torch.from_numpy(val_label)
            print('val_brain=', val_brain.shape)
            print('val_image=', val_image.shape)
            print('val_text=', val_text.shape)
            # 得到四个torch变量: val_brain, val_image, val_text, val_label

        train_brain = torch.from_numpy(train_brain)
        test_brain = torch.from_numpy(test_brain)
        train_image = torch.from_numpy(train_image)
        test_image = torch.from_numpy(test_image)
        train_text = torch.from_numpy(train_text)
        test_text = torch.from_numpy(test_text)
        train_label = torch.from_numpy(train_label)
        test_label = torch.from_numpy(test_label)
        # 将上面的所有变量都转成torch对象


        print('train_brain=', train_brain.shape)
        print('train_image=', train_image.shape)
        print('train_text=', train_text.shape)
        print('test_brain=', test_brain.shape)
        print('test_image=', test_image.shape)
        print('test_text=', test_text.shape)

        self.m1_dim = train_brain.shape[1]
        self.m2_dim = train_image.shape[1]
        self.m3_dim = train_text.shape[1]

        train_dataset = torch.utils.data.TensorDataset(train_brain, train_image, train_text, train_label)
        test_dataset = torch.utils.data.TensorDataset(test_brain, test_image, test_text,test_label)
        # train_dataset 和 test_dataset 都是torch的可迭代数据, 数据格式都是: (脑模态特征, 图像模态特征, 文本模态特征, 标签)

        self.dataset_train = train_dataset
        self.dataset_test = test_dataset

        if self.flags.test_type == 'normal':
            val_dataset = torch.utils.data.TensorDataset(val_brain, val_image, val_text, val_label)
            self.dataset_val = val_dataset

        if self.flags.aug_type == 'image_text':
            aug_dataset = torch.utils.data.TensorDataset(aug_image, aug_text)
            self.dataset_aug = aug_dataset
        elif self.flags.aug_type == 'text_only':
            aug_dataset = torch.utils.data.TensorDataset(aug_text)
            self.dataset_aug = aug_dataset
        elif self.flags.aug_type == 'image_only':
            aug_image = torch.utils.data.TensorDataset(aug_image)
            self.dataset_aug = aug_image
        elif self.flags.aug_type == 'no_aug':
            print('no augmentation')
        # 将上面提到的增强和验证数据集都变成torch的可迭代数据


    def set_optimizer(self):
        # optim.Adam(需要优化的参数, lr=学习率, betas=[一阶动量beta, 二阶动量beta])
        optimizer = optim.Adam(
            itertools.chain(self.mm_vae.parameters(),self.Q1.parameters(),self.Q2.parameters(),self.Q3.parameters()),
            lr=self.flags.initial_learning_rate,
            betas=(self.flags.beta_1, self.flags.beta_2))
        # 优化全部参数
        optimizer_mvae = optim.Adam(
            list(self.mm_vae.parameters()),
            lr=self.flags.initial_learning_rate,
            betas=(self.flags.beta_1, self.flags.beta_2))
        # 优化VAE的参数
        optimizer_Qnet = optim.Adam(
            itertools.chain(self.Q1.parameters(),self.Q2.parameters(),self.Q3.parameters()),
            lr=self.flags.initial_learning_rate,
            betas=(self.flags.beta_1, self.flags.beta_2))
        # 优化Q的参数
        self.optimizer = {'mvae':optimizer_mvae,'Qnet':optimizer_Qnet,'all':optimizer}

        # optim.lr_scheduler.StepLR(优化器类, step_size=每隔多少个epoch将学习率乘以gamma, gamma=乘多少)
        scheduler_mvae = optim.lr_scheduler.StepLR(optimizer_mvae, step_size=20, gamma=1.0)
        scheduler_Qnet = optim.lr_scheduler.StepLR(optimizer_Qnet, step_size=20, gamma=1.0)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=1.0)
        self.scheduler = {'mvae': scheduler_mvae, 'Qnet': scheduler_Qnet, 'all': scheduler}


    def set_Qmodel(self):
        # 设置Q, 论文的Fig.4 A中的Q
        Q1 = QNet(input_dim=self.flags.m1_dim, latent_dim=self.flags.class_dim).cuda()
        Q2 = QNet(input_dim=self.flags.m2_dim, latent_dim=self.flags.class_dim).cuda()
        Q3 = QNet(input_dim=self.flags.m3_dim, latent_dim=self.flags.class_dim).cuda()
        return Q1, Q2 ,Q3  # 三个模态的Q(nn.Module派生类)

    def set_rec_weights(self):
        # 设置重构损失权重
        weights = dict()
        weights['brain'] = self.flags.beta_m1_rec
        weights['image'] = self.flags.beta_m2_rec
        weights['text'] = self.flags.beta_m3_rec
        return weights  # 返回一个权重字典

    def set_style_weights(self):
        # 设置style权重
        weights = dict()
        weights['brain'] = self.flags.beta_m1_style
        weights['image'] = self.flags.beta_m2_style
        weights['text'] = self.flags.beta_m3_style
        return weights  # 返回一个权重字典
