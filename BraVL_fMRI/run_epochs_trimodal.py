import os
import numpy as np
import math
import random
import torch
from torch.autograd import Variable
import torch.distributions as dist
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from divergence_measures.kl_div import calc_kl_divergence
from sklearn.svm import SVC
from sklearn.metrics import top_k_accuracy_score
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns
import csv
from utils import utils
from utils.TBLogger import TBLogger

sns.set(rc={'figure.figsize':(11.7,8.27)})  # 将seaborn的图片大小设置成: 11.7 x 8.27
palette = sns.color_palette("bright", 2)  # 用于设置seaborn的颜色
# torch.set_default_tensor_type(torch.DoubleTensor)  # torch的float默认设置为双精度
torch.set_default_tensor_type(torch.FloatTensor)
TINY = 1e-8  # 近似无穷小
CONSTANT = 1e5  # 近似无穷大
# global variables
SEED = 2021  # 随机数种子
SAMPLE1 = None
if SEED is not None:
    # 设置随机数种子
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)

def log_mean_exp(value, dim=0, keepdim=False):
    return torch.logsumexp(value, dim, keepdim=keepdim) - math.log(value.size(dim))

def m_cubo( x, qz_x, px_zs, zss):
    lpz = dist.Normal(torch.zeros(1,zss.size(2)).cuda(),torch.ones(1,zss.size(2)).cuda()).log_prob(zss).sum(-1)
    lqz_x = qz_x.log_prob(zss).sum(-1)
    if 'brain' in px_zs.keys() and 'image' in px_zs.keys() and 'text' in px_zs.keys():
        lpx_z1 = px_zs['image'].log_prob(x['image']).sum(-1)
        lpx_z2 = px_zs['text'].log_prob(x['text']).sum(-1)
        lpx_z3 = px_zs['brain'].log_prob(x['brain']).sum(-1)
        cubo = 0.5*log_mean_exp(2*(lpz+lpx_z1+lpx_z2+lpx_z3-lqz_x))
    elif 'brain' not in px_zs.keys() and 'image' in px_zs.keys() and 'text' in px_zs.keys():
        lpx_z1 = px_zs['image'].log_prob(x['image']).sum(-1)
        lpx_z2 = px_zs['text'].log_prob(x['text']).sum(-1)
        cubo = 0.5*log_mean_exp(2*(lpz+lpx_z1+lpx_z2-lqz_x))
    return cubo.mean()

def log_li(x_var, dist_info):
    mean   = dist_info[0]  # Qnet对均值的预测, (batch_size, 32)
    std    = dist_info[1]  # tensor(0.75)
    epsilon = (x_var - mean) / (std + TINY)
    pi = Variable(torch.ones(1) * np.pi).to(x_var.device)
    logli = - 0.5 * torch.log(2 * pi) - torch.log(std + TINY) - 0.5 * torch.pow(epsilon,2)
    return logli.sum(1)

def mutual_info(exp,px_zs,z):
    # 计算互信息
    # px_zs['brain']是dist.Normal对象, z是一个tensor对象
    if 'brain' in px_zs.keys() and 'image' in px_zs.keys() and 'text' in px_zs.keys():
        q1 = exp.Q1(px_zs['brain'].loc)
        q2 = exp.Q2(px_zs['image'].loc)
        q3 = exp.Q3(px_zs['text'].loc)
        # q1, q2, q3 分别为 Q1, Q2, Q3 根据生成的x_hat对z的重构
        # Q1, Q2, Q3 都返回 [对均值的预测, tensor(0.5)]
        # 下面是计算互信息
        mi1 = log_li(z,q1).mean()
        mi2 = log_li(z,q2).mean()
        mi3 = log_li(z,q3).mean()
        return mi1 + mi2 + mi3
    elif 'brain' not in px_zs.keys() and 'image' in px_zs.keys() and 'text' in px_zs.keys():
        q2 = exp.Q2(px_zs['image'].loc)
        q3 = exp.Q3(px_zs['text'].loc)
        mi2 = log_li(z,q2).mean()
        mi3 = log_li(z,q3).mean()
        return mi2 + mi3
    elif 'brain' in px_zs.keys() and 'image' not in px_zs.keys() and 'text' not in px_zs.keys():
        q1 = exp.Q1(px_zs['brain'].loc)
        mi1 = log_li(z,q1).mean()
        return mi1
    elif 'brain' not in px_zs.keys() and 'image' in px_zs.keys() and 'text' not in px_zs.keys():
        q2 = exp.Q2(px_zs['image'].loc)
        mi2 = log_li(z,q2).mean()
        return mi2
    elif 'brain' not in px_zs.keys() and 'image' not in px_zs.keys() and 'text' in px_zs.keys():
        q3 = exp.Q3(px_zs['text'].loc)
        mi3 = log_li(z,q3).mean()
        return mi3
    elif 'brain' in px_zs.keys() and 'image' in px_zs.keys() and 'text' not in px_zs.keys():
        q1 = exp.Q1(px_zs['brain'].loc)
        q2 = exp.Q2(px_zs['image'].loc)
        mi1 = log_li(z,q1).mean()
        mi2 = log_li(z,q2).mean()
        return mi1+mi2
    elif 'brain' in px_zs.keys() and 'image' not in px_zs.keys() and 'text' in px_zs.keys():
        q1 = exp.Q1(px_zs['brain'].loc)
        q3 = exp.Q3(px_zs['text'].loc)
        mi1 = log_li(z,q1).mean()
        mi3 = log_li(z,q3).mean()
        return mi1+mi3


def calc_log_probs(exp, result, batch):
    mods = exp.modalities
    log_probs = dict()
    weighted_log_prob = 0.0
    for m, m_key in enumerate(mods.keys()):
        if m_key in batch[0].keys():
            mod = mods[m_key]
            log_probs[mod.name] = -mod.calc_log_prob(result['rec'][mod.name],
                                                     batch[0][mod.name],
                                                     exp.flags.batch_size)
            weighted_log_prob += exp.rec_weights[mod.name]*log_probs[mod.name]
        else:
            mod = mods[m_key]
            log_probs[mod.name] = 0
            weighted_log_prob += exp.rec_weights[mod.name]*log_probs[mod.name]
    return log_probs, weighted_log_prob


def calc_klds(exp, result):
    latents = result['latents']['subsets']
    klds = dict()
    for m, key in enumerate(latents.keys()):
        mu, logvar = latents[key]
        klds[key] = calc_kl_divergence(mu, logvar,
                                       norm_value=exp.flags.batch_size)
    return klds


def calc_klds_style(exp, result):
    latents = result['latents']['modalities']
    klds = dict()
    for m, key in enumerate(latents.keys()):
        if key.endswith('style'):
            mu, logvar = latents[key]
            klds[key] = calc_kl_divergence(mu, logvar,
                                           norm_value=exp.flags.batch_size)
    return klds


def calc_style_kld(exp, klds):
    mods = exp.modalities
    style_weights = exp.style_weights
    weighted_klds = 0.0
    for m, m_key in enumerate(mods.keys()):
        weighted_klds += style_weights[m_key]*klds[m_key+'_style']
    return weighted_klds

def shuffle(a):
    return a[torch.randperm(a.size()[0])]

def true_neg_idx(data, shuffle_data):
    a = data.mean(1)
    b = shuffle_data.mean(1)
    index = torch.arange(0, len(a))
    idx = index[a!=b]
    return idx

def negative_sample_generator(batch_new, batch, case):
    batch_d = batch_new[0]
    data = batch
    if 'brain' in batch_d.keys() and 'image' in batch_d.keys() and 'text' in batch_d.keys():
        if case==1:
            shuffle_data = shuffle(data[0])
            idx = true_neg_idx(data[0], shuffle_data)
            neg_batch = [shuffle_data[idx,:], data[1][idx,:], data[2][idx,:]]
            batch = [neg_batch[0], neg_batch[1], neg_batch[2]]
            batch_new[0] = dict()
            batch_new[0] = {'brain': batch[0], 'image': batch[1], 'text': batch[2]}
            # batch_new[1] = data[3]
            return batch_new

        elif case == 2:
            shuffle_data = shuffle(data[1])
            idx = true_neg_idx(data[1], shuffle_data)
            neg_batch = [data[0][idx,:], shuffle_data[idx,:], data[2][idx,:]]
            batch = [neg_batch[0], neg_batch[1], neg_batch[2]]
            batch_new[0] = dict()
            batch_new[0] = {'brain': batch[0], 'image': batch[1], 'text': batch[2]}
            # batch_new[1] = data[3]
            return batch_new

        elif case == 3:
            shuffle_data = shuffle(data[2])
            idx = true_neg_idx(data[2], shuffle_data)
            neg_batch = [data[0][idx,:], data[1][idx,:], shuffle_data[idx,:]]
            batch = [neg_batch[0], neg_batch[1], neg_batch[2]]
            batch_new[0] = dict()
            batch_new[0] = {'brain': batch[0], 'image': batch[1], 'text': batch[2]}
            # batch_new[1] = data[3]
            return batch_new

        elif case == 4:
            shuffle_data0 = shuffle(data[0])
            idx1 = true_neg_idx(data[0], shuffle_data0)
            shuffle_data1 = shuffle(data[1])
            idx2 = true_neg_idx(data[1], shuffle_data1)
            idx = np.unique(np.concatenate((idx1,idx2),axis=0))
            neg_batch = [shuffle_data0[idx,:], shuffle_data1[idx,:], data[2][idx,:]]
            batch = [neg_batch[0], neg_batch[1], neg_batch[2]]
            batch_new[0] = dict()
            batch_new[0] = {'brain': batch[0], 'image': batch[1], 'text': batch[2]}
            # batch_new[1] = data[3]
            return batch_new

        elif case == 5:
            shuffle_data0 = shuffle(data[0])
            idx1 = true_neg_idx(data[0], shuffle_data0)
            shuffle_data2 = shuffle(data[2])
            idx2 = true_neg_idx(data[2], shuffle_data2)
            idx = np.unique(np.concatenate((idx1,idx2),axis=0))
            neg_batch = [shuffle_data0[idx,:], data[1][idx,:], shuffle_data2[idx,:]]
            batch = [neg_batch[0], neg_batch[1], neg_batch[2]]
            batch_new[0] = dict()
            batch_new[0] = {'brain': batch[0], 'image': batch[1], 'text': batch[2]}
            # batch_new[1] = data[3]
            return batch_new

        elif case == 6:
            shuffle_data1 = shuffle(data[1])
            idx1 = true_neg_idx(data[1], shuffle_data1)
            shuffle_data2 = shuffle(data[2])
            idx2 = true_neg_idx(data[2], shuffle_data2)
            idx = np.unique(np.concatenate((idx1,idx2),axis=0))
            neg_batch = [data[0][idx,:], shuffle_data1[idx,:], shuffle_data2[idx,:]]
            batch = [neg_batch[0], neg_batch[1], neg_batch[2]]
            batch_new[0] = dict()
            batch_new[0] = {'brain': batch[0], 'image': batch[1], 'text': batch[2]}
            # batch_new[1] = data[3]
            return batch_new

    elif 'brain' not in batch_d.keys() and 'image' in batch_d.keys() and 'text' in batch_d.keys():
        if case == 1:
            shuffle_data = shuffle(data[0])
            idx = true_neg_idx(data[0], shuffle_data)
            neg_batch = [shuffle_data[idx,:], data[1][idx,:]]
            batch = [neg_batch[0], neg_batch[1]]
            batch_new[0] = dict()
            batch_new[0] = {'image': batch[0], 'text': batch[1]}
            # batch_new[1] = data[3]
            return batch_new
        elif case == 2:
            shuffle_data = shuffle(data[1])
            idx = true_neg_idx(data[1], shuffle_data)
            neg_batch = [data[0][idx,:], shuffle_data[idx,:]]
            batch = [neg_batch[0], neg_batch[1]]
            batch_new[0] = dict()
            batch_new[0] = {'image': batch[0], 'text': batch[1]}
            # batch_new[1] = data[3]
            return batch_new

def basic_routine_epoch(exp, batch, epoch):
    # set up weights
    # 输入:
    # exp是一个BrainImageText类对象
    # batch = {0: {'image':batch[0],'text':batch[1]}}
    beta_style = exp.flags.beta_style  # 默认是1.0
    beta_content = exp.flags.beta_content  # 默认是1.0
    beta = exp.flags.beta + epoch * 0.01  # exp.flags.beta默认是0.0, 这是权重惩罚项初始权重
    if beta>1.0:
        beta = 1.0  # beta最大是1
    rec_weight = 1.0
    lambda1 = exp.flags.lambda1  # 默认0.001, 同模态互信息惩罚权重


    mm_vae = exp.mm_vae
    batch_d = batch[0]
    mods = exp.modalities
    for k, m_key in enumerate(batch_d.keys()):
        batch_d[m_key] = Variable(batch_d[m_key]).to(exp.flags.device)
    results = mm_vae(batch_d)

    log_probs, weighted_log_prob = calc_log_probs(exp, results, batch)
    group_divergence = results['joint_divergence']

    klds = calc_klds(exp, results)
    z = results['class_embeddings']
    px_zs = results['rec']
    intra_mi = -mutual_info(exp,px_zs,z)
    if exp.flags.factorized_representation:
        klds_style = calc_klds_style(exp, results)

    if (exp.flags.modality_jsd or exp.flags.modality_moe
        or exp.flags.joint_elbo):
        if exp.flags.factorized_representation:
            kld_style = calc_style_kld(exp, klds_style)
        else:
            kld_style = 0.0
        kld_content = group_divergence
        kld_weighted = beta_style * kld_style + beta_content * kld_content
        elbo_loss = rec_weight * weighted_log_prob + beta * kld_weighted
    elif exp.flags.modality_poe:
        klds_joint = {'content': group_divergence,
                      'style': dict()}
        elbos = dict()
        for m, m_key in enumerate(mods.keys()):
            mod = mods[m_key]
            if exp.flags.factorized_representation:
                kld_style_m = klds_style[m_key + '_style']
            else:
                kld_style_m = 0.0
            klds_joint['style'][m_key] = kld_style_m
            if exp.flags.poe_unimodal_elbos:
                if m_key=='brain' and 'brain' in batch_d.keys():
                    i_batch_mod = {m_key: batch_d[m_key]}
                    r_mod = mm_vae(i_batch_mod)
                    log_prob_mod = -mod.calc_log_prob(r_mod['rec'][m_key],
                                                      batch_d[m_key],
                                                      exp.flags.batch_size)
                    log_prob = {m_key: log_prob_mod}
                    klds_mod = {'content': klds[m_key],
                                'style': {m_key: kld_style_m}}
                    elbo_mod = utils.calc_elbo(exp, m_key, log_prob, klds_mod)
                    elbos[m_key] = elbo_mod
                elif m_key=='brain' and 'brain' not in batch_d.keys():
                    elbos[m_key] = 0
                elif m_key=='image' and 'image' in batch_d.keys():
                    i_batch_mod = {m_key: batch_d[m_key]}
                    r_mod = mm_vae(i_batch_mod)
                    log_prob_mod = -mod.calc_log_prob(r_mod['rec'][m_key],
                                                      batch_d[m_key],
                                                      exp.flags.batch_size)
                    log_prob = {m_key: log_prob_mod}
                    klds_mod = {'content': klds[m_key],
                                'style': {m_key: kld_style_m}}
                    elbo_mod = utils.calc_elbo(exp, m_key, log_prob, klds_mod)
                    elbos[m_key] = elbo_mod
                elif m_key=='image' and 'image' not in batch_d.keys():
                    elbos[m_key] = 0
                elif m_key == 'text' and 'text' in batch_d.keys():
                    i_batch_mod = {m_key: batch_d[m_key]}
                    r_mod = mm_vae(i_batch_mod)
                    log_prob_mod = -mod.calc_log_prob(r_mod['rec'][m_key],
                                                      batch_d[m_key],
                                                      exp.flags.batch_size)
                    log_prob = {m_key: log_prob_mod}
                    klds_mod = {'content': klds[m_key],
                                'style': {m_key: kld_style_m}}
                    elbo_mod = utils.calc_elbo(exp, m_key, log_prob, klds_mod)
                    elbos[m_key] = elbo_mod
                elif m_key=='text' and 'text' not in batch_d.keys():
                    elbos[m_key] = 0
        elbo_joint = utils.calc_elbo(exp, 'joint', log_probs, klds_joint)
        elbos['joint'] = elbo_joint
        elbo_loss = sum(elbos.values())

    total_loss = elbo_loss + lambda1 * intra_mi

    out_basic_routine = dict()
    out_basic_routine['results'] = results
    out_basic_routine['log_probs'] = log_probs
    out_basic_routine['total_loss'] = total_loss
    out_basic_routine['klds'] = klds
    out_basic_routine['intra_mi'] = intra_mi
    out_basic_routine['elbo_loss'] = elbo_loss
    return out_basic_routine


def elbo_contrast(exp, batch, epoch):
    # set up weights
    beta_style = exp.flags.beta_style
    beta_content = exp.flags.beta_content
    beta = exp.flags.beta + epoch * 0.01
    if beta>1.0:
        beta = 1.0
    rec_weight = 1.0

    mm_vae = exp.mm_vae
    batch_d = batch[0]
    mods = exp.modalities
    for k, m_key in enumerate(batch_d.keys()):
        batch_d[m_key] = Variable(batch_d[m_key]).to(exp.flags.device)
    results = mm_vae(batch_d, K=30)
    cubo = m_cubo(batch_d, results['qz_x'], results['px_zs'], results['zss'])

    log_probs, weighted_log_prob = calc_log_probs(exp, results, batch)
    group_divergence = results['joint_divergence']

    klds = calc_klds(exp, results)
    z = results['class_embeddings']

    neg_batch_size = z.shape[0]
    if exp.flags.factorized_representation:
        klds_style = calc_klds_style(exp, results)

    if (exp.flags.modality_jsd or exp.flags.modality_moe
        or exp.flags.joint_elbo):
        if exp.flags.factorized_representation:
            kld_style = calc_style_kld(exp, klds_style)
        else:
            kld_style = 0.0
        kld_content = group_divergence
        kld_weighted = beta_style * kld_style + beta_content * kld_content
        elbo_loss = rec_weight * weighted_log_prob + beta * kld_weighted
    elif exp.flags.modality_poe:
        klds_joint = {'content': group_divergence,
                      'style': dict()}
        elbos = dict()
        for m, m_key in enumerate(mods.keys()):
            mod = mods[m_key]
            if exp.flags.factorized_representation:
                kld_style_m = klds_style[m_key + '_style']
            else:
                kld_style_m = 0.0
            klds_joint['style'][m_key] = kld_style_m
            if exp.flags.poe_unimodal_elbos:
                if m_key=='brain' and 'brain' in batch_d.keys():
                    i_batch_mod = {m_key: batch_d[m_key]}
                    r_mod = mm_vae(i_batch_mod)
                    log_prob_mod = -mod.calc_log_prob(r_mod['rec'][m_key],
                                                      batch_d[m_key],
                                                      exp.flags.batch_size)
                    log_prob = {m_key: log_prob_mod}
                    klds_mod = {'content': klds[m_key],
                                'style': {m_key: kld_style_m}}
                    elbo_mod = utils.calc_elbo(exp, m_key, log_prob, klds_mod)
                    elbos[m_key] = elbo_mod
                elif m_key=='brain' and 'brain' not in batch_d.keys():
                    elbos[m_key] = 0
                elif m_key=='image' and 'image' in batch_d.keys():
                    i_batch_mod = {m_key: batch_d[m_key]}
                    r_mod = mm_vae(i_batch_mod)
                    log_prob_mod = -mod.calc_log_prob(r_mod['rec'][m_key],
                                                      batch_d[m_key],
                                                      exp.flags.batch_size)
                    log_prob = {m_key: log_prob_mod}
                    klds_mod = {'content': klds[m_key],
                                'style': {m_key: kld_style_m}}
                    elbo_mod = utils.calc_elbo(exp, m_key, log_prob, klds_mod)
                    elbos[m_key] = elbo_mod
                elif m_key=='image' and 'image' not in batch_d.keys():
                    elbos[m_key] = 0
                elif m_key == 'text' and 'text' in batch_d.keys():
                    i_batch_mod = {m_key: batch_d[m_key]}
                    r_mod = mm_vae(i_batch_mod)
                    log_prob_mod = -mod.calc_log_prob(r_mod['rec'][m_key],
                                                      batch_d[m_key],
                                                      exp.flags.batch_size)
                    log_prob = {m_key: log_prob_mod}
                    klds_mod = {'content': klds[m_key],
                                'style': {m_key: kld_style_m}}
                    elbo_mod = utils.calc_elbo(exp, m_key, log_prob, klds_mod)
                    elbos[m_key] = elbo_mod
                elif m_key=='text' and 'text' not in batch_d.keys():
                    elbos[m_key] = 0

        elbo_joint = utils.calc_elbo(exp, 'joint', log_probs, klds_joint)
        elbos['joint'] = elbo_joint
        elbo_loss = sum(elbos.values())

    # elbo_scale = - elbo_loss / CONSTANT
    elbo_scale = cubo / CONSTANT
    out_basic_routine = dict()
    out_basic_routine['elbo_nega_sample_loss'] = torch.log(elbo_scale.exp().sum() * exp.flags.batch_size / neg_batch_size + TINY) * CONSTANT
    return out_basic_routine

def update_Qnet(exp, batch):
    # 更新 Q1, Q2, Q3 三个网络的参数
    # 输入: exp: BrainImageText类对象
    #      batch: {0: {'image': image数据, 'text': text数据}}
    with torch.no_grad():
        # 下面操作不计算梯度
        mm_vae = exp.mm_vae  # 一个VAEtrimodal类对象
        batch_d = batch[0]  # batch_d = {'image': image数据, 'text': text数据}
        # 让batch_d中的数据变为能够自动求导的Variable对象
        for k, m_key in enumerate(batch_d.keys()):
            batch_d[m_key] = Variable(batch_d[m_key]).cuda()
        results = mm_vae(batch_d)  # 前向计算, forward定义在 ./utils/BaseMMVae.py
        # results是一个特别恶心的字典
        z = results['class_embeddings']  # 子集之间做MOE之后融合得到的高斯分布的一个样本(可导采样)
        px_zs = results['rec']
        #'px_zs': {
        #     'brain': 一个dist.Normal对象, 均值为DecoderBrain网络输入为class_embeddings时输出的x_hat, 对数方差为0.75,
        #     'image': 一个dist.Normal对象, 均值为DecoderImage网络输入为class_embeddings时输出的x_hat, 对数方差为0.75,
        #     'text' : 一个dist.Normal对象, 均值为DecoderText 网络输入为class_embeddings时输出的x_hat, 对数方差为0.75,
        # }
    intra_mi = -mutual_info(exp, px_zs, z)  # 计算一个模态自己的分布和隐变量z之间的互信息
    return intra_mi  # 返回同模态间的互信息

def train_aug(epoch, exp, tb_logger):
    # 用增强的数据来训练
    # 输入: epoch是一个int, exp是一个BrainImageText类对象, tb_logger是一个TBLogger类对象
    mm_vae = exp.mm_vae  # mm_vae是一个VAEtrimodal类对象
    mm_vae.train()  # 将mm_vae设置为训练模式
    exp.mm_vae = mm_vae  # 更新exp的mm_vae属性
    lambda2 = exp.flags.lambda2  # 不同模态互信息惩罚项的默认权重, 论文式11的λ2

    # 根据不同的增强方法进行训练
    if exp.flags.aug_type == 'image_text':
        # 图像和文本均增强
        print('aug type: image_text')
        # 创建DataLoader
        # aug_dataset = torch.utils.data.TensorDataset(aug_image, aug_text)
        aug_loader = DataLoader(exp.dataset_aug,                  # 数据集
                                batch_size=exp.flags.batch_size,  # 批量大小
                                shuffle=True,                     # 随机打乱数据集顺序
                                num_workers=8,                    # 用于数据加载的进程数
                                drop_last=True)                   # 如果数据不能被批量大小整除, 丢弃最后一个批量

        for iteration, batch in enumerate(aug_loader):
            batch = [torch.tensor(tensor) for tensor in batch]
            batch = [tensor.cuda() for tensor in batch]

            batch_new = {}
            batch_new[0] = dict()
            batch_new[0] = {'image':batch[0],'text':batch[1]}  # batch[0]是aug_image数据, batch[1]是aug_text数据
            # batch_new[1] = batch[0]
            batch = [batch[0], batch[1]]

            # brain ---> Encoder ---> x_b  ------- \                / -------> Decoder ---> x'_b ---> Q_1 ---> q1
            # image ---> Encoder ---> x_i  ---------(MoPoE)==> z ==> --------> Decoder ---> x'_i ---> Q_2 ---> q2
            # text  ---> Encoder ---> x_t  ------- /           |    \ -------> Decoder ---> x'_t ---> Q_3 ---> q3
            #                                    /             |                  |                    |
            #                                  /               |                  |                    |
            #                             q_phi(z|x)         p(z|x)           p_theta(x|z)         Q_psai(z|x)
            #                         以phi为参数的网络         条件分布         以theta为参数的网络     以psai为参数的网络

            # 输入: x, 输出 x', 中间的隐藏向量是z
            # z的生成: q_phi(z|x)
            # x的解码: p_theta(x|z)
            # Q_psai(z|x) 用于同模态互信息下界紧致的辅助分布


            # 分两个阶段学习, 看论文 Algorithm 1
            # Stage 1
            # 第一阶段, 优化: Q1, Q2, Q3 三个网络(更新参数psai), 让Q_psai(z|x)逼近p(z|x), 即最大化z和q1,q2,q3的互信息
            intra_mi = update_Qnet(exp, batch_new)  # 返回同模态间的互信息 (z和q1 + z和q2 + z和q3)
            # 清空累计的梯度
            exp.optimizer['Qnet'].zero_grad()
            exp.optimizer['mvae'].zero_grad()
            intra_mi.backward()  # 反向传播
            exp.optimizer['Qnet'].step()  # 参数更新

            # Stage 2
            # 第二阶段, 优化: Encoder和Decoder两个网络(更新参数phi和theta), 让q_phi能够更好地编码(语义对齐),
            # 即最大化: (1)z和q1,q2,q3的互信息, (2)q1,q2,q3之间的互信息
            basic_routine = basic_routine_epoch(exp, batch_new, epoch)
            total_loss = basic_routine['total_loss']
            elbo_loss = basic_routine['elbo_loss']

            exp.optimizer['mvae'].zero_grad()
            exp.optimizer['Qnet'].zero_grad()

            neg_batch_new = negative_sample_generator(batch_new, batch, case=1)
            basic_routine_contrast=elbo_contrast(exp, neg_batch_new, epoch)
            elbo_nega_sample_loss_case_1 = basic_routine_contrast['elbo_nega_sample_loss']

            neg_batch_new = negative_sample_generator(batch_new, batch, case=2)
            basic_routine_contrast=elbo_contrast(exp, neg_batch_new, epoch)
            elbo_nega_sample_loss_case_2 = basic_routine_contrast['elbo_nega_sample_loss']

            elbo_nega_sample_loss = elbo_nega_sample_loss_case_1 + elbo_nega_sample_loss_case_2
            inter_mi_loss = elbo_loss + elbo_nega_sample_loss/2.0

            total_loss = total_loss + lambda2 * inter_mi_loss
            total_loss.backward()
            exp.optimizer['mvae'].step()
            print('====> Epoch: {:03d} Train loss: {:.4f}'.format(epoch, total_loss))
    elif exp.flags.aug_type == 'text_only':
        print('aug type: text_only')
        aug_loader = DataLoader(exp.dataset_aug, batch_size=exp.flags.batch_size,
                                shuffle=True,
                                num_workers=8, drop_last=True)
        for iteration, batch in enumerate(aug_loader):
            batch_new = {}
            batch_new[0] = dict()
            batch_new[0] = {'text': batch[0]}
            # batch_new[1] = batch[0]

            # Stage 1
            intra_mi = update_Qnet(exp, batch_new)
            exp.optimizer['Qnet'].zero_grad()
            exp.optimizer['mvae'].zero_grad()
            intra_mi.backward()
            exp.optimizer['Qnet'].step()

            # Stage 2
            basic_routine = basic_routine_epoch(exp, batch_new, epoch)
            total_loss = basic_routine['total_loss']

            exp.optimizer['mvae'].zero_grad()
            exp.optimizer['Qnet'].zero_grad()

            total_loss.backward()
            exp.optimizer['mvae'].step()
            print('====> Epoch: {:03d} Train loss: {:.4f}'.format(epoch, total_loss))
    elif exp.flags.aug_type == 'image_only':
        print('aug type: image_only')
        aug_loader = DataLoader(exp.dataset_aug, batch_size=exp.flags.batch_size,
                                shuffle=True,
                                num_workers=8, drop_last=True)
        for iteration, batch in enumerate(aug_loader):
            batch_new = {}
            batch_new[0] = dict()
            batch_new[0] = {'image': batch[0]}
            # batch_new[1] = batch[0]

            # Stage 1
            intra_mi = update_Qnet(exp, batch_new)
            exp.optimizer['Qnet'].zero_grad()
            exp.optimizer['mvae'].zero_grad()
            intra_mi.backward()
            exp.optimizer['Qnet'].step()

            # Stage 2
            basic_routine = basic_routine_epoch(exp, batch_new, epoch)
            total_loss = basic_routine['total_loss']

            exp.optimizer['mvae'].zero_grad()
            exp.optimizer['Qnet'].zero_grad()

            total_loss.backward()
            exp.optimizer['mvae'].step()
            print('====> Epoch: {:03d} Train loss: {:.4f}'.format(epoch, total_loss))
    elif exp.flags.aug_type == 'no_aug':
        print('aug type: no augmentation')

def train(epoch, exp, tb_logger):
    mm_vae = exp.mm_vae
    mm_vae.train()
    exp.mm_vae = mm_vae
    lambda2 = exp.flags.lambda2

    test_loader = DataLoader(exp.dataset_test, batch_size=exp.flags.batch_size,
                          shuffle=True,
                          num_workers=8, drop_last=True)

    for iteration, batch in enumerate(test_loader):
        batch = [torch.tensor(tensor) for tensor in batch]
        batch = [tensor.cuda() for tensor in batch]
        batch_new = {}
        batch_new[0] = dict()
        batch_new[0] = {'image':batch[1],'text':batch[2]}
        # batch_new[1] = batch[3]
        batch = [batch[1],batch[2]]

        # Stage 1
        intra_mi = update_Qnet(exp, batch_new)
        exp.optimizer['Qnet'].zero_grad()
        exp.optimizer['mvae'].zero_grad()
        intra_mi.backward()
        exp.optimizer['Qnet'].step()

        # Stage 2
        basic_routine = basic_routine_epoch(exp, batch_new, epoch)
        total_loss = basic_routine['total_loss']
        elbo_loss = basic_routine['elbo_loss']

        exp.optimizer['mvae'].zero_grad()
        exp.optimizer['Qnet'].zero_grad()

        neg_batch_new = negative_sample_generator(batch_new, batch, case=1)
        basic_routine_contrast=elbo_contrast(exp, neg_batch_new, epoch)
        elbo_nega_sample_loss_case_1 = basic_routine_contrast['elbo_nega_sample_loss']

        neg_batch_new = negative_sample_generator(batch_new, batch, case=2)
        basic_routine_contrast=elbo_contrast(exp, neg_batch_new, epoch)
        elbo_nega_sample_loss_case_2 = basic_routine_contrast['elbo_nega_sample_loss']

        elbo_nega_sample_loss = elbo_nega_sample_loss_case_1 + elbo_nega_sample_loss_case_2
        inter_mi_loss = elbo_loss + elbo_nega_sample_loss/2.0

        total_loss = total_loss + lambda2 * inter_mi_loss
        total_loss.backward()
        exp.optimizer['mvae'].step()

    d_loader = DataLoader(exp.dataset_train, batch_size=exp.flags.batch_size,
                          shuffle=True,
                          num_workers=8, drop_last=True)

    for iteration, batch in enumerate(d_loader):
        batch_new = {}
        batch_new[0] = dict()
        batch_new[0] = {'brain':batch[0],'image':batch[1],'text':batch[2]}
        # batch_new[1] = batch[3]

        # Stage 1
        intra_mi = update_Qnet(exp, batch_new)
        exp.optimizer['Qnet'].zero_grad()
        exp.optimizer['mvae'].zero_grad()
        intra_mi.backward()
        exp.optimizer['Qnet'].step()

        # Stage 2
        basic_routine = basic_routine_epoch(exp, batch_new, epoch)
        results = basic_routine['results']
        total_loss = basic_routine['total_loss']
        klds = basic_routine['klds']
        log_probs = basic_routine['log_probs']
        intra_mi = basic_routine['intra_mi']
        elbo_loss = basic_routine['elbo_loss']

        exp.optimizer['mvae'].zero_grad()
        exp.optimizer['Qnet'].zero_grad()

        neg_batch_new = negative_sample_generator(batch_new, batch, case=1)
        basic_routine_contrast=elbo_contrast(exp, neg_batch_new, epoch)
        elbo_nega_sample_loss_case_1 = basic_routine_contrast['elbo_nega_sample_loss']

        neg_batch_new = negative_sample_generator(batch_new, batch, case=2)
        basic_routine_contrast=elbo_contrast(exp, neg_batch_new, epoch)
        elbo_nega_sample_loss_case_2 = basic_routine_contrast['elbo_nega_sample_loss']

        neg_batch_new = negative_sample_generator(batch_new, batch, case=3)
        basic_routine_contrast=elbo_contrast(exp, neg_batch_new, epoch)
        elbo_nega_sample_loss_case_3 = basic_routine_contrast['elbo_nega_sample_loss']

        neg_batch_new = negative_sample_generator(batch_new, batch, case=4)
        basic_routine_contrast=elbo_contrast(exp, neg_batch_new, epoch)
        elbo_nega_sample_loss_case_4 = basic_routine_contrast['elbo_nega_sample_loss']

        neg_batch_new = negative_sample_generator(batch_new, batch, case=5)
        basic_routine_contrast=elbo_contrast(exp, neg_batch_new, epoch)
        elbo_nega_sample_loss_case_5 = basic_routine_contrast['elbo_nega_sample_loss']

        neg_batch_new = negative_sample_generator(batch_new, batch, case=6)
        basic_routine_contrast=elbo_contrast(exp, neg_batch_new, epoch)
        elbo_nega_sample_loss_case_6 = basic_routine_contrast['elbo_nega_sample_loss']

        elbo_nega_sample_loss = elbo_nega_sample_loss_case_1 + elbo_nega_sample_loss_case_2 + elbo_nega_sample_loss_case_3 + elbo_nega_sample_loss_case_4 + elbo_nega_sample_loss_case_5 + elbo_nega_sample_loss_case_6
        inter_mi_loss = elbo_loss + elbo_nega_sample_loss/6.0
        total_loss = total_loss + lambda2 * inter_mi_loss
        total_loss.backward()
        exp.optimizer['mvae'].step()

        tb_logger.write_training_logs(results, total_loss, log_probs, klds, -inter_mi_loss)
        print('====> Epoch: {:03d} Train loss: {:.4f} ELBO: {:.4f} IntraMI: {:.4f} InterMI: {:.4f}'.format(epoch,
                                                                                                           total_loss,
                                                                                                           -elbo_loss,
                                                                                                           -intra_mi,
                                                                                                           -inter_mi_loss))

def test(epoch, exp, tb_logger):
    with torch.no_grad():
        mm_vae = exp.mm_vae
        mm_vae.eval()
        exp.mm_vae = mm_vae
        lambda2 = exp.flags.lambda2

        d_loader = DataLoader(exp.dataset_test, batch_size=200,
                            shuffle=True,
                            num_workers=8, drop_last=False)

        for iteration, batch in enumerate(d_loader):
            batch = [torch.tensor(tensor) for tensor in batch]
            batch = [tensor.cuda() for tensor in batch]
            batch_new = {}
            batch_new[0] = dict()
            batch_new[0] = {'brain': batch[0], 'image': batch[1], 'text': batch[2]}
            # batch_new[1] = batch[3]
            basic_routine = basic_routine_epoch(exp, batch_new, epoch)
            results = basic_routine['results']
            total_loss = basic_routine['total_loss']
            klds = basic_routine['klds']
            log_probs = basic_routine['log_probs']
            intra_mi = basic_routine['intra_mi']
            elbo_loss = basic_routine['elbo_loss']

            neg_batch_new = negative_sample_generator(batch_new, batch, case=1)
            basic_routine_contrast = elbo_contrast(exp, neg_batch_new, epoch)
            elbo_nega_sample_loss_case_1 = basic_routine_contrast['elbo_nega_sample_loss']

            neg_batch_new = negative_sample_generator(batch_new, batch, case=2)
            basic_routine_contrast = elbo_contrast(exp, neg_batch_new, epoch)
            elbo_nega_sample_loss_case_2 = basic_routine_contrast['elbo_nega_sample_loss']

            neg_batch_new = negative_sample_generator(batch_new, batch, case=3)
            basic_routine_contrast = elbo_contrast(exp, neg_batch_new, epoch)
            elbo_nega_sample_loss_case_3 = basic_routine_contrast['elbo_nega_sample_loss']

            neg_batch_new = negative_sample_generator(batch_new, batch, case=4)
            basic_routine_contrast = elbo_contrast(exp, neg_batch_new, epoch)
            elbo_nega_sample_loss_case_4 = basic_routine_contrast['elbo_nega_sample_loss']

            neg_batch_new = negative_sample_generator(batch_new, batch, case=5)
            basic_routine_contrast = elbo_contrast(exp, neg_batch_new, epoch)
            elbo_nega_sample_loss_case_5 = basic_routine_contrast['elbo_nega_sample_loss']

            neg_batch_new = negative_sample_generator(batch_new, batch, case=6)
            basic_routine_contrast = elbo_contrast(exp, neg_batch_new, epoch)
            elbo_nega_sample_loss_case_6 = basic_routine_contrast['elbo_nega_sample_loss']

            elbo_nega_sample_loss = elbo_nega_sample_loss_case_1 + elbo_nega_sample_loss_case_2 + elbo_nega_sample_loss_case_3 + elbo_nega_sample_loss_case_4 + elbo_nega_sample_loss_case_5 + elbo_nega_sample_loss_case_6
            inter_mi_loss = elbo_loss + elbo_nega_sample_loss / 6.0
            total_loss = total_loss + lambda2 * inter_mi_loss
            tb_logger.write_testing_logs(results, total_loss, log_probs, klds, -inter_mi_loss)
            print('====> Epoch: {:03d} Test loss: {:.4f} ELBO: {:.4f} IntraMI: {:.4f} InterMI: {:.4f}'.format(epoch,
                                                                                                           total_loss,
                                                                                                           -elbo_loss,
                                                                                                           -intra_mi,
                                                                                                           -inter_mi_loss))

def image_text_inference(exp, type):
    with torch.no_grad():
        mm_vae = exp.mm_vae
        mm_vae.eval()
        exp.mm_vae = mm_vae
        if type=='zsl':
            image_text_data = {'image': exp.dataset_test.tensors[1].cuda(),'text': exp.dataset_test.tensors[2].cuda()}
            label = exp.dataset_test.tensors[3]
            brain = exp.dataset_test.tensors[0]
        elif type=='normal':
            image_text_data = {'image': exp.dataset_val.tensors[1].cuda(),'text': exp.dataset_val.tensors[2].cuda()}
            label = exp.dataset_val.tensors[3]
            brain = exp.dataset_test.tensors[0]
        results = mm_vae(image_text_data)
        z = results['class_embeddings']
        brain_rec = mm_vae.lhoods['brain'](*mm_vae.decoders['brain'](None, z))
        return z.cpu().numpy(), label.cpu().numpy(), brain_rec.loc.cpu().numpy(), brain.cpu().numpy()

def brain_inference(exp, type):
    with torch.no_grad():
        mm_vae = exp.mm_vae
        mm_vae.eval()
        exp.mm_vae = mm_vae
        if type == 'zsl':
            data = {'brain':exp.dataset_test.tensors[0].cuda()}
            label = exp.dataset_test.tensors[3]
            image = exp.dataset_test.tensors[1]
            text = exp.dataset_test.tensors[2]
        elif type == 'normal':
            data = {'brain':exp.dataset_val.tensors[0].cuda()}
            label = exp.dataset_val.tensors[3]
            image = exp.dataset_test.tensors[1]
            text = exp.dataset_test.tensors[2]
        results = mm_vae(data)
        z = results['class_embeddings']
        image_rec = mm_vae.lhoods['image'](*mm_vae.decoders['image'](None, z))
        text_rec = mm_vae.lhoods['text'](*mm_vae.decoders['text'](None, z))
        return z.cpu().numpy(), label.cpu().numpy(), image_rec.loc.cpu().numpy(), text_rec.loc.cpu().numpy(), image.cpu().numpy(), text.cpu().numpy()

def image_inference(exp, type):
    with torch.no_grad():
        mm_vae = exp.mm_vae
        mm_vae.eval()
        exp.mm_vae = mm_vae
        if type == 'zsl':
            data = {'image':exp.dataset_test.tensors[1].cuda()}
            label = exp.dataset_test.tensors[3]
        elif type == 'normal':
            data = {'image':exp.dataset_val.tensors[1].cuda()}
            label = exp.dataset_val.tensors[3]
        results = mm_vae(data)
        z = results['class_embeddings']
        return z.cpu().numpy(), label.cpu().numpy()

def text_inference(exp, type):
    with torch.no_grad():
        mm_vae = exp.mm_vae
        mm_vae.eval()
        exp.mm_vae = mm_vae
        if type == 'zsl':
            data = {'text':exp.dataset_test.tensors[2].cuda()}
            label = exp.dataset_test.tensors[3]
        elif type == 'normal':
            data = {'text':exp.dataset_val.tensors[2].cuda()}
            label = exp.dataset_val.tensors[3]
        results = mm_vae(data)
        z = results['class_embeddings']
        return z.cpu().numpy(), label.cpu().numpy()

def run_classification_test(exp,observation, type):
    if observation=='image_text':
        z_train = []
        train_label = []
        for i in range(5):
            z, label,brain_rec,brain= image_text_inference(exp,type)
            z_train.append(z)
            train_label.append(label)
        z_train = np.vstack(z_train)
        train_label = np.vstack(train_label)

        X = np.concatenate((brain, brain_rec),axis=0)
        sgn=np.concatenate((np.ones_like(np.squeeze(label)),np.zeros_like(np.squeeze(label))),axis=0)

        # tsne = TSNE()
        # X_embedded = tsne.fit_transform(X)
        # sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=sgn, style=sgn, legend='brief', palette=palette)
        # path = './results/' + exp.flags.dataname + '_' + exp.flags.sbj + '_' + exp.flags.roi + '_' + exp.flags.aug_type + '_' + exp.flags.text_model + '_' + exp.flags.image_model.split('/')[-1] + '_' + str(exp.flags.lambda1) + '_' + str(exp.flags.lambda2) + '_' + str(exp.flags.class_dim) + '_' + exp.flags.method
        # plt.savefig(path+'_brain_vs_brain_rec.pdf',dpi=500)

    elif observation == 'image':
        z_train = []
        train_label = []
        for i in range(5):
            z, label= image_inference(exp,type)
            z_train.append(z)
            train_label.append(label)
        z_train = np.vstack(z_train)
        train_label = np.vstack(train_label)
    elif observation == 'text':
        z_train = []
        train_label = []
        for i in range(5):
            z, label= text_inference(exp,type)
            z_train.append(z)
            train_label.append(label)
        z_train = np.vstack(z_train)
        train_label = np.vstack(train_label)

    z_test, test_label, image_rec, text_rec, image, text = brain_inference(exp,type)

    classifiers = [
        SVC(gamma=0.00001, C=1.0, probability=True),
        ]
    for clf in classifiers:
        clf.fit(z_train, train_label)
        score = clf.score(z_test, test_label)
        print(f"{observation}\n"
              f"Classification report for classifier {clf}:\n"
              f"{score}\n")
        probas = clf.predict_proba(z_test)
        top_acc = top_k_accuracy_score(test_label, probas, k=5)
        print(f"{observation}\n"
              f"Classification Top 5 Acc for classifier {clf}:\n"
              f"{top_acc}\n")

    return score, top_acc

def create_csv(path,top1,top5):
    with open(path,'w') as f:
        csv_writer = csv.writer(f)
        head = ["top1","top5"]
        csv_writer.writerow(head)

def write_csv(path,top1,top5):
    with open(path, 'a+') as f:
        csv_writer = csv.writer(f)
        row = []
        row.append(top1)
        row.append(top5)
        csv_writer.writerow(row)

def run_epochs_trimodal(exp):
    # initialize summary writer
    # 输入: exp是一个BrainImageText类的对象
    writer = SummaryWriter(exp.flags.dir_logs)  # PyTorch可视化工具
    tb_logger = TBLogger(exp.flags.str_experiment, writer)  # TBLogger类对象
    str_flags = utils.save_and_log_flags(exp.flags)  # 将flags保存到指定的rar文件, 并获得一个字符串形式的flags表示
    tb_logger.writer.add_text('FLAGS', str_flags, 0)  # 将flags输出到TensorBoard的日志文件中
    lr_list = []
    print('training epochs progress:')
    for epoch in range(exp.flags.start_epoch, exp.flags.end_epoch):  # 默认: for epoch in range(0, 100):
        utils.printProgressBar(epoch, exp.flags.end_epoch)  # 画进度条
        # one epoch of training and testing
        # 在每个epoch更新, 每过一段时间自动下调学习率
        exp.scheduler['Qnet'].step()
        exp.scheduler['mvae'].step()
        ###
        lr_list.append(exp.optimizer['Qnet'].state_dict()['param_groups'][0]['lr'])  # 将Qnet的学习率存入lr_list
        train_aug(epoch, exp, tb_logger)  # 用增强过的数据来训练
        train(epoch, exp, tb_logger)      # 训练
        test(epoch, exp, tb_logger)       # 测试
        # save checkpoints after every 1 epochs
        if (epoch + 1) % 100 == 0 or (epoch + 1) == exp.flags.end_epoch:
            dir_network_epoch = os.path.join(exp.flags.dir_checkpoints, str(epoch).zfill(4))
            if not os.path.exists(dir_network_epoch):
                os.makedirs(dir_network_epoch)
            exp.mm_vae.save_networks()
            torch.save(exp.mm_vae.state_dict(),
                       os.path.join(dir_network_epoch, exp.flags.mm_vae_save))

        print('lr = ',lr_list[-1])
    # plt.plot(range(exp.flags.end_epoch), lr_list, color='r')
    # plt.show()

    if exp.flags.test_type=='normal':
        top1, top5 = run_classification_test(exp,'image_text', 'normal')
    elif exp.flags.test_type=='zsl':
        top1, top5 = run_classification_test(exp, 'image_text', 'zsl')
        path = './results/'+exp.flags.dataname+'_'+exp.flags.sbj+'_'+exp.flags.roi+'_'+exp.flags.aug_type+'_'+exp.flags.text_model+'_'+exp.flags.image_model.split('/')[-1]+'_'+str(exp.flags.lambda1)+'_'+str(exp.flags.lambda2)+'_'+'_'+str(exp.flags.class_dim)+'_'+exp.flags.method+'_image_text.csv'
        create_csv(path, top1, top5)
        write_csv(path, top1, top5)
