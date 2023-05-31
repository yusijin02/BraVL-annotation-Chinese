import os
import torch

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)                               当前的循环次数
        total       - Required  : total iterations (Int)                                总共的循环次数
        prefix      - Optional  : prefix string (Str)                                   进度条的前缀字符
        suffix      - Optional  : suffix string (Str)                                   进度条的后缀字符
        decimals    - Optional  : positive number of decimals in percent complete (Int) 百分比的显示位数
        length      - Optional  : character length of bar (Int)                         进度条的总长度
        fill        - Optional  : bar fill character (Str)                              进度条的填充字符
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))  # 进度条百分比
    filledLength = int(length * iteration // total)                                     # 填充的长度
    bar = fill * filledLength + '-' * (length - filledLength)                           # 整个进度条
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')             # 输出进度条, end='\r'保证下一行将覆盖上一行的进度条
    # Print New Line on Complete
    if iteration == total:                                                              # 如果已经完成, 输出新一行, 让进度条不被覆盖
        print()

def get_likelihood(str):
    if str == 'laplace':
        pz = dist.Laplace
    elif str == 'bernoulli':
        pz = dist.Bernoulli
    elif str == 'normal':
        pz = dist.Normal
    elif str == 'categorical':
        pz = dist.OneHotCategorical
    else:
        print('likelihood not implemented')
        pz = None
    return pz


def reweight_weights(w):
    # 将原始的权重归一化, 使其是一个概率分布
    w = w / w.sum()
    return w


def mixture_component_selection(flags, mus, logvars, w_modalities=None):
    # 将不同模态的信息混合
    # if not defined, take pre-defined weights
    # 输入:
    # flags状态字典
    # mus和logvars是(模态数, batch_size, 32)形状的
    # weights是(模态数, )形状的
    num_components = mus.shape[0]  # 模态数
    num_samples = mus.shape[1]     # batch_size
    if w_modalities is None:
        w_modalities = torch.Tensor(flags.alpha_modalities).to(flags.device)  # 使用默认的权重
        # [权重, 脑模态的权重, 视觉模态的权重, 文本模态的权重]

    # 下面是获得三个模态的样本起始和结束index
    idx_start = []
    idx_end = []
    for k in range(0, num_components):
        if k == 0:
            i_start = 0  # 对于第一个模态, index起始点为0
        else:
            i_start = int(idx_end[k-1])  # 对于后面的模态, index的起始点为上一个模态的结束点
        if k == w_modalities.shape[0]-1:
            i_end = num_samples  # 对于最后一个模态, index的结束点为num_samples, 也就是batch_size
        else:
            i_end = i_start + int(torch.floor(num_samples*w_modalities[k]))  # 对于前面的模态, 以权重为比例挑选样本
        idx_start.append(i_start)
        idx_end.append(i_end)
    idx_end[-1] = num_samples

    mu_sel = torch.cat([mus[k, idx_start[k]:idx_end[k], :] for k in range(w_modalities.shape[0])])
    logvar_sel = torch.cat([logvars[k, idx_start[k]:idx_end[k], :] for k in range(w_modalities.shape[0])])
    # 返回每个模态被选中的均值和对数方差
    return [mu_sel, logvar_sel]  # 两个的形状都是(模态数, w * batch_size, 32)


def calc_elbo(exp, modality, recs, klds):
    flags = exp.flags
    mods = exp.modalities
    s_weights = exp.style_weights
    r_weights = exp.rec_weights
    kld_content = klds['content']
    if modality == 'joint':
        w_style_kld = 0.0
        w_rec = 0.0
        klds_style = klds['style']
        for k, m_key in enumerate(mods.keys()):
                w_style_kld += s_weights[m_key] * klds_style[m_key]
                w_rec += r_weights[m_key] * recs[m_key]
        kld_style = w_style_kld
        rec_error = w_rec
    else:
        beta_style_mod = s_weights[modality]
        #rec_weight_mod = r_weights[modality]
        rec_weight_mod = 1.0
        kld_style = beta_style_mod * klds['style'][modality]
        rec_error = rec_weight_mod * recs[modality]
    div = flags.beta_content * kld_content + flags.beta_style * kld_style
    elbo = rec_error + flags.beta * div
    return elbo


def save_and_log_flags(flags):
    #filename_flags = os.path.join(flags.dir_experiment_run, 'flags.json')
    #with open(filename_flags, 'w') as f:
    #    json.dump(flags.__dict__, f, indent=2, sort_keys=True)

    filename_flags_rar = os.path.join(flags.dir_experiment_run, 'flags.rar')
    torch.save(flags, filename_flags_rar)  # 将flags保存到指定的rar文件
    str_args = ''
    for k, key in enumerate(sorted(flags.__dict__.keys())):
        str_args = str_args + '\n' + key + ': ' + str(flags.__dict__[key])
    return str_args  # flags的字符串表示


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Unflatten(torch.nn.Module):
    def __init__(self, ndims):
        super(Unflatten, self).__init__()
        self.ndims = ndims

    def forward(self, x):
        return x.view(x.size(0), *self.ndims)
