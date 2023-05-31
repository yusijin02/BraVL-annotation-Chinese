
import os
from datetime import datetime

def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    # else:
    #     shutil.rmtree(dir_name, ignore_errors=True)
    #     os.makedirs(dir_name)


def get_str_experiments(flags):
    # 获得当前环境路径
    # 输入: 状态变量字典
    # 输出: str, 当前环境路径
    # Caller: create_dir_structure(flags, train=True)
    dateTimeObj = datetime.now()  # 获取当前时间
    dateStr = dateTimeObj.strftime("%Y_%m_%d")  # 将当前时间转为: 年_月_日 格式
    str_experiments = flags.dataset + '_' + dateStr  #  flags.dataset = "DIR-Wiki"
    # str_experiments = "DIR-Wiki_年_月_日"
    return str_experiments

def create_dir_structure(flags, train=True):
    # 更新flag, 构建存放本次训练输出的目录
    # 输入: 状态变量字典, 是否训练(默认是)
    # 输出: 更新后的状态变量字典
    # Caller: /main_trimodal.py
    if train:
        # 训练模式
        str_experiments = get_str_experiments(flags)  # 获得当前环境路径
        flags.dir_experiment_run = os.path.join(flags.dir_experiment, str_experiments)  # 日志路径
        flags.str_experiment = str_experiments
        # flags.str_experiment = './logs/DIR-Wiki_年_月_日'
    else:
        # 预测模式
        flags.dir_experiment_run = flags.dir_experiment

    print(flags.dir_experiment_run)
    if train:
        create_dir(flags.dir_experiment_run)
        # 如果是训练模式, 则创建一个目录用于存放这次训练的输出

    flags.dir_checkpoints = os.path.join(flags.dir_experiment_run, 'checkpoints')
    # flags.dir_checkpoints = './logs/DIR-Wiki_年_月_日/checkpoints'
    if train:
        create_dir(flags.dir_checkpoints)
        # 如果是训练模式, 则创建一个目录用于存放这次训练的检查点

    flags.dir_logs = os.path.join(flags.dir_experiment_run, 'logs')
    # 日志目录
    # flags.dir_logs = './logs/DIR-Wiki_年_月_日/logs'
    if train:
        create_dir(flags.dir_logs)
        # 如果是训练模式 则创建一个目录用于存放本次训练的日志
    print(flags.dir_logs)
    return flags
