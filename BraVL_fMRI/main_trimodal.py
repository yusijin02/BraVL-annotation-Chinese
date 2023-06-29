import sys
import os
import json
import torch
import torch.nn as nn
from run_epochs_trimodal import run_epochs_trimodal
from utils.filehandling import create_dir_structure
from brain_image_text.flags import parser
from brain_image_text.experiment import BrainImageText
# torch.set_default_tensor_type(torch.DoubleTensor)
if __name__ == '__main__':
    FLAGS = parser.parse_args()
    # use_cuda = torch.cuda.is_available()
    # FLAGS.device = torch.device('cuda:0' if use_cuda else 'cpu')
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.cuda_visable_device
    FLAGS.device_ids = FLAGS.cuda_visable_device.split(",")
    FLAGS.device_ids = range(len(FLAGS.device_ids))

    print(FLAGS.cuda_visable_device)
    print(FLAGS.device_ids)

    if FLAGS.method == 'poe':
        FLAGS.modality_poe=True
    elif FLAGS.method == 'moe':
        FLAGS.modality_moe=True
    elif FLAGS.method == 'jsd':
        FLAGS.modality_jsd=True
    elif FLAGS.method == 'joint_elbo':
        # 默认
        FLAGS.joint_elbo=True
    else:
        print('method implemented...exit!')
        sys.exit()
    print(FLAGS.modality_poe)
    print(FLAGS.modality_moe)
    print(FLAGS.modality_jsd)
    print(FLAGS.joint_elbo)

    FLAGS.alpha_modalities = [FLAGS.div_weight_uniform_content, FLAGS.div_weight_m1_content,
                              FLAGS.div_weight_m2_content, FLAGS.div_weight_m3_content]
    # 四个alpha, [权重正则, 脑模态的权重正则, 视觉模态的权重正则, 文本模态的权重正则]
    # 这四个alpha之和应该为1, 如果输入的和不是1, 将在 ./utils/BaseMMVae.py 里保证

    FLAGS = create_dir_structure(FLAGS)  # /utils/filehandling.py
    # 创建目录并更新FLAGS, 用于存放本次实验的输出

    alphabet_path = os.path.join(os.getcwd(), 'alphabet.json')
    # os.getcwd() 返回当前这个py文件所在的目录, 相当于"."
    # alphabet_path = "./alphabet.json"
    with open(alphabet_path) as alphabet_file:
        alphabet = str(''.join(json.load(alphabet_file)))
    # alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'"\/|_@#$%^&*~`+-=<>()[]{} \n"
    # alphabet是一个包含了所用英文字母, 数字, 常用符号的字符串

    mst = BrainImageText(FLAGS, alphabet)
    mst.set_optimizer()


    total_params = sum(p.numel() for p in mst.mm_vae.parameters())
    print('num parameters model: ' + str(total_params))
    run_epochs_trimodal(mst)  # 扫100个epoch

    ##########################################################
    # 命令行参数
    ##########################################################
    #### 训练 TRAINING
    # batch_size, 批量大小, int, 512
    # initial_learning_rate, 初始学习率, float, 0.0001
    # beta_1, Adam的beta_1, float, 0.9
    # beta_2, Adam的beta_2, float, 0.999
    # start_epoch, 开始的epoch, int, 0
    # end_epoch, 结束的epoch, int, 100

    #### 数据格式 DATA DEPENDENT
    # class_dim, 共同隐藏factor(论文里Fig.4的z)的维度, int, 32
    # mm_vae_save, 双模态模型的保存, str, 'mm_vae'
    # load_saved, 是否加载已保存的模型, bool, False

    #### 目录 DIRECTORIES
    # dir_experiment, 保存日志的目录, str, './logs'
    # dataname, 数据集名字, str, 'DIR-Wiki'
    # sbj, fMRI的被试对象, str, 'sub-03'
    # roi, 关注的区域(论文里的Region of Interesting), str, 'LVC_HVC_IT'
    # text_model, 文本的embedding模型, str, 'GPTNeo'
    # image_model, 图像的embedding模型, str, 'pytorch/repvgg_b3g4'
    # stability_ratio, 看起来这个东西没啥卵用 就是文件命名而已, str, ''
    # test_type, 测试的方式, str, 'zsl'
    # aug_type, 数据增强的方式, str, 'image_text'  # [no_aug, image_text, image_only, text_only]
    # method, 训练模型的方法, str, 'joint_elbo'
    # modality_jsd, ??????????, bool, False
    # modality_poe, 是否使用POE, bool, False
    # modality_moe, 是否使用MOE, bool, False
    # joint_elbo, 是否使用joint的ELBO, bool, False
    # poe_unimodal_elbos, 是否使用POE+单模态ELBO, bool, True
    # factorized_representation, ????????, bool, False

    #### 损失项权重 LOSS TERM WEIGHTS
    # beta, 权重惩罚项的默认初始权重, float, 0.0
    # beta_style, ????????, float, 1.0
    # beta_content, ?????????,float, 1.0
    # lambda1, 同模态互信息惩罚项的默认权重, float, 0.001
    # lambda2, 不同模态互信息惩罚项的默认权重, float, 0.001

    #### 数据维度
    # m1_dim, 脑数据的维度, int, /utils/BaseFlags.py中读取的train_brain数据的维度
    # m2_dim, 图像的维度, int, /utils/BaseFlags.py中读取的train_image数据的维度
    # m3_dim, 文本的维度, int, /utils/BaseFlags.py中读取的train_text数据的维度

    # dataset, 数据集的名字, str, Brain_Image_Text
    # style_m1_dim, 脑模态???????, int, 0
    # style_m2_dim, 图像模态???????, int, 0
    # style_m3_dim, 文本模态???????, int, 0

    # num_hidden_layers, 隐藏层个数, int, 2
    # likelihood_m1, 脑模态输出的分布, str, 'laplace'
    # likelihood_m2, 图像模态输出的分布, str, 'laplace'
    # likelihood_m3, 文本模态输出的分布, str, 'laplace'

    # 三个模态的style权重:
    # beta_m1_style, 脑模态的style权重, float, 1.0
    # beta_m2_style, 图像模态的style权重, float, 1.0
    # beta_m3_style, 文本模态的style权重, float, 1.0

    # 三个模态重构损失的权重:
    # beta_m1_rec, 脑模态重构损失权重, float, 1.0
    # beta_m2_rec, 图像模态重构损失权重, float, 1.0
    # beta_m3_rec, 文本模态重构损失权重, float, 1.0

    # div_weight_m1_content, 脑模态??????????????, float, 0.25
    # div_weight_m2_content, 图像模态??????????????, float, 0.25
    # div_weight_m3_content, 文本模态??????????????, float, 0.25
    # div_weight_uniform_content, ?????????, float, 0.25



