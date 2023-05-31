import os

import torch
import torch.nn as nn

from utils import utils
from utils.BaseMMVae import BaseMMVae


class VAEtrimodal(BaseMMVae, nn.Module):
    def __init__(self, flags, modalities, subsets):
        super().__init__(flags, modalities, subsets)

class VAEbimodal(BaseMMVae, nn.Module):
    def __init__(self, flags, modalities, subsets):
        super().__init__(flags, modalities, subsets)

# 上面的subsets都是一个类似下面的字典:
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



