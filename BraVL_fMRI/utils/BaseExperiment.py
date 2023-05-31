import os
from abc import ABC, abstractmethod
from itertools import chain, combinations

class BaseExperiment(ABC):
    def __init__(self, flags):
        self.flags = flags
        self.name = flags.dataset

        self.modalities = None
        self.num_modalities = None
        self.subsets = None
        self.dataset_train = None
        self.dataset_test = None
        self.Q1, self.Q2, self.Q3 = None,None,None
        self.mm_vae = None
        self.clfs = None
        self.optimizer = None
        self.rec_weights = None
        self.style_weights = None

        self.test_samples = None
        self.paths_fid = None


    @abstractmethod
    def set_model(self):
        pass

    @abstractmethod
    def set_Qmodel(self):
        pass

    @abstractmethod
    def set_modalities(self):
        pass

    @abstractmethod
    def set_dataset(self):
        pass


    @abstractmethod
    def set_optimizer(self):
        pass

    @abstractmethod
    def set_rec_weights(self):
        pass

    @abstractmethod
    def set_style_weights(self):
        pass


    def set_subsets(self):
        num_mods = len(list(self.modalities.keys()))
        # 模态数
        # 一般这里是3

        """
        powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3)
        (1,2,3)
        """
        xs = list(self.modalities)
        # xs = ["brain", "image", "text"]
        # note we return an iterator rather than a list
        subsets_list = chain.from_iterable(combinations(xs, n) for n in range(len(xs)+1))
        # combinations(xs, n) 返回一个list of元组, 每个元组是xs这个列表选n个的结果
        # n遍历0到3: 分别取0个元素, 1个元素, 2个元素, 3个元素
        # chain.from_iterable() 将多个列表展开成一个列表
        # subsets_list = powerset(["brain", "image", "text"])
        # = [(), ('brain',), ('image',), ('text',), ('brain', 'image'), ('brain', 'text'), ('image', 'text'), ('brain', 'image', 'text')]
        subsets = dict()
        for k, mod_names in enumerate(subsets_list):
            mods = []
            for l, mod_name in enumerate(sorted(mod_names)):
                mods.append(self.modalities[mod_name])
            key = '_'.join(sorted(mod_names))
            subsets[key] = mods
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
        return subsets
