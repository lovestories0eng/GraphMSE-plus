import torch
from threading import Lock

import numpy as np

class GlobalStore:
    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        # 确保线程安全
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
            self, 
            metapath_embedding_store=None,
            metapath_cnt=None,
            dataset=None,
            sample_times=None,
            sample_num=None
        ):
        if not hasattr(self, 'initialized'):
            self.initialized = True

            self.dataset = ""
            self.sample_times = 0
            self.sample_num = 0

            self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.metapath_embedding_store = metapath_embedding_store
            self.metapath_cnt = metapath_cnt

            self.train_loss = []
            self.val_loss = []

            self.downstream_losses = []

            self.micro_f1 = []
            self.macro_f1 = []

            # key: metapath type, value: [cosine similarity]
            self.sim_dict = {}
            # key: metapath type, value: [euclidean dist]
            self.euclidean_dict = {}

    @classmethod
    def get_instance(cls, metapath_embedding_store=None, metapath_cnt=None):
        if cls._instance is None:
            cls._instance = cls(metapath_embedding_store, metapath_cnt)
        return cls._instance
    
    def save_data(self):
        suffix = "__" + self.dataset + "_" + str(self.sample_times) + "_" + str(self.sample_num)

        # 保存 train_loss
        np.savetxt("result/train_loss" + suffix + ".txt", self.train_loss)

        # 保存 val_loss
        np.savetxt("result/val_loss" + suffix + ".txt", self.train_loss)

        # 保存 downstream_loss
        np.savetxt("result/downstream_losses" + suffix + ".txt", self.downstream_losses)

        f1_score = []
        f1_score.append(self.micro_f1)
        f1_score.append(self.macro_f1)
        # 保存 micro_f1, macro_f1
        np.savetxt("result/f1_score" + suffix + ".txt", np.array(f1_score))

        # 保存所有的 metapath embedding
        for i in range(len(self.metapath_embedding_store)):
            cur_list = self.metapath_embedding_store[i]
            cur_arr = np.array([e.cpu().detach().numpy() for e in cur_list])
            np.savetxt("result/metapath_embedding_" + "round" + str(i) + suffix + ".txt", cur_arr)

        # 保存余弦相似度
        for i in range(self.metapath_cnt):
            for j in range(self.metapath_cnt):
                np.savetxt("result/cosine_similarity" + "_" + str(i) + "_" + str(j) + suffix + ".txt", np.array(self.sim_dict[(i, j)]))

        # 保存欧几里得距离
        for i in range(self.metapath_cnt):
            for j in range(self.metapath_cnt):
                np.savetxt("result/euclidean_distance" + "_" + str(i) + "_" + str(j) + suffix + ".txt", np.array(self.euclidean_dict[(i, j)]))

    def update_sim_dict(self, metapath_label, metapath_embedding_label, cosine_similarity):
        if (metapath_label, metapath_embedding_label) in self.sim_dict:
            self.sim_dict[(metapath_label, metapath_embedding_label)].append(cosine_similarity)
        else:
            self.sim_dict[(metapath_label, metapath_embedding_label)] = [cosine_similarity]
    
    def update_euclidean_dict(self, metapath_label, metapath_embedding_label, euclidean_dist):
        if (metapath_label, metapath_embedding_label) in self.euclidean_dict:
            self.euclidean_dict[(metapath_label, metapath_embedding_label)].append(euclidean_dist)
        else:
            self.euclidean_dict[(metapath_label, metapath_embedding_label)] = [euclidean_dist]

    def f1_score_append(self, to_append_micro, to_append_macro):
        self.micro_f1.append(to_append_micro)
        self.macro_f1.append(to_append_macro)

    def train_loss_append(self, to_append):
        self.train_loss.append(to_append)

    def val_loss_append(self, to_append):
        self.val_loss.append(to_append)

    def downstream_losses_append(self, to_append):
        self.downstream_losses.append(to_append)

    def update(self, params: dict):
        # 遍历字典，把存在的属性都更新
        for key, value in params.items():
            if hasattr(self, key):  # 确认这个属性存在
                setattr(self, key, value)
            else:
                print(f"Warning: {key} is not a valid attribute of GlobalStore")

    def cal_dist(self):
        if self.metapath_embedding_store is None or self.metapath_cnt is None:
            raise ValueError("GlobalStore 未正确初始化")
        
        length = len(self.metapath_embedding_store)
        dist_sum = 0
        if length > 1:
            last_store = self.metapath_embedding_store[-2]
            cur_store = self.metapath_embedding_store[-1]
            for i in range(self.metapath_cnt):
                dist_sum += torch.dist(last_store[i], cur_store[i])
        return dist_sum
