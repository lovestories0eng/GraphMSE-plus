import os
import pickle as pkl
import networkx as nx
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA


class HeteData():
    def __init__(self, data_root="data", dataset="DBLP", train_percent=None, shuffle=False):
        """
        The datasets include DBLP, IMDB, and ACM.
        Edge_0, edge_1, edge_2, and edge_3 are lists of heterogeneous edges, while
        Edgelist is a list of homogeneous edges generated by discarding edge-type information.
        """
        self.dataset = dataset
        self.path = os.path.join(data_root, dataset)
        self.type_num = 3

        if os.path.exists(self.path):
            print("Dataset root: {}".format(self.path))
        else:
            raise FileNotFoundError
        if shuffle == False:
            feature_path = self.path + '/node_features.pkl'
        else:
            print("Shuffled feature loaded.")
            feature_path = self.path + '/shuffled_features.pkl'
        with open(feature_path, 'rb') as f:
            self.x = pkl.load(f)
        self.hete_graph = []
        for i in range(4):
            self.hete_graph.append(
                nx.read_edgelist(self.path + "/edge_" + str(i), create_using=nx.DiGraph(), nodetype=int))
        '''
        Define meta-path according to the dataset
        The next_type dict ensure the correctness of meta-path connection
        '''
        if self.dataset == "DBLP":
            self.node_dict = {"P": 0, "C": 1, "A": 2}
            self.edge_type = {0: "PA", 1: "AP", 2: "PC", 3: "CP"}
            self.next_type = {0: [1], 1: [0, 2], 2: [3], 3: [0, 2]}
        if self.dataset == "IMDB":
            self.node_dict = {"D": 0, "A": 1, "M": 2}
            self.edge_type = {0: "MD", 1: "DM", 2: "MA", 3: "AM"}
            self.next_type = {0: [1], 1: [0, 2], 2: [3], 3: [0, 2]}
        if self.dataset == "ACM":
            self.node_dict = {"A": 0, "S": 1, "P": 2}
            self.edge_type = {0: "PA", 1: "AP", 2: "PS", 3: "SP"}
            self.next_type = {0: [1], 1: [0, 2], 2: [3], 3: [0, 2]}

        with open(self.path + '/node_type', 'rb') as f:
            self.node_type = pkl.load(f)

        self.homo_graph = nx.read_edgelist(self.path + "/edgelist", nodetype=int)
        label_path = self.path + '/labels'
        if train_percent != None:
            label_path += '_'
            label_path += str(train_percent)
        label_path += '.pkl'
        print("Label file:", label_path)

        with open(label_path, 'rb') as f:
            labels = pkl.load(f)
        self.train_list = np.array(labels[0], dtype=np.int64)
        self.val_list = np.array(labels[1], dtype=np.int64)
        self.test_list = np.array(labels[2], dtype=np.int64)

    def get_dict_of_list(self):
        result = []
        for g in self.hete_graph:
            result.append(nx.to_dict_of_lists(g))
        return result

    def get_metapath_dict(self):
        return self.next_type

    def get_metapath_name(self):
        return self.edge_type



