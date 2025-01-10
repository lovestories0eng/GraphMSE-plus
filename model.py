import torch
import torch.nn as nn
import torch.nn.init as init
import random
import sys


class GraphMSE(nn.Module):
    def __init__(self, metapath_list, input_dim, pre_embed_dim, select_dim, mlp_settings, endnode_test=False, mean_test=False):
        super(GraphMSE, self).__init__()
        assert 'layer_list' in mlp_settings.keys()
        assert 'dropout_list' in mlp_settings.keys()
        assert 'activation' in mlp_settings.keys()
        assert mlp_settings['activation'] in ['sigmoid', 'relu', 'tanh']
        self.mean_test = mean_test
        self.metapath = metapath_list
        self.node_type = len(metapath_list)
        # input_dim: 输入维度。
        self.input_dim = input_dim
        # pre_embed_dim: 中间表示（embedding）的维度。
        self.pre_embed_dim = pre_embed_dim
        # select_dim: 分类层输出维度。
        self.select_dim = select_dim
        # metapath_mlp: 每种元路径对应一个 MLP 模型，用于处理输入数据。
        self.metapath_mlp = nn.ModuleDict()
        # mlp_settings: 多层感知机（MLP）配置，包括层数、激活函数、dropout等。
        self.mlp_settings = mlp_settings
        self.type_weight = nn.ParameterDict()
        self.type_trained = {}
        self.classify_layer = nn.Linear(pre_embed_dim, select_dim)
        self.weight = {}
        self.attn = nn.ModuleDict()
        # metapath2index: 元路径到索引的映射。
        self.metapath2index = {}

        # 遍历每一个类型的节点
        for node_type in metapath_list:
            self.metapath_mlp[node_type] = nn.ModuleDict()
            self.type_weight[node_type] = nn.Parameter(torch.Tensor(input_dim, pre_embed_dim))
            init.kaiming_uniform_(self.type_weight[node_type])
            self.metapath2index[node_type] = {}
            metapath_index = 0
            # 遍历每一个节点所对应的相应元路径
            for metapath in metapath_list[node_type]:
                # 对每一个节点所对应的元路径进行编号
                self.metapath2index[node_type][metapath] = metapath_index
                metapath_index += 1

                '''
                    检查是否需要构建 MLP
                    如果为空，不构建多层网络，只使用一个简单的线性层 Linear。
                    如果不为空，按照 layer_list 构建多层网络。

                    endnode_test 为 True:
                        直接将输入特征维度 input_dim 映射到预嵌入维度 pre_embed_dim。
                        用于处理特殊情况：只需对单一节点特征进行变换。

                    endnode_test 为 False:
                        输入维度是 (len(metapath) - 1) * input_dim:
                        说明元路径的长度大于 1(包含多个节点),需要将多个节点的特征拼接作为输入。
                        输出维度仍为 pre_embed_dim。

                    整体结构:
                        无论哪种情况，均返回一个简单的 nn.Sequential 容器，仅包含一个线性层。
                '''
                if len(self.mlp_settings['layer_list']) == 0:
                    if endnode_test:
                        self.metapath_mlp[node_type][metapath] = nn.Sequential(
                            nn.Linear(input_dim, pre_embed_dim),
                        )
                    else:
                        self.metapath_mlp[node_type][metapath] = nn.Sequential(
                            nn.Linear((len(metapath) - 1) * input_dim, pre_embed_dim),
                        )
                # 构建多层感知机
                else:
                    layer_dim = self.mlp_settings['layer_list']
                    if endnode_test:
                        self.metapath_mlp[node_type][metapath] = nn.Sequential(
                            nn.Linear(input_dim, layer_dim[0]), )
                    else:
                        self.metapath_mlp[node_type][metapath] = nn.Sequential(
                            nn.Linear((len(metapath) - 1) * input_dim, layer_dim[0]), )
                    # 添加激活函数
                    self.metapath_mlp[node_type][metapath].add_module('1',
                                                                      self.mlp_activation(
                                                                          mlp_settings['activation']))
                    '''
                        遍历 layer_dim 构建隐藏层，依次连接：
                        线性层:
                            输入维度 layer_dim[i]，输出维度 layer_dim[i + 1]。
                            连接不同层的特征维度。
                        Dropout 层：
                            如果 dropout_list 中有值，则插入 Dropout 层，避免过拟合。
                        激活函数：
                            添加用户指定的激活函数(如 ReLU)。
                        模块编号：
                            每添加一个模块，更新编号 cur,以保持唯一性。
                    '''
                    cur = 2
                    for i in range(len(layer_dim) - 1):
                        self.metapath_mlp[node_type][metapath].add_module(str(cur),
                                                                          nn.Linear(layer_dim[i], layer_dim[i + 1]))
                        cur += 1
                        if i < len(mlp_settings['dropout_list']):
                            self.metapath_mlp[node_type][metapath].add_module(str(cur), nn.Dropout(
                                mlp_settings['dropout_list'][i]))
                            cur += 1
                        self.metapath_mlp[node_type][metapath].add_module(str(cur), self.mlp_activation(
                            mlp_settings['activation']))
                        cur += 1
                    #  添加最后一层线性变换
                    self.metapath_mlp[node_type][metapath].add_module(str(cur),
                                                                      nn.Linear(layer_dim[-1], pre_embed_dim))

            '''
                基于注意力的权重计算器，用于对元路径特征进行加权
                模块的整体顺序为：
                    Tanh 激活函数：
                    对输入特征进行非线性变换。
                    第一线性层：
                    执行特征变换（保持维度）。
                    第二线性层：
                    将特征压缩到标量。
                    LeakyReLU 激活函数：
                    提供非线性映射，避免梯度消失。

                这个注意力机制模块的作用是根据输入特征生成每条元路径的重要性得分：

                输入：
                    特征维度为 2 * pre_embed_dim,通常由中心节点嵌入和元路径嵌入拼接而成。
                输出：
                    一个标量(1),表示当前元路径的重要性。
            '''
            self.attn[node_type] = nn.Sequential(
                nn.Tanh(),
                nn.Linear(2 * pre_embed_dim, 2 * pre_embed_dim, bias=False),
                nn.Linear(2 * pre_embed_dim, 1, bias=False),
                nn.LeakyReLU(0.1),
            )

    def mlp_activation(self, type):
        if type == 'sigmoid':
            return nn.Sigmoid()
        if type == 'tanh':
            return nn.Tanh()
        if type == 'relu':
            return nn.ReLU()

    def metapath_aggregate(self, tensor):
        if self.mean_test:
            return tensor.mean(dim=0, keepdim=True)
        return tensor.sum(dim=0, keepdim=True)

    def forward(self, rows_dict, feature_dict, start_select=False):
        center_node = {}
        injective = {}
        pre_embed = {}
        weight = {}
        GAN_input = {}
        for type in feature_dict:
            self.type_trained[type] = True
            injective[type] = {}
            for metapath in feature_dict[type]:
                if len(metapath) == 1:
                    center_node[type] = feature_dict[type][metapath].mm(self.type_weight[type])
                    continue
                # 将相同种类metapth做映射，再相加
                x_dim = self.pre_embed_dim
                injective_mapping = self.metapath_mlp[type][metapath](feature_dict[type][metapath])
                features_per_node = list(injective_mapping.split(rows_dict[type][metapath]))
                for i in range(len(rows_dict[type][metapath])):
                    if rows_dict[type][metapath][i] == 0:
                        features_per_node[i] = torch.zeros([1, x_dim]).to(
                            "cuda" if next(self.parameters()).is_cuda else "cpu")
                injective[type][metapath] = torch.cat(list(map(self.metapath_aggregate, features_per_node)), dim=0)
            GAN_input[type] = injective[type]
            weight[type] = torch.empty(len(self.metapath2index[type])).to(
                "cuda" if next(self.parameters()).is_cuda else "cpu")
            for metapath in injective[type]:
                if start_select:
                    weight[type][self.metapath2index[type][metapath]] = self.attn[type](torch.cat(
                        [center_node[type], injective[type][metapath]], dim=1)).mean()

            if start_select:
                weight[type] = weight[type].softmax(dim=0)
                for metapath in injective[type]:
                    injective[type][metapath] *= weight[type][self.metapath2index[type][metapath]]
                self.weight[type] = weight[type]

            embed = torch.stack(list(injective[type].values()), dim=2).mean(dim=2)
            pre_embed[type] = self.classify_layer(center_node[type] + embed)
        return pre_embed, GAN_input

    def show_metapath_importance(self, type=None):
        if type != None:
            weight = self.weight[type]
            for metapath in self.metapath2index[type]:
                print("Meta-path: ", metapath, "\tWeight: ", weight[self.metapath2index[type][metapath]].item())
            return
        for type in self.type_trained:
            weight = self.weight[type]
            for metapath in self.metapath2index[type]:
                print("type: ", type, "\tMeta-path: ", metapath, "\tWeight: ",
                      weight[self.metapath2index[type][metapath]].item())

    def save_metapath_importance(self, file):
        for type in self.type_trained:
            weight = self.weight[type]
            for metapath in self.metapath2index[type]:
                file.write(metapath + "," + str(weight[self.metapath2index[type][metapath]].item()) + ",")
            file.write('end\n')



class Discriminator(nn.Module):
    def __init__(self, info_section, type_num):
        super(Discriminator, self).__init__()
        self.info_section = info_section
        self.type_num = type_num
        self.classifier = nn.ModuleList()
        for i in range(type_num):
            self.classifier.append(
                nn.Sequential(
                    nn.Linear(info_section, 1),
                    nn.Sigmoid(),
                )
            )

    def forward(self, GAN_input, Shuffle=False):
        result = {}
        for metapath in GAN_input:
            result[metapath] = []
            section = GAN_input[metapath].split([self.info_section] * self.type_num, dim=1)
            for i in range(self.type_num):
                if not Shuffle:
                    result[metapath].append(self.classifier[i](section[i]))
                else:
                    choice = list(range(self.type_num))
                    choice.remove(i)
                    result[metapath].append(self.classifier[i](section[random.choice(choice)]))
            result[metapath] = torch.cat(result[metapath], dim=1)
        return result
