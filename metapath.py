import numpy as np
import torch
from data import HeteData
import random


def enum_longest_metapath_index(name_dict, type_dict, length):
    # 枚举最长的metapath编号列表
    # 即不保存metapath的任何前缀
    '''
      name_dict = {
        0: ("P", "A"),  # 类型 0 的名称是 "P"，连接类型是 "A"
        1: ("A", "P"),  # 类型 1 的名称是 "A"，连接类型是 "P"
        2: ("P", "C"),  # 类型 2 的名称是 "P"，连接类型是 "C"
        3: ("C", "P")   # 类型 3 的名称是 "C"，连接类型是 "P"
      }

      type_dict = {
        0: [1],        # 类型 0 可以连接到类型 1
        1: [0, 2],     # 类型 1 可以连接到类型 0 或类型 2
        2: [3],        # 类型 2 可以连接到类型 3
        3: [0, 2]      # 类型 3 可以连接到类型 0 或类型 2
      }

      length = 4         # 元路径的长度为 4

      output: 元路径的边类型

        [    
            [0, 1, 0], 
            [0, 1, 2], 
            [1, 0, 1], 
            [1, 2, 3], 
            [2, 3, 0], 
            [2, 3, 2], 
            [3, 0, 1], 
            [3, 2, 3]
        ]
    '''
    hop = []
    # hop = [[0], [1], [2], [3]]
    for type in type_dict.keys():
        hop.append([type])
    # 控制扩展的次数。length - 2 是因为初始时元路径长度已经是 1，还需要扩展 length - 2 次。
    for i in range(length - 2):
        new_hop = []
        for path in hop:
            for next_type in type_dict[path[-1]]:
                new_hop.append(path + [next_type])
        hop = new_hop
    return hop


def enum_all_metapath(name_dict, type_dict, length):
    '''
      name_dict = {
        0: ("P", "A"),  # 类型 0 的名称是 "P"，连接类型是 "A"
        1: ("A", "P"),  # 类型 1 的名称是 "A"，连接类型是 "P"
        2: ("P", "C"),  # 类型 2 的名称是 "P"，连接类型是 "C"
        3: ("C", "P")   # 类型 3 的名称是 "C"，连接类型是 "P"
      }

      type_dict = {
        0: [1],        # 类型 0 可以连接到类型 1
        1: [0, 2],     # 类型 1 可以连接到类型 0 或类型 2
        2: [3],        # 类型 2 可以连接到类型 3
        3: [0, 2]      # 类型 3 可以连接到类型 0 或类型 2
      }

      length = 4         # 元路径的长度为 4

      {
        'PA': [0], 'AP': [1], 'PC': [2], 'CP': [3], 
        'PAP': [0, 1], 'APA': [1, 0], 'APC': [1, 2], 'PCP': [2, 3], 'CPA': [3, 0], 'CPC': [3, 2], 
        'PAPA': [0, 1, 0], 'PAPC': [0, 1, 2], 'APAP': [1, 0, 1], 'APCP': [1, 2, 3], 'PCPA': [2, 3, 0], 'PCPC': [2, 3, 2], 'CPAP': [3, 0, 1], 'CPCP': [3, 2, 3]
      }
    '''
    hop = []
    path_list = []
    for type in type_dict.keys():
        hop.append([type])
    # 包含一条边的元路径
    path_list.extend(hop)
    for i in range(length - 2):
        new_hop = []
        for path in hop:
            for next_type in type_dict[path[-1]]:
                new_hop.append(path + [next_type])
        hop = new_hop
        # 把包含两条边、三条边，。。。，length - 1 条边的元路径都加进来
        path_list.extend(hop)
    # 找到每条边的名称
    path_dict = {}
    for path in path_list:
        name = name_dict[path[0]][0]
        for index in path:
            name += name_dict[index][1]
        path_dict[name] = path
    return path_dict


def enum_metapath_name(name_dict, type_dict, length):
    # 枚举所有可能的metapath名字
    # 结果按类型返回
    '''
      name_dict = {
        0: ("P", "A"),  # 类型 0 的名称是 "P"，连接类型是 "A"
        1: ("A", "P"),  # 类型 1 的名称是 "A"，连接类型是 "P"
        2: ("P", "C"),  # 类型 2 的名称是 "P"，连接类型是 "C"
        3: ("C", "P")   # 类型 3 的名称是 "C"，连接类型是 "P"
      }

      type_dict = {
        0: [1],        # 类型 0 可以连接到类型 1
        1: [0, 2],     # 类型 1 可以连接到类型 0 或类型 2
        2: [3],        # 类型 2 可以连接到类型 3
        3: [0, 2]      # 类型 3 可以连接到类型 0 或类型 2
      }

      length = 4         # 元路径的长度为 4

      {
        'P': ['PA', 'PC', 'PAP', 'PCP', 'PAPA', 'PAPC', 'PCPA', 'PCPC'], 
        'A': ['AP', 'APA', 'APC', 'APAP', 'APCP'], 
        'C': ['CP', 'CPA', 'CPC', 'CPAP', 'CPCP']
      }
    '''
    hop = []
    path_list = []
    result_dict = {}
    for type in type_dict.keys():
        hop.append([type])
        result_dict[name_dict[type][0]] = []
    path_list.extend(hop)
    for i in range(length - 2):
        new_hop = []
        for path in hop:
            for next_type in type_dict[path[-1]]:
                new_hop.append(path + [next_type])
        hop = new_hop
        path_list.extend(hop)
    for path in path_list:
        name = name_dict[path[0]][0]
        for index in path:
            name += name_dict[index][1]
        if len(name) > 1:
            result_dict[name[0]].append(name)
    return result_dict


def search_all_path(graph_list, src_node, name_list, metapath_list, metapath_name, path_single_limit=None):
    '''
        遍历一组元路径，
        调用 search_single_path 来搜索符合这些元路径的路径，
        并将结果汇总到 path_dict 中
        path_dict: 包含元路径的字典, 其中键是元路径名称, 值是符合该元路径的节点索引路径列表.
    '''
    # metapath_name: data.get_metapath_name
    
    path_dict = {}
    for path in metapath_list:
        path_dict.update(search_single_path(graph_list, src_node, name_list, path, metapath_name, path_single_limit))
    return path_dict



def search_single_path(graph_list, src_node, name_list, type_sequence, metapath_name, path_single_limit):
    '''
        graph_list: 一个列表，其中每个元素是一个图，图是用邻接链表表示的。每个图代表一种边类型的图。
        src_node: 起始节点，搜索从这个节点开始。
        name_list: 存储元路径名称的字典，后面会根据这些元路径进行路径匹配。
        type_sequence: 一个列表，表示一个元路径的边类型序列。每个元素是一个整数，表示某种边的类型。
        metapath_name: 包含每个元路径名称的字典，帮助根据边类型确定路径名称。
        path_single_limit: 限制每个节点最多探索的邻居数量，防止过度扩展。

        在异构图中根据给定的边类型序列 (type_sequence) 从源节点 (src_node) 出发, 搜索所有可能的路径的函数.
        它的主要目标是根据一个元路径 (meta-path) 的边类型序列, 从源节点出发, 逐步沿着边类型序列的路径探索, 直到达到目标。
    '''

    '''
        目的: 检查起始节点 src_node 是否在指定的图中存在，并且是否有邻居（即与其他节点有连接）。
        解析:
        graph_list[type_sequence[0]] 访问图中类型为 type_sequence[0] 的边（即元路径的第一个边类型）的邻接链表。
        graph_list[type_sequence[0]][src_node] 获取起始节点 src_node 的邻居列表。
        如果该节点不存在或没有任何邻居，返回一个空字典，表示无法从该节点出发搜索路径。

    '''
    if src_node not in graph_list[type_sequence[0]] or len(graph_list[type_sequence[0]][src_node]) == 0:
        return {}
    path_result = [[[src_node]]]

    hop = len(type_sequence)
    # 执行邻接矩阵BFS搜索
    for l in range(hop):
        path_result.append([])
        for list in path_result[l]:
            path_result[l + 1].extend(list_appender(list, graph_list, type_sequence[l], path_single_limit))
    # 将搜索结果做量的限制，然后按 metapath 名字保存下来
    path_dict = {}
    #  初始化第一个元路径名称。
    fullname = metapath_name[type_sequence[0]][0]
    # 将第一层的路径保存到 path_dict 中，fullname[0] 表示元路径的起始名称。
    path_dict[fullname[0]] = path_result[0]
    # 对于后续的每个边类型（根据 type_sequence），依次拼接元路径名称（例如 PA、AP 等）。
    for i in type_sequence:
        fullname += metapath_name[i][1]
    # 遍历完整的元路径名称 (fullname)
    for i in range(len(fullname)):
        '''
            对每个阶段的路径 (path_result[i]), 检查是否在 name_list 中存在该元路径名称。
            如果存在，则将当前路径存入 path_dict。
        '''
        if len(path_result[i]) != 0 and fullname[0:i + 1] in name_list[fullname[0]]:
            path_dict[fullname[0:i + 1]] = path_result[i]

    return path_dict


def list_appender(list, graph_list, type, path_limit):
    # 在每条 metapath 的基础上再 BFS 搜一步。
    # type: 边的类型
    # path_limit: 每个节点最多探索的邻居数量，防止过度扩展。
    result = []

    # 如果路径最后一个节点不在当前类型的图里
    if list[-1] not in graph_list[type]: return []

    # 如果没有数量限制并且邻居节点数量大于 path_limit
    if path_limit != None and len(graph_list[type][list[-1]]) > path_limit:
        neighbors = random.sample(graph_list[type][list[-1]], path_limit)
    else:
        neighbors = graph_list[type][list[-1]]
    for neighbor in neighbors:
        # 把新邻居点扩进来
        if neighbor not in list:
            result.append(list + [neighbor])
    return result


def index_to_features(path_dict, x, select_method="all_node"):
    '''
        将点序列编号变为features矩阵
        预先申请空间以加快速度

        path_dict: 包含元路径的字典, 其中键是元路径名称, 值是符合该元路径的节点索引路径列表.
        x: 是一个形状为 (num_nodes, feature_dim) 的矩阵, 表示图中所有节点的特征。每一行代表一个节点的特征向量。
        select_method: 表示如何选择特征。"all_node" 表示选择路径上的所有节点特征, "end_node" 表示仅选择路径末尾节点的特征。
    '''
    '''
        初始化一个空字典 result_dict, 用于存储最终的特征矩阵。
        字典的键是元路径名称, 值是该元路径下节点的特征矩阵。
    '''
    result_dict = {}
    for name in path_dict.keys():
        if len(name) == 1:
            # path_dict[name][0][0]: 起始节点索引
            result_dict[name] = x[None, path_dict[name][0][0], :]
            # 标识元路径的起点是属于哪个节点类型
            result_dict['src_type'] = name
            continue
        np_index = np.array(path_dict[name], dtype=np.int64)
        if select_method == "end_node":
                np_x = np.empty([np_index.shape[0], x.shape[1]])
                np_x[:, 0:x.shape[1]] = x[np_index[:, -1], :]
        else:
            # np_index.shape[0]: 该节点该类型元路径的数量
            # (np_index.shape[1] - 1) * x.shape[1]: 去除元路径起始节点后的维度
            np_x = np.empty([np_index.shape[0], (np_index.shape[1] - 1) * x.shape[1]])
            for i in range(np_index.shape[1] - 1):
                np_x[:, i * x.shape[1] : (i + 1) * x.shape[1]] = x[np_index[:, i + 1], :]
        result_dict[name] = np_x

    return result_dict


def combine_features_dict(list_of_node_dict, batch_src_index, batch_src_label, DEVICE):
    '''
        将多个点的特征字典按 metapath 堆叠起来
        首先取 metapath 并集

        list_of_node_dict: [{ metapath-type: feature }, ..., {}] 多个节点特征字典的列表，每个字典包含多个 metapath 类型的特征矩阵。
        batch_src_index: [index1, index2, ..., indexn] 节点在图中的真实索引。
        batch_src_label: [label1, label2, ..., labeln] 节点对应的标签（如分类标签）。
    '''
    # 记录每个类型的 metapath 并集。
    metapath_dict = {}
    # 存储每个 metapath 对应的堆叠特征矩阵。
    feature_dict = {}
    '''
        记录每个 metapath 对应节点的特征行数。
        一个节点针对某一类型的元路径可能有多条, 后面需要aggregate, 因此需要记录下行数
    '''
    row_dict = {}
    # 记录每个 metapath 对应特征的列数。
    column_dict = {}
    # 按照节点的类型（src_type）进行分类的字典。
    type_dict = {}
    tensor_dict = {}
    # 存储每个类型节点的索引。
    index_dict = {}
    # 存储每个类型节点的标签。
    label_dict = {}
    # 先统计点的类型数目，并将点的编号分好类存在字典里
    for index in range(len(list_of_node_dict)):
        # 根据元路径的起始节点类型进行分类
        type = list_of_node_dict[index]['src_type']
        if type not in type_dict:
            type_dict[type] = []
            index_dict[type] = []
            label_dict[type] = []

        # 存储索引
        type_dict[type].append(index)
        # 存储标签 {'M': [label1, label2, label3, ..., labeln]}
        label_dict[type].append(batch_src_label[index])
        # 存储节点在图中的真实索引
        index_dict[type].append(batch_src_index[index])

    # type_dict: {'M': [0, 1, 2, 3, 4, ..., 95]}
    for type in type_dict:
        # 初始化每类的特征、张量和行号记录字典
        '''
            metapath_dict: {'M': {'MD', 'M', 'MAMA', 'MAMD', 'MDM', 'src_type', 'MA', 'MAM', 'MDMA'}}
        '''
        metapath_dict[type] = set()
        feature_dict[type] = {}
        # tensor_dict 并没有进行使用过
        tensor_dict[type] = {}
        row_dict[type] = {}
        column_dict[type] = {}
        # 把每类的label转为Tensor
        label_dict[type] = torch.Tensor(label_dict[type]).long().to(DEVICE)
        for node_index in type_dict[type]:
            # 对每类点的 metapath 取并集
            metapath_dict[type].update(list_of_node_dict[node_index].keys())
        # 移除多余的‘src_type' key。这个key在设计上必然存在。
        metapath_dict[type].remove('src_type')

    for type in type_dict:
        # 遍历不同的 metapath 类型
        for metapath in metapath_dict[type]:
            # 初始化行数列表
            row_dict[type][metapath] = []
            for node_index in type_dict[type]:
                # 对每个点的每个 metapath 统计特征行数，记录特征行数
                if metapath not in list_of_node_dict[node_index]:
                    # 该点没有此 metapath，记录0
                    row_dict[type][metapath].append(0)
                else:
                    # 记录行数，对每个类型的每个 metapath，初始化该 metapath 对应的行数列表
                    row_dict[type][metapath].append(list_of_node_dict[node_index][metapath].shape[0])
                    # 记录列数
                    column_dict[type][metapath] = list_of_node_dict[node_index][metapath].shape[1]
            # 初始化总特征矩阵
            # 将行数加总
            stack_list = []
            for i in range(len(type_dict[type])):
                if row_dict[type][metapath][i] == 0:
                    # 该点没有该metapath，跳过
                    continue
                else:
                    # 收集每个节点在某个特定 metapath 上的特征矩阵（torch.Tensor），然后将这些特征矩阵按行堆叠成一个大的特征矩阵。
                    # 这个过程是为了将各个节点的特征按照相同的 metapath 合并起来，形成统一的特征表示。
                    stack_list.append(torch.from_numpy(list_of_node_dict[type_dict[type][i]][metapath]))
            # 最后利用torch.cat节约时间
            feature_dict[type][metapath] = torch.cat(stack_list, dim=0).float().to(DEVICE)
    return feature_dict, index_dict, label_dict, row_dict

