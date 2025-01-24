import torch
import random
import pickle
import os
from metapath import enum_all_metapath, search_all_path


def load_graph_data(folder_path):
    """
    从文件夹加载图数据并构建 graph_list 格式。
    """
    graph_list = {}

    # 加载边
    edges_cites = torch.load(f"{folder_path}/cites_edges.pt")
    edges_written_by = torch.load(f"{folder_path}/written_by_edges.pt")
    edges_published_in = torch.load(f"{folder_path}/published_in_edges.pt")

    # 确保边数据为列表形式
    edges_cites = edges_cites.t().tolist()  # 转置并转换为列表，每列变成一个 [src, dst]
    edges_written_by = edges_written_by.t().tolist()
    edges_published_in = edges_published_in.t().tolist()

    # 初始化 graph_list
    graph_list['Paper'] = {}
    graph_list['Author'] = {}
    graph_list['Venue'] = {}

    # 解析 cites 边
    for src, dst in edges_cites:
        if src not in graph_list['Paper']:
            graph_list['Paper'][src] = []
        graph_list['Paper'][src].append(dst)

    # 解析 written_by 边
    for src, dst in edges_written_by:
        if src not in graph_list['Paper']:
            graph_list['Paper'][src] = []
        if dst not in graph_list['Author']:
            graph_list['Author'][dst] = []
        graph_list['Paper'][src].append(dst)
        graph_list['Author'][dst].append(src)

    # 解析 published_in 边
    for src, dst in edges_published_in:
        if src not in graph_list['Paper']:
            graph_list['Paper'][src] = []
        if dst not in graph_list['Venue']:
            graph_list['Venue'][dst] = []
        graph_list['Paper'][src].append(dst)
        graph_list['Venue'][dst].append(src)

    # 打印解析结果
    print("Graph List:")
    for node_type, neighbors in graph_list.items():
        print(f"{node_type}: {len(neighbors)} nodes")
    return graph_list


def generate_metapaths(graph_list, max_length):
    """
    根据图数据生成所有可能的元路径模式（限制最大长度）。
    """
    # 构造 name_dict，记录每种类型的缩写名
    name_dict = {
        'Paper': ['P', 'P'],
        'Author': ['A', 'A'],
        'Venue': ['V', 'V']
    }

    # 构造 type_dict，记录每种类型的可能邻居类型
    type_dict = {
        'Paper': ['Author', 'Venue', 'Paper'],
        'Author': ['Paper'],
        'Venue': ['Paper']
    }

    # 调用 enum_all_metapath 枚举元路径
    metapaths = enum_all_metapath(name_dict, type_dict, max_length)

    return metapaths


def extract_metapath_instances(graph_list, selected_nodes, metapaths, path_single_limit=None):
    """
    为指定节点集合提取元路径实例，并根据模式生成标签。
    保存结构为：{node_id: {label: [instances...]}}。
    """
    result = {}  # 使用字典存储，以节点为键

    for node in selected_nodes:
        node_dict = {}  # 当前节点的所有元路径实例，按标签分组

        for metapath_index, (metapath_name, metapath_list) in enumerate(metapaths.items()):
            # 元路径类型序列 (name_list)
            name_list = metapath_list

            # 搜索元路径实例
            paths = search_all_path(
                graph_list,
                src_node=node,
                name_list=name_list,
                metapath_list=[name_list],
                metapath_name=metapath_name,
                path_single_limit=path_single_limit,
            )
            # 将找到的路径添加到对应的标签下
            if metapath_index not in node_dict:
                node_dict[metapath_index] = []
            node_dict[metapath_index].extend(paths.keys())  # paths 是一个字典，键是实例路径

        # 保存当前节点的实例
        result[node] = node_dict

    return result


if __name__ == "__main__":
    # 参数设置
    folder_path = "output_tf-idf"  # 包含节点特征和边文件的文件夹
    max_metapath_length = 3  # 限制元路径最大长度
    random_seed = 42  # 固定随机种子以确保结果一致
    random.seed(random_seed)

    # 动态生成保存文件夹名称
    save_path = f"./metapath_results_separate_length_{max_metapath_length}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 加载图数据
    graph_list = load_graph_data(folder_path)

    # 自动生成所有元路径模式
    metapaths = generate_metapaths(graph_list, max_metapath_length)
    print(f"Generated {len(metapaths)} metapaths.")
    print("Sample metapaths:", metapaths)

    # 构建标签到元路径的映射
    label_to_metapath = {index: (name, path) for index, (name, path) in enumerate(metapaths.items())}

    # 随机选择 10% 的节点
    num_papers = len(graph_list['Paper'])
    num_authors = len(graph_list['Author'])
    num_venues = len(graph_list['Venue'])

    selected_papers = random.sample(list(graph_list['Paper'].keys()), int(0.01 * num_papers))
    selected_authors = random.sample(list(graph_list['Author'].keys()), int(0.01 * num_authors))
    selected_venues = random.sample(list(graph_list['Venue'].keys()), int(0.01 * num_venues))

    # 提取元路径实例和标签
    paper_results = extract_metapath_instances(graph_list, selected_papers, metapaths)
    author_results = extract_metapath_instances(graph_list, selected_authors, metapaths)
    venue_results = extract_metapath_instances(graph_list, selected_venues, metapaths)

    # 保存结果
    with open(os.path.join(save_path, "paper_metapath_instances.pkl"), "wb") as f:
        pickle.dump(paper_results, f)

    with open(os.path.join(save_path, "author_metapath_instances.pkl"), "wb") as f:
        pickle.dump(author_results, f)

    with open(os.path.join(save_path, "venue_metapath_instances.pkl"), "wb") as f:
        pickle.dump(venue_results, f)

    with open(os.path.join(save_path, "metapath_patterns.pkl"), "wb") as f:
        pickle.dump(metapaths, f)

    # 保存标签到模式的映射
    with open(os.path.join(save_path, "label_to_metapath.pkl"), "wb") as f:
        pickle.dump(label_to_metapath, f)

    print(f"Metapath instances, labels, and patterns have been saved successfully in {save_path}.")