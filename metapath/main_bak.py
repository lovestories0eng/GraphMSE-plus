import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import metapath as type_list
import sklearn.metrics as sm
from multiprocessing.dummy import Pool as ThreadPool
from model_bak import MetapathEncoder, MetapathClassifier
from data import HeteData
from store import GlobalStore

import argparse

def node_search_wrapper(index):
    return type_list.search_all_path(
        graph_list, 
        index, 
        metapath_name, 
        metapath_list, 
        data.get_metapath_name(), 
        single_path_limit
    )

def index_to_feature_wrapper(dict):
    return type_list.index_to_features(dict, data.x)

def combine_dict(batch_feature, batch_type):
    length = len(batch_feature)
    features = {}
    for i in range(length):
        if batch_type[i] not in features:
            features[batch_type[i]] = []
        features[batch_type[i]].append(batch_feature[i])
    
    return features

def get_feature_dict(batch_size, metapath_cnt, feature_dict):
    # 均匀采样，每类采样 num_per_type 个
    num_per_type = batch_size // metapath_cnt

    batch_choice = []
    batch_feature = []
    batch_type = []

    for metapath_type in feature_dict:
        cur_feature = feature_dict[metapath_type]
        cur_type_metapath_feature_num = len(cur_feature)

        batch_choice_per_type = np.random.choice(cur_type_metapath_feature_num, num_per_type)
        batch_feature_per_type = [cur_feature[i] for i in batch_choice_per_type]
        batch_type_per_type = [metapath_type] * num_per_type

        batch_choice.extend(batch_choice_per_type)
        batch_feature.extend(batch_feature_per_type)
        batch_type.extend(batch_type_per_type)

    cur_features_train_dict = combine_dict(batch_feature, batch_type)
    return cur_features_train_dict

def contrastive_loss(embeddings, labels, margin=1.0):
    # embeddings: 当前批次的 embeddings
    # labels: 元路径所对应的标签
    # margin: 对比损失的阈值
    batch_size = embeddings.size(0)
    loss = 0
    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            # 计算欧几里得距离
            dist = torch.norm(embeddings[i] - embeddings[j], p=2)
            y = labels[i] == labels[j]
            y = y.float()
            # 对比损失
            loss += y * dist ** 2 + (1 - y) * F.relu(margin - dist) ** 2
    return loss / (batch_size * (batch_size + 1) / 2)

def center_loss(embeddings, labels):
    # embeddings: 当前批次的特征
    # labels: 当前批次的标签

    batch_size = embeddings.size(0)
    # 计算每一种类型的 center
    centers = {}
    for i in range(batch_size):
        cur_label = (labels[i]).item()
        if cur_label not in centers:
            centers[cur_label] = [embeddings[i]]
        else:
            centers[cur_label].append(embeddings[i])

    for label in centers.copy():
        embeddings_per_label = centers[label]
        cur_tensor = torch.stack(embeddings_per_label).to(DEVICE)
        centers[label] = (cur_tensor.mean(dim=0, keepdim=True)).flatten()

    loss = 0
    for i in range(batch_size):
        cur_label = (labels[i]).item()
        center = centers[cur_label]
        loss += F.mse_loss(embeddings[i], center)
    return loss / batch_size

def combined_loss(embeddings, labels, margin=1.0, lambda_center=0.5):
    # 计算对比损失
    contrast_loss_val = contrastive_loss(embeddings, labels, margin)
    
    # 计算中心损失
    center_loss_val = center_loss(embeddings, labels)
    
    # 综合损失
    total_loss = contrast_loss_val + lambda_center * center_loss_val
    return total_loss

def data_split(split, dict_to_split):
    split_dict = []
    for i in range(len(split)):
        split_dict.append({})
    last_break = 0
    cur_break = 0
    for i in range(len(split)):
        d = split_dict[i]
        cur_break = split[i]
        for cur_type in dict_to_split:
            cur_value = dict_to_split[cur_type]
            d[cur_type] = cur_value[int(len(cur_value) * last_break):int(len(cur_value) * cur_break)]
        last_break = split[i]

    return split_dict

def train(model, metapath_index_dict, val_num, epochs=100, save_path='best_model.pth'):
    # 存储训练完得到的元路径类型 embedding
    metapath_embeddings = []
    embedding_labels = []
    metapath_mlps = nn.ModuleDict()

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        model.train()

        cur_features_train_dict = get_feature_dict(batch_size, metapath_cnt, MPE_train_dict)

        optimizer.zero_grad()
        outputs, metapath_labels, metapath_embeddings, embedding_labels, metapath_mlps = model(cur_features_train_dict, metapath_index_dict)
            
        global_store.metapath_embedding_store.append(metapath_embeddings)
            
        loss = combined_loss(outputs, metapath_labels)
        loss.backward()
        optimizer.step()
        global_store.train_loss_append(loss.item())

        # # 验证模型，采样一部分数据集
        cur_features_val_dict = get_feature_dict(val_num, metapath_cnt, MPE_val_dict)
        val_loss = val(model, cur_features_val_dict, metapath_index_dict)
        global_store.val_loss_append(val_loss)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Val Loss: {val_loss}')

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        # 保存模型
        torch.save(model.state_dict(), save_path)

    return model, metapath_embeddings, embedding_labels, metapath_mlps

def val(model, features, metapath_index_dict):
    model.eval()
    with torch.no_grad():
        outputs, metapath_labels, *_ = model(features, metapath_index_dict)
        loss = combined_loss(outputs, metapath_labels)

    return loss.item()

def test(model, features, metapath_index_dict, metapath_embeddings, embedding_labels):
    model.eval()

    with torch.no_grad():
        outputs, metapath_labels, *_ = model(features, metapath_index_dict)

        cur_len = outputs.shape[0]
        for i in range(cur_len):
            label_index = embedding_labels.index(metapath_labels[i])
            cur_metapath_embedding = metapath_embeddings[label_index]
            cos_similarity = F.cosine_similarity(
                outputs[i].unsqueeze(0), cur_metapath_embedding.unsqueeze(0)
            )
            euclidean_dist = torch.dist(outputs[i], cur_metapath_embedding)

            cur_label = metapath_labels[i].item()

            for j in range(len(embedding_labels)):
                label_index = embedding_labels[j]
                cur_metapath_embedding = metapath_embeddings[label_index]

                cos_similarity = F.cosine_similarity(
                    outputs[i].unsqueeze(0), cur_metapath_embedding.unsqueeze(0)
                )
                euclidean_dist = torch.dist(outputs[i], cur_metapath_embedding)

                global_store.update_sim_dict(cur_label, embedding_labels[j], cos_similarity.item())
                global_store.update_euclidean_dict(cur_label, embedding_labels[j], euclidean_dist.item())

def downstram_task(model, feature_dict_train, feature_dict_test, metapath_index_dict, batch_size, epochs=100):
    # 自动将模型输出的 raw scores（未归一化的分数）通过 Softmax 转换为概率，然后计算这些概率和目标标签之间的损失
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        cur_features_train_dict = get_feature_dict(batch_size, metapath_cnt, feature_dict_train)

        optimizer.zero_grad()
        outputs, metapath_labels = model(cur_features_train_dict, metapath_index_dict)
        loss = criterion(outputs, metapath_labels)
        loss.backward()
        optimizer.step()

        global_store.downstream_losses_append(loss.item())

        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

    model.eval()
    with torch.no_grad():
        outputs, metapath_labels = model(feature_dict_test, metapath_index_dict)
        loss = criterion(outputs, metapath_labels)

        _, y_pred = torch.max(outputs, dim=1)
        y_true = metapath_labels.cpu().numpy().tolist()

        y_pred = y_pred.cpu().numpy().tolist()

        micro_f1 = sm.f1_score(y_true, y_pred, average='micro')
        macro_f1 = sm.f1_score(y_true, y_pred, average='macro')

        global_store.f1_score_append(micro_f1, macro_f1)

        print("test finished, loss: ", loss)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, help="数据集")
    parser.add_argument("--sample_times", type=int, help="每个类型元路径的采样数量")
    parser.add_argument("--sample_num", type=int, help="用于聚合的元路径数量")

    args = parser.parse_args()

    dataset = args.dataset
    sample_times = args.sample_times
    sample_num = args.sample_num

    global_store = GlobalStore.get_instance()
    DEVICE = global_store.DEVICE

    global_store.update({
        "dataset": dataset,
        "sample_times": sample_times,
        "sample_num": sample_num
    })

    train_percent = None  # 20, 40, 60, 80
    metapath_length = 5
    mlp_settings = {'layer_list': [256], 'dropout_list': [0.2], 'activation': 'sigmoid'}
    single_path_limit = 5
    # 学习率
    learning_rate = 0.01
    num_epochs = 100

    # 获取数据
    data = HeteData(dataset=dataset, train_percent=train_percent)
    graph_list = data.get_dict_of_list()


    # 节点特征
    x = data.x

    node_embedding_dim = x.shape[1]

    metapath_name = type_list.enum_metapath_name(data.get_metapath_name(), data.get_metapath_dict(), metapath_length)
    metapath_list = type_list.enum_longest_metapath_index(data.get_metapath_name(), data.get_metapath_dict(), metapath_length)

    # 并发地执行多个任务
    num_thread = 12
    pool = ThreadPool(num_thread)

    metapath = pool.map(node_search_wrapper, range(data.x.shape[0]))
    features = pool.map(index_to_feature_wrapper, metapath)

    # {key: metapath name, value: metapath}
    type_list = {}
    # {key: metapath name, value: metapath feature}
    type_feature = {}

    for node_dict in metapath:
        for metapath_name, metapath_list in node_dict.items():
            if metapath_name not in type_list:
                type_list[metapath_name] = []
            type_list[metapath_name].extend(metapath_list)

    for node_dict in features:
        for metapath_name, metapath_feature in node_dict.items():
            if metapath_name not in type_feature:
                type_feature[metapath_name] = []
            type_feature[metapath_name].extend(metapath_feature)

    num_threshold = 4096

    # 遍历字典的副本
    for key, value in type_list.copy().items():
        # 删除长度小于 num_threshold 的键值对
        if len(value) < num_threshold or len(key) == 1:
            del type_list[key]

    for key, value in type_feature.copy().items():
        # 删除长度小于 num_threshold 的键值对
        if len(value) < num_threshold or len(key) == 1:
            del type_feature[key]

    downstream_type_list = {}
    downstream_type_feature = {}
    # 80% 数据用于训练，20% 数据用于下游任务
    for key, value in type_list.copy().items():
        cur_len = len(type_list[key])
        downstream_type_list[key] = type_list[key][int(cur_len * 0.8):]
        type_list[key] = type_list[key][:int(cur_len * 0.8)]

        downstream_type_feature[key] = type_feature[key][int(cur_len * 0.8):]
        type_feature[key] = type_feature[key][:int(cur_len * 0.8)]

    metapath_types = list(type_list.keys())

    MPE_dict = {}
    metapath_index_dict = {}
    metapath_cnt = 0

    # 采样 MPE
    for metapath_type in metapath_types:
        cur_feature = type_feature[metapath_type]
        cur_type_metapath_feature_num = len(cur_feature)
        MPE_list = []

        for i in range(sample_times):
            cur_metapath_index = np.random.choice(cur_type_metapath_feature_num, sample_num)
            cur_metapath_feature = [cur_feature[i] for i in cur_metapath_index]
            MPE_list.append(cur_metapath_feature)

        MPE_dict[metapath_type] = MPE_list
        metapath_index_dict[metapath_type] = metapath_cnt
        metapath_cnt += 1

    global_store.update({"metapath_cnt": metapath_cnt})

    [MPE_train_dict, MPE_val_dict, MPE_test_dict] = data_split([0.6, 0.8, 1], MPE_dict)

    print('start training...')

    # 让 batch_size 为 metapath_cnt 的整数倍
    batch_size = metapath_cnt * 8
    val_num = metapath_cnt * 8

    info_section = 64
    pre_embed_dim = info_section * data.type_num


    # 调用训练函数
    model = MetapathEncoder(metapath_types, node_embedding_dim, pre_embed_dim, mlp_settings, 128).to(DEVICE)
    trained_model, metapath_embeddings, embedding_labels, metapath_mlps = train(
        model,
        metapath_index_dict,
        val_num,
        epochs=num_epochs
    )

    test(trained_model, MPE_test_dict, metapath_index_dict, metapath_embeddings, embedding_labels)


    DownstreamModel = MetapathClassifier(metapath_types, mlp_settings, node_embedding_dim, metapath_embeddings, embedding_labels, False).to(DEVICE)
    DownstreamModelConcat = MetapathClassifier(metapath_types, mlp_settings, node_embedding_dim, metapath_embeddings, embedding_labels, True).to(DEVICE)

    [train_data, test_data] = data_split([0.7, 1], downstream_type_feature)
    downstream_batch_size = metapath_cnt * 1

    # 调用测试函数
    print('Testing [...metapath instance feature]')
    downstram_task(
        DownstreamModel,
        train_data,
        test_data,
        metapath_index_dict,
        downstream_batch_size
    )

    print('Testing [...metapath instance feature, ...metapath embedding]')
    downstram_task(
        DownstreamModelConcat,
        train_data,
        test_data,
        metapath_index_dict,
        downstream_batch_size
    )

    global_store.save_data()

    print('done!')
# nohup python -u main_bak.py --dataset IMDB --sample_times 4096 --sample_num 128 < /dev/null  > output.log 2>&1 &
# nohup ./experiment.sh < /dev/null  > output.log 2>&1 &