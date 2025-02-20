import torch
import torch.nn as nn
import numpy as np
import metapath as type_list
import sklearn.metrics as sm
from multiprocessing.dummy import Pool as ThreadPool
from model import MetapathClassifier
from data import HeteData

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

def train(model, features_train, labels_train, types_train, features_val, labels_val, types_val, epochs=100):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float('inf')
    best_model_state = None

    # 将 features_train 和 types_train 列表转换为 numpy.ndarray
    features_train = np.array(features_train)
    types_train = np.array(types_train)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in range(num_batch_per_epoch):
            optimizer.zero_grad()
            outputs = model(torch.tensor(features_train).to(DEVICE), torch.tensor(types_train).to(DEVICE))
            loss = criterion(outputs, torch.tensor(labels_train).to(DEVICE))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / num_batch_per_epoch

        # 验证模型
        val_loss, val_accuracy = val(model, features_val, labels_val, types_val)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}')

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model

def val(model, features, labels, types):
    model.eval()
    with torch.no_grad():
        outputs = model(torch.tensor(features).to(DEVICE), torch.tensor(types).to(DEVICE))
        loss = criterion(outputs, torch.tensor(labels).to(DEVICE))
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == torch.tensor(labels).to(DEVICE)).float().mean().item()

    return loss.item(), accuracy

def test(model, features, labels, types):
    model.eval()
    with torch.no_grad():
        outputs = model(torch.tensor(features).to(DEVICE), torch.tensor(types).to(DEVICE))
        loss = criterion(outputs, torch.tensor(labels).to(DEVICE))
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == torch.tensor(labels).to(DEVICE)).float().mean().item()
        f1_score = sm.f1_score(labels, preds.cpu(), average='macro')

    return loss.item(), accuracy, f1_score
    

dataset = "IMDB"
train_percent = None  # 20, 40, 60, 80
metapath_length = 4
mlp_settings = {'layer_list': [256, 128], 'dropout_list': [0.3, 0.3], 'activation': 'sigmoid'}
single_path_limit = 10

data = HeteData(dataset=dataset, train_percent=train_percent)
graph_list = data.get_dict_of_list()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 获取预处理数据

# 节点特征
x = data.x

node_embedding_dim = x.shape[1]

# 自动将模型输出的 raw scores（未归一化的分数）通过 Softmax 转换为概率，然后计算这些概率和目标标签之间的损失
criterion = nn.CrossEntropyLoss().to(DEVICE)

metapath_name = type_list.enum_metapath_name(data.get_metapath_name(), data.get_metapath_dict(), metapath_length)
metapath_list = type_list.enum_longest_metapath_index(data.get_metapath_name(), data.get_metapath_dict(), metapath_length)

# 并发地执行多个任务
num_thread = 12
pool = ThreadPool(num_thread)

metapath = pool.map(node_search_wrapper, range(data.x.shape[0]))
features = pool.map(index_to_feature_wrapper, metapath)

# {key: metapath name, value: matapath}
type_list = {}
# {key: metapath name, value: matapath feature}
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

metapath_type = list(type_list.keys())

features_train = []
features_val = []
features_test = []

metapath_index_train = []
metapath_index_val = []
metapath_index_test = []

metapath_type_train = []
metapath_type_val = []
metapath_type_test = []

metapath_cnt = 0
for metapath_type in type_list:
    # 只选择节点数大于等于 1000 的元路径
    if len(type_feature[metapath_type]) < 1000:
        continue
    
    cur_feature = type_feature[metapath_type]

    # 把 cur_feature 分成 6:2:2
    features_train.extend(cur_feature[:int(len(cur_feature) * 0.6)])
    features_val.extend(cur_feature[int(len(cur_feature) * 0.6):int(len(cur_feature) * 0.8)])
    features_test.extend(cur_feature[int(len(cur_feature) * 0.8):])

    metapath_index_train.extend([metapath_cnt] * int(len(cur_feature) * 0.6))
    metapath_index_val.extend([metapath_cnt] * (int(len(cur_feature) * 0.8) - int(len(cur_feature) * 0.6)))
    metapath_index_test.extend([metapath_cnt] * (len(cur_feature) - int(len(cur_feature) * 0.8)))

    metapath_type_train.extend([metapath_cnt] * int(len(cur_feature) * 0.6))
    metapath_type_val.extend([metapath_cnt] * (int(len(cur_feature) * 0.8) - int(len(cur_feature) * 0.6)))
    metapath_type_test.extend([metapath_cnt] * (len(cur_feature) - int(len(cur_feature) * 0.8)))

    metapath_cnt += 1

# 学习率
learning_rate = 0.01 
# 每个 epoch 循环的批次数
num_batch_per_epoch = 5
num_epochs = 100

# 调用训练函数
model = MetapathClassifier(metapath_type, node_embedding_dim, metapath_cnt, mlp_settings).to(DEVICE)
trained_model = train(
    model, 
    features_train, 
    metapath_index_train,
    metapath_type_train,
    features_val, 
    metapath_index_val, 
    metapath_type_val,
    epochs=num_epochs
)

# 调用测试函数
test_loss, test_accuracy, test_f1_score = test(
    trained_model, 
    features_test, 
    metapath_index_test,
    metapath_type_test
)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}, Test F1 Score: {test_f1_score}')
