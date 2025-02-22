import torch
import torch.nn as nn
import numpy as np
import metapath as type_list
import sklearn.metrics as sm
from multiprocessing.dummy import Pool as ThreadPool
from model import MetapathClassifier, MetapathEmbedding
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

def combine_dict(batch_feature, batch_type):
    length = len(batch_feature)
    features = {}
    for i in range(length):
        if batch_type[i] not in features:
            features[batch_type[i]] = []
        features[batch_type[i]].append(batch_feature[i])
    
    return features


def train(
        model, 
        metapath_index_dict,
        val_num,
        epochs=100,
        save_path='best_model.pth'  # 添加保存路径参数
    ):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for _ in range(num_batch_per_epoch):
            batch_choice = np.random.choice(len(train_list), batch_size)
            batch_feature = [train_list[i][0] for i in batch_choice]
            batch_type = [train_list[i][1] for i in batch_choice]

            cur_features_train_dict = combine_dict(batch_feature, batch_type)

            optimizer.zero_grad()
            outputs, metapath_label = model(cur_features_train_dict, metapath_index_dict)
            loss = criterion(outputs, torch.tensor(metapath_label).to(DEVICE))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()


        # 验证模型，采样一部分数据集
        batch_val_choice = np.random.choice(len(val_list), val_num)
        batch_val_feature = [val_list[i][0] for i in batch_val_choice]
        batch_val_type = [val_list[i][1] for i in batch_val_choice]

        cur_features_val_dict = combine_dict(batch_val_feature, batch_val_type)

        val_loss, val_accuracy = val(model, cur_features_val_dict, metapath_index_dict)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}')

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        # 保存模型
        torch.save(model.state_dict(), save_path)

    return model

def val(model, features, metapath_index_dict):
    model.eval()
    with torch.no_grad():
        outputs, metapath_labels = model(features, metapath_index_dict)
        loss = criterion(outputs, torch.tensor(metapath_labels).to(DEVICE))
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == torch.tensor(metapath_labels).to(DEVICE)).float().mean().item()

    return loss.item(), accuracy

def test(model, features, metapath_index_dict):
    model.eval()
    with torch.no_grad():
        outputs, metapath_label = model(features, metapath_index_dict)
        loss = criterion(outputs, torch.tensor(metapath_label).to(DEVICE))
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == torch.tensor(metapath_label).to(DEVICE)).float().mean().item()
        f1_score = sm.f1_score(metapath_label, preds.cpu(), average='macro')

    return loss.item(), accuracy, f1_score
    

dataset = "IMDB"
train_percent = None  # 20, 40, 60, 80
metapath_length = 4
mlp_settings = {'layer_list': [256], 'dropout_list': [0.5], 'activation': 'sigmoid'}
single_path_limit = 5
batch_size = 64
# 学习率
learning_rate = 0.01 
# 每个 epoch 循环的批次数
num_batch_per_epoch = 5
num_epochs = 100
# num_epochs = 1

# 获取数据
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

metapath_types = list(type_list.keys())
val_num = batch_size * num_batch_per_epoch * len(metapath_types)

train_list = []
val_list = []
test_list = []

features_train_dict = {}
features_val_dict  = {}
features_test_dict  = {}

metapath_index_dict  = {}

metapath_cnt = 0
for metapath_type in type_list:
    # 只选择节点数大于等于 1000 的元路径
    if len(type_feature[metapath_type]) < 1000:
        continue
    
    cur_feature = type_feature[metapath_type]

    # 训练集、验证集、测试集分成 6:2:2

    features_train_dict[metapath_type] = cur_feature[:int(len(cur_feature) * 0.6)]
    features_val_dict[metapath_type] = cur_feature[int(len(cur_feature) * 0.6):int(len(cur_feature) * 0.8)]
    features_test_dict[metapath_type] = cur_feature[int(len(cur_feature) * 0.8):]

    train_list.extend([metapath_feature, metapath_type] for metapath_feature in features_train_dict[metapath_type])
    val_list.extend([metapath_feature, metapath_type] for metapath_feature in features_val_dict[metapath_type])
    test_list.extend([metapath_feature, metapath_type] for metapath_feature in features_test_dict[metapath_type])

    metapath_index_dict[metapath_type] = metapath_cnt

    metapath_cnt += 1

print('start training...')

info_section = 64
pre_embed_dim = info_section * data.type_num

# 调用训练函数
model = MetapathClassifier(metapath_types, node_embedding_dim, pre_embed_dim, metapath_cnt, mlp_settings).to(DEVICE)
trained_model = train(
    model,
    metapath_index_dict,
    val_num,
    epochs=num_epochs
)

# 调用测试函数
test_loss, test_accuracy, test_f1_score = test(
    trained_model,
    features_test_dict,
    metapath_index_dict
)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}, Test F1 Score: {test_f1_score}')

def train_MPE():
    pass

def val_MPE():
    pass

def test_MPE():
    pass

MPE_dict = {}
sample_times = 100

# 采样 MPE
for metapath_type in metapath_types:
    cur_feature = type_feature[metapath_type]
    cur_type_metapath_feature_num = len(cur_feature)

    sample_min = 100
    sample_max = cur_type_metapath_feature_num

    MPE_list = []
    
    # 每一类路径采样 100 次
    for i in range(sample_times):
        sample_num = np.random.randint(sample_min, sample_max)
        cur_metapath_index = np.random.choice(cur_type_metapath_feature_num, sample_num)
        cur_metapath_feature = [cur_feature[i] for i in cur_metapath_index]

        MPE_list.append(cur_metapath_feature)

    MPE_dict[metapath_type] = MPE_list

train_MPE_list = []
val_MPE_list = []
test_MPE_list = []

MPE_train_dict = {}
MPE_val_dict = {}
MPE_test_dict = {}

for metapath_type in MPE_dict:
    cur_feature = MPE_dict[metapath_type]

    MPE_train_dict[metapath_type] = cur_feature[:int(len(cur_feature) * 0.6)]
    MPE_val_dict[metapath_type] = cur_feature[int(len(cur_feature) * 0.6):int(len(cur_feature) * 0.8)]
    MPE_test_dict[metapath_type] = cur_feature[int(len(cur_feature) * 0.8):]

    train_MPE_list.extend([metapath_feature, metapath_type] for metapath_feature in MPE_train_dict[metapath_type])
    val_MPE_list.extend([metapath_feature, metapath_type] for metapath_feature in MPE_val_dict[metapath_type])
    test_MPE_list.extend([metapath_feature, metapath_type] for metapath_feature in MPE_test_dict[metapath_type])

model_MPE = MetapathEmbedding(trained_model, pre_embed_dim, 256).to(DEVICE)
trained_model_MPE = train_MPE(
    model_MPE,
    metapath_index_dict,
    val_num,
    epochs=num_epochs
)

test_loss_MPE, test_accuracy_MPE, test_f1_score_MPE = test_MPE(
    trained_model_MPE,
    MPE_test_dict,
    metapath_index_dict
)
print(f'Test Loss: {test_loss_MPE}, Test Accuracy: {test_accuracy_MPE}, Test F1 Score: {test_f1_score_MPE}')