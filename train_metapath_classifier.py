import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

from torch.utils.data import Dataset


class MetaPathClassifier(nn.Module):
    def __init__(self, paper_dim, author_dim, venue_dim, hidden_dim, num_classes, paper_x, author_x, venue_x):
        """
        paper_dim, author_dim, venue_dim: 各类节点的特征维度
        hidden_dim: 统一投影维度
        num_classes: 分类类别（不同元路径类型）
        paper_x, author_x, venue_x: 各类节点的特征张量（用于索引）
        """
        
        
        super(MetaPathClassifier, self).__init__()

        # 存储节点特征（forward 时用索引访问）
        self.paper_x = paper_x
        self.author_x = author_x
        self.venue_x = venue_x
        
        # 三种类型节点的线性投影
        self.proj_paper = nn.Linear(paper_dim, hidden_dim)
        self.proj_author = nn.Linear(author_dim, hidden_dim)
        self.proj_venue = nn.Linear(venue_dim, hidden_dim)

        # 字典映射 node_type -> 对应的投影层
        self.domain_to_proj = {
            "Paper": self.proj_paper,
            "Author": self.proj_author,
            "Venue": self.proj_venue
        }
        self.domain_to_x = {
            "Paper": self.paper_x,
            "Author": self.author_x,
            "Venue": self.venue_x
        }

        # GRU 进行序列聚合
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)

        # 分类器
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, batch_paths):
        """
        batch_paths: list of [ (node_type, node_id), (node_type, node_id), ... ]
        """
        # 确保 node_id 在 x_all 范围内
        # if node_id >= x_all.shape[0]:
        #     print(f"Warning: node_id {node_id} out of bounds for {dom}, max index {x_all.shape[0] - 1}")
            # continue  # 跳过该路径，防止程序崩溃

        # x = x_all[node_id].to(DEVICE)  # 取特定节点的特征，并移动到 GPU
        seq_tensors = []
        for path in batch_paths:
            emb_list = []
            for (dom, node_id) in path:
                x_all = self.domain_to_x[dom]
                # x = x_all[node_id]  # 取特定节点的特征
                x = x_all[node_id].to(DEVICE) 
                proj = self.domain_to_proj[dom](x)  # 经过投影
                emb_list.append(proj)
            
            seq_tensors.append(torch.stack(emb_list, dim=0))

        # 使用 pack_sequence 处理变长序列
        packed = torch.nn.utils.rnn.pack_sequence(seq_tensors, enforce_sorted=False)

        # 送入 GRU
        _, h_n = self.gru(packed.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")))

        instance_emb = h_n.squeeze(0)  # 取最后时刻的 hidden state
        logits = self.classifier(instance_emb)  # 分类器
        return logits

def collate_fn(batch):
    """
    处理 DataLoader 读取的 batch，保证每个 batch 中的序列能够被正确处理
    batch: list of (path, label)
      - path: list of (node_type, node_id) tuples
      - label: int (metapath type label)
    
    返回:
    - paths: list of list of (dom, id) (仍然是 list of list 格式)
    - labels: tensor of shape [batch_size]
    """
    paths, labels = zip(*batch)  # tuple of list, tuple of int
    labels = torch.tensor(labels, dtype=torch.long)  # 转换为 tensor
    return paths, labels  # paths 仍然是 list of list，后续模型自己处理

class MetaPathInstanceDataset(Dataset):
    def __init__(self, paths, labels):
        """
        paths: list of [ (type, node_id), (type, node_id), ... ]
        labels: list of int (same length as paths)
        """
        self.paths = paths
        self.labels = labels
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        return self.paths[idx], self.labels[idx]

#######################################
# 1. 一些超参数与常量配置
#######################################
FEATURE_PATH = "./output_tf-idf"  # 节点特征文件所在目录
METAPATH_PATH = "./metapath_results_separate_length_3"  # 元路径实例结果所在目录

HIDDEN_DIM = 64
BATCH_SIZE = 16
NUM_EPOCHS = 5
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

#######################################
# 2. 加载节点特征
#######################################
def load_node_features(feature_path):
    print("\n[Step 1] Loading node features...")
    paper_x = torch.load(os.path.join(feature_path, "paper_features.pt"))  # [num_papers, Fp]
    author_x = torch.load(os.path.join(feature_path, "author_features.pt"))  # [num_authors, Fa]
    venue_x = torch.load(os.path.join(feature_path, "venue_features.pt"))  # [num_venues, Fv]
    
    print(f" - Paper features: {paper_x.shape}")
    print(f" - Author features: {author_x.shape}")
    print(f" - Venue features: {venue_x.shape}")
    return paper_x, author_x, venue_x

#######################################
# 3. 加载元路径实例
#######################################
def load_metapath_instances(metapath_path):
    print("\n[Step 2] Loading metapath instances...")
    with open(os.path.join(metapath_path, "paper_metapath_instances.pkl"), "rb") as f:
        paper_results = pickle.load(f)
    with open(os.path.join(metapath_path, "author_metapath_instances.pkl"), "rb") as f:
        author_results = pickle.load(f)
    with open(os.path.join(metapath_path, "venue_metapath_instances.pkl"), "rb") as f:
        venue_results = pickle.load(f)
    with open(os.path.join(metapath_path, "label_to_metapath.pkl"), "rb") as f:
        label_to_metapath = pickle.load(f)

    print(f" - Loaded {len(paper_results)} paper metapath instances")
    print(f" - Loaded {len(author_results)} author metapath instances")
    print(f" - Loaded {len(venue_results)} venue metapath instances")
    print(f" - Number of metapath types: {len(label_to_metapath)}")

    return paper_results, author_results, venue_results, label_to_metapath

#######################################
# 4. 解析元路径实例
#######################################
def parse_path_string(path_obj, type_seq):
    """
    解析元路径实例，支持元组 (1, 10, 2) 和字符串 "1-10-2"
    """
    if isinstance(path_obj, tuple):
        node_ids = list(path_obj)  # tuple 转 list
    elif isinstance(path_obj, str):
        node_ids = list(map(int, path_obj.split("-")))  # 字符串转整型 list
    else:
        raise ValueError(f"Unexpected path format: {path_obj}")

    assert len(node_ids) == len(type_seq), f"Mismatch in node count and type_seq: {node_ids} vs {type_seq}"
    
    return [(type_seq[i], node_ids[i]) for i in range(len(node_ids))]

def build_dataset_from_instances(paper_results, author_results, venue_results, label_to_metapath):
    """
    构造训练数据集
    """
    print("\n[Step 3] Building dataset from metapath instances...")

    all_paths = []
    all_labels = []

    def collect_data(result_dict):
        for node_id, label_dict in result_dict.items():
            for label, path_list in label_dict.items():
                _, type_seq = label_to_metapath[label]
                for ps in path_list:
                    path = parse_path_string(ps, type_seq)
                    all_paths.append(path)
                    all_labels.append(label)

    collect_data(paper_results)
    collect_data(author_results)
    collect_data(venue_results)

    print(f" - Total instances: {len(all_paths)}")
    max_id = max(max(paths) for paths in all_paths if len(paths) > 0)
    min_id = min(min(paths) for paths in all_paths if len(paths) > 0)
    print(f"Min node_id: {min_id}, Max node_id: {max_id}")
    
    # print(f"Paper feature shape: {paper_x.shape}")
    # print(f"Author feature shape: {author_x.shape}")
    # print(f"Venue feature shape: {venue_x.shape}")
    
    return all_paths, all_labels

#######################################
# 5. 训练与评估
#######################################
def train_one_epoch(model, optimizer, criterion, dataloader, epoch):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    print(f"\n[Training] Epoch {epoch}")
    for batch_idx, (batch_paths, batch_labels) in enumerate(dataloader):
        batch_labels = batch_labels.to(DEVICE)
        optimizer.zero_grad()

        logits = model(batch_paths)
        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch_labels)
        preds = logits.argmax(dim=1)
        correct = (preds == batch_labels).sum().item()
        total_correct += correct
        total_samples += len(batch_labels)

        if batch_idx % 10 == 0:
            print(f" - Batch {batch_idx}/{len(dataloader)}: Loss={loss.item():.4f}")

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    print(f" - Epoch {epoch} finished. Train Loss: {avg_loss:.4f}, Train Acc: {accuracy:.4f}")
    return avg_loss, accuracy

def evaluate(model, criterion, dataloader):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    print("\n[Evaluation] Running validation...")
    with torch.no_grad():
        for batch_paths, batch_labels in dataloader:
            batch_labels = batch_labels.to(DEVICE)
            logits = model(batch_paths)
            loss = criterion(logits, batch_labels)

            total_loss += loss.item() * len(batch_labels)
            preds = logits.argmax(dim=1)
            correct = (preds == batch_labels).sum().item()
            total_correct += correct
            total_samples += len(batch_labels)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    print(f" - Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy

#######################################
# 6. 主训练流程
#######################################
def main():
    # (1) 加载数据
    paper_x, author_x, venue_x = load_node_features(FEATURE_PATH)
    paper_results, author_results, venue_results, label_to_metapath = load_metapath_instances(METAPATH_PATH)
    num_classes = len(label_to_metapath)

    # (2) 解析元路径实例
    all_paths, all_labels = build_dataset_from_instances(paper_results, author_results, venue_results, label_to_metapath)

    # (3) 划分训练集和测试集
    dataset_size = len(all_paths)
    train_size = int(0.8 * dataset_size)
    train_dataset = MetaPathInstanceDataset(all_paths[:train_size], all_labels[:train_size])
    val_dataset = MetaPathInstanceDataset(all_paths[train_size:], all_labels[train_size:])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # (4) 训练模型
    # 获取各类节点的特征维度
    paper_dim = paper_x.shape[1]
    author_dim = author_x.shape[1]
    venue_dim = venue_x.shape[1]
    print("[Step 4] Moving features to device...")
    paper_x = paper_x.to(DEVICE)
    author_x = author_x.to(DEVICE)
    venue_x = venue_x.to(DEVICE)
    print(" - All features moved to", DEVICE)
    # 获取元路径类别数
    num_classes = len(label_to_metapath)

    # 正确初始化模型
    model = MetaPathClassifier(
        paper_dim, author_dim, venue_dim,  # 各类节点特征维度
        HIDDEN_DIM,  # 统一隐藏层维度
        num_classes,  # 分类类别数
        paper_x, author_x, venue_x  # 训练时用于查找的节点特征
    ).to(DEVICE)
    # model = MetaPathClassifier(...).to(DEVICE)
    print(f"Model is on: {next(model.parameters()).device}")
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, NUM_EPOCHS + 1):
        train_one_epoch(model, optimizer, criterion, train_loader, epoch)
        evaluate(model, criterion, val_loader)

if __name__ == "__main__":
    main()