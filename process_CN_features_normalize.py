import torch
from sklearn.preprocessing import StandardScaler

def load_graph_data(output_dir):
    """
    从保存的文件中加载节点特征和边索引。
    """
    graph_data = {
        "Paper": {
            "features": torch.load(f"{output_dir}/paper_features.pt")
        },
        "Author": {
            "features": torch.load(f"{output_dir}/author_features.pt")
        },
        "Venue": {
            "features": torch.load(f"{output_dir}/venue_features.pt")
        },
        "edges": {
            "cites": torch.load(f"{output_dir}/cites_edges.pt"),
            "written_by": torch.load(f"{output_dir}/written_by_edges.pt"),
            "published_in": torch.load(f"{output_dir}/published_in_edges.pt")
        }
    }
    return graph_data

def normalize_features(features):
    """
    使用 Z-Score 标准化特征
    """
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    return torch.tensor(normalized_features, dtype=torch.float)

def process_and_normalize_graph(graph_data):
    """
    对所有节点特征进行标准化处理。
    """
    # 处理 Paper 节点特征
    if "Paper" in graph_data:
        print("Standardizing Paper features...")
        paper_features = graph_data["Paper"]["features"].numpy()
        graph_data["Paper"]["features"] = normalize_features(paper_features)

    # 处理 Author 节点特征
    if "Author" in graph_data:
        print("Standardizing Author features...")
        author_features = graph_data["Author"]["features"].numpy()
        graph_data["Author"]["features"] = normalize_features(author_features)

    # 处理 Venue 节点特征
    if "Venue" in graph_data:
        print("Standardizing Venue features...")
        venue_features = graph_data["Venue"]["features"].numpy()
        graph_data["Venue"]["features"] = normalize_features(venue_features)

    return graph_data

def save_normalized_graph(graph_data, output_dir):
    """
    保存标准化后的图数据到文件。
    """
    # 保存节点特征
    torch.save(graph_data["Paper"]["features"], f"{output_dir}/paper_features_normalized.pt")
    torch.save(graph_data["Author"]["features"], f"{output_dir}/author_features_normalized.pt")
    torch.save(graph_data["Venue"]["features"], f"{output_dir}/venue_features_normalized.pt")

    # 保存边索引（保持不变）
    torch.save(graph_data["edges"]["cites"], f"{output_dir}/cites_edges.pt")
    torch.save(graph_data["edges"]["written_by"], f"{output_dir}/written_by_edges.pt")
    torch.save(graph_data["edges"]["published_in"], f"{output_dir}/published_in_edges.pt")

    print("Normalized graph data saved successfully!")

if __name__ == "__main__":
    # 输入和输出文件夹路径
    input_dir = "/Users/willow/Desktop/Experiments/Hete/datasets/output"  # 保存的原始图数据文件夹
    output_dir = "/Users/willow/Desktop/Experiments/Hete/datasets/output_normalized"  # 保存标准化后的图数据

    # 加载图数据
    graph_data = load_graph_data(input_dir)

    # 标准化节点特征
    graph_data = process_and_normalize_graph(graph_data)

    # 保存标准化后的图数据
    save_normalized_graph(graph_data, output_dir)