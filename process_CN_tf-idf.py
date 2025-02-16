import torch
from torch_geometric.data import HeteroData
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import json
import os

# === 数据解析部分 ===

def parse_acm_file(file_path):
    """
    解析 ACM 数据集，生成论文、作者、会议以及边的关系。
    """
    papers = {}
    authors = {}
    venues = {}
    edges_cites = []
    edges_written_by = []
    edges_published_in = []

    with open(file_path, 'r', encoding='latin-1') as f:
        current_paper = {}
        for line in f:
            line = line.strip()
            if line.startswith("#*"):  # Paper title
                if 'index' in current_paper:
                    papers[current_paper['index']] = current_paper
                current_paper = {"title": line[2:], "authors": [], "citations": [], "venue": "", "year": None}
            elif line.startswith("#@"):  # Authors
                authors_list = line[2:].split(",") if line[2:] else []
                current_paper['authors'] = [author.strip() for author in authors_list]
            elif line.startswith("#t"):  # Year
                current_paper['year'] = int(line[2:]) if line[2:].isdigit() else None
            elif line.startswith("#c"):  # Venue
                current_paper['venue'] = line[2:].strip()
            elif line.startswith("#index"):  # Paper index
                current_paper['index'] = int(line[6:])
            elif line.startswith("#%"):  # Citations
                citation_id = line[2:].strip()
                if citation_id.isdigit():
                    current_paper['citations'].append(int(citation_id))
        # 保存最后一篇论文
        if 'index' in current_paper:
            papers[current_paper['index']] = current_paper

    # 创建作者和会议的字典
    for paper_id, paper in papers.items():
        # 添加 written_by 边
        for author in paper['authors']:
            if author.strip():
                if author not in authors:
                    authors[author] = len(authors)
                edges_written_by.append((paper_id, authors[author]))
        # 添加 published_in 边
        if paper['venue']:
            if paper['venue'] not in venues:
                venues[paper['venue']] = len(venues)
            edges_published_in.append((paper_id, venues[paper['venue']]))
        # 添加 citation 边
        for cited_paper in paper['citations']:
            edges_cites.append((paper_id, cited_paper))

    # 打印解析结果
    print(f"Total papers parsed: {len(papers)}")
    print(f"Total authors parsed: {len(authors)}")
    print(f"Total venues parsed: {len(venues)}")
    print(f"Total citation edges: {len(edges_cites)}")
    print(f"Total written_by edges: {len(edges_written_by)}")
    print(f"Total published_in edges: {len(edges_published_in)}")

    return papers, authors, venues, edges_cites, edges_written_by, edges_published_in

# def parse_acm_file(file_path):
#     """
#     解析 ACM 数据集，生成论文、作者、会议以及边的关系。
#     """
#     papers = {}
#     authors = {}
#     venues = {}
#     edges_cites = []
#     edges_written_by = []
#     edges_published_in = []

#     with open(file_path, 'r', encoding='latin-1') as f:
#         current_paper = {}
#         for line in f:
#             line = line.strip()
#             if line.startswith("#*"):  # Paper title
#                 if 'index' in current_paper:
#                     papers[current_paper['index']] = current_paper
#                 current_paper = {"title": line[2:], "authors": [], "citations": [], "venue": "", "year": None}
#             elif line.startswith("#@"):  # Authors
#                 authors_list = line[2:].split(",") if line[2:] else []
#                 current_paper['authors'] = [author.strip() for author in authors_list]
#             elif line.startswith("#t"):  # Year
#                 current_paper['year'] = int(line[2:]) if line[2:].isdigit() else None
#             elif line.startswith("#c"):  # Venue
#                 current_paper['venue'] = line[2:].strip()
#             elif line.startswith("#index"):  # Paper index
#                 current_paper['index'] = int(line[6:])
#             elif line.startswith("#%"):  # Citations
#                 citation_id = line[2:].strip()
#                 if citation_id.isdigit():
#                     current_paper['citations'].append(int(citation_id))
#         # 保存最后一篇论文
#         if 'index' in current_paper:
#             papers[current_paper['index']] = current_paper

#     # 创建作者和会议的字典
#     for paper_id, paper in papers.items():
#         # 添加 written_by 边
#         for author in paper['authors']:
#             if author.strip():
#                 if author not in authors:
#                     authors[author] = len(authors)
#                 edges_written_by.append((paper_id, authors[author]))
#         # 添加 published_in 边
#         if paper['venue']:
#             if paper['venue'] not in venues:
#                 venues[paper['venue']] = len(venues)
#             edges_published_in.append((paper_id, venues[paper['venue']]))
#         # 添加 citation 边
#         for cited_paper in paper['citations']:
#             edges_cites.append((paper_id, cited_paper))

#     return papers, authors, venues, edges_cites, edges_written_by, edges_published_in

# === 论文特征提取部分 ===
def extract_tfidf_features(titles, max_features=100):
    """
    使用 TF-IDF 提取论文标题的特征。
    """
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(titles)
    return tfidf_matrix.toarray()

def extract_paper_features(papers, edges_cites, max_features=100):
    """
    为论文节点提取特征，包括标题的 TF-IDF 特征。
    """
    titles = [paper['title'] for paper in papers.values()]
    tfidf_features = extract_tfidf_features(titles, max_features=max_features)
    
    # 提取标题长度
    title_lengths = np.array([len(title) for title in titles]).reshape(-1, 1)
    
    # 提取发表年份
    years = np.array([paper['year'] if paper['year'] else 0 for paper in papers.values()]).reshape(-1, 1)
    
    # 计算被引用次数
    citation_counts = {paper_id: 0 for paper_id in papers}
    for src, dst in edges_cites:
        if dst in citation_counts:
            citation_counts[dst] += 1
    citations = np.array([citation_counts[paper_id] for paper_id in papers]).reshape(-1, 1)
    
    # 合并所有特征
    paper_features = np.hstack([tfidf_features, title_lengths, years, citations])
    return torch.tensor(paper_features, dtype=torch.float)

# === 异构图构建部分 ===
def build_hetero_graph(papers, authors, venues, edges_cites, edges_written_by, edges_published_in, max_features=100):
    """
    构建 PyTorch Geometric 的异构图，并提取节点特征。
    """
    data = HeteroData()
    
    # 提取 Paper 节点特征
    data['Paper'].x = extract_paper_features(papers, edges_cites, max_features=max_features)
    
    # 提取 Author 节点特征
    author_paper_counts = {author_id: 0 for author_id in range(len(authors))}
    for _, author_id in edges_written_by:
        author_paper_counts[author_id] += 1
    author_features = torch.tensor([author_paper_counts[author_id] for author_id in range(len(authors))], dtype=torch.float).unsqueeze(1)
    data['Author'].x = author_features

    # 提取 Venue 节点特征
    venue_paper_counts = {venue_id: 0 for venue_id in range(len(venues))}
    for _, venue_id in edges_published_in:
        venue_paper_counts[venue_id] += 1
    venue_features = torch.tensor([venue_paper_counts[venue_id] for venue_id in range(len(venues))], dtype=torch.float).unsqueeze(1)
    data['Venue'].x = venue_features

    # 添加边
    if edges_cites:
        data['Paper', 'cites', 'Paper'].edge_index = torch.tensor(edges_cites, dtype=torch.long).t().contiguous()
    if edges_written_by:
        data['Paper', 'written_by', 'Author'].edge_index = torch.tensor(edges_written_by, dtype=torch.long).t().contiguous()
    if edges_published_in:
        data['Paper', 'published_in', 'Venue'].edge_index = torch.tensor(edges_published_in, dtype=torch.long).t().contiguous()

    return data

# === 数据保存部分 ===
def save_graph_data(graph_data, output_dir):
    """
    保存异构图的节点、边和特征到文件。
    """
    for node_type in graph_data.node_types:
        torch.save(graph_data[node_type].x, f"{output_dir}/{node_type.lower()}_features.pt")
    for edge_type in graph_data.edge_types:
        edge_index = graph_data[edge_type].edge_index
        torch.save(edge_index, f"{output_dir}/{edge_type[1]}_edges.pt")
    print("Graph data saved successfully!")

# === 主程序 ===
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "outputacm.txt")
    output_dir = os.path.join(current_dir, "output_tf-idf")
    # /Users/willow/Desktop/Experiments/Hete/datasets/output_tf-idf
    max_features = 100  # 设置 TF-IDF 的特征维度

    # 解析数据
    papers, authors, venues, edges_cites, edges_written_by, edges_published_in = parse_acm_file(file_path)

    # 构建异构图
    graph_data = build_hetero_graph(papers, authors, venues, edges_cites, edges_written_by, edges_published_in, max_features=max_features)

    # 保存数据
    save_graph_data(graph_data, output_dir)