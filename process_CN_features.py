import torch
from torch_geometric.data import HeteroData
import json

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
                if 'index' in current_paper:  # 保存上一篇论文
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
            if author.strip():  # 确保作者字段非空
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

    return papers, authors, venues, edges_cites, edges_written_by, edges_published_in


def build_hetero_graph(papers, authors, venues, edges_cites, edges_written_by, edges_published_in):
    """
    根据解析的数据构建 PyTorch Geometric 的异构图，并提取特征。
    """
    data = HeteroData()

    # === Paper 特征 ===
    title_lengths = torch.tensor([len(paper['title']) for paper in papers.values()], dtype=torch.float).unsqueeze(1)
    years = torch.tensor([paper['year'] if paper['year'] else 0 for paper in papers.values()], dtype=torch.float).unsqueeze(1)
    citation_counts = {paper_id: 0 for paper_id in papers}
    for src, dst in edges_cites:
        if dst in citation_counts:
            citation_counts[dst] += 1
    citations = torch.tensor([citation_counts[paper_id] for paper_id in papers], dtype=torch.float).unsqueeze(1)
    data['Paper'].x = torch.cat([title_lengths, years, citations], dim=1)

    # === Author 特征 ===
    author_paper_counts = {author_id: 0 for author_id in range(len(authors))}
    for _, author_id in edges_written_by:
        author_paper_counts[author_id] += 1
    author_features = torch.tensor([author_paper_counts[author_id] for author_id in range(len(authors))], dtype=torch.float).unsqueeze(1)
    data['Author'].x = author_features

    # === Venue 特征 ===
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


def save_graph_data(hetero_graph, output_dir):
    """
    保存异构图的节点、边和特征到文件。
    """
    # 保存节点特征
    torch.save(hetero_graph['Paper'].x, f"{output_dir}/paper_features.pt")
    torch.save(hetero_graph['Author'].x, f"{output_dir}/author_features.pt")
    torch.save(hetero_graph['Venue'].x, f"{output_dir}/venue_features.pt")

    # 保存边索引
    torch.save(hetero_graph['Paper', 'cites', 'Paper'].edge_index, f"{output_dir}/cites_edges.pt")
    torch.save(hetero_graph['Paper', 'written_by', 'Author'].edge_index, f"{output_dir}/written_by_edges.pt")
    torch.save(hetero_graph['Paper', 'published_in', 'Venue'].edge_index, f"{output_dir}/published_in_edges.pt")

    # 保存图的整体信息为 JSON
    graph_info = {
        "num_papers": hetero_graph['Paper'].num_nodes,
        "num_authors": hetero_graph['Author'].num_nodes,
        "num_venues": hetero_graph['Venue'].num_nodes,
        "num_cites_edges": hetero_graph['Paper', 'cites', 'Paper'].edge_index.shape[1],
        "num_written_by_edges": hetero_graph['Paper', 'written_by', 'Author'].edge_index.shape[1],
        "num_published_in_edges": hetero_graph['Paper', 'published_in', 'Venue'].edge_index.shape[1],
    }
    with open(f"{output_dir}/graph_info.json", "w") as f:
        json.dump(graph_info, f, indent=4)


if __name__ == "__main__":
    # 文件路径
    file_path = "/Users/willow/Desktop/Experiments/Hete/datasets/outputacm.txt"
    output_dir = "/Users/willow/Desktop/Experiments/Hete/datasets/output"

    # 解析文件
    papers, authors, venues, edges_cites, edges_written_by, edges_published_in = parse_acm_file(file_path)

    # 构建异构图
    hetero_graph = build_hetero_graph(papers, authors, venues, edges_cites, edges_written_by, edges_published_in)

    # 保存图数据
    save_graph_data(hetero_graph, output_dir)

    print("Graph data saved successfully!")