import torch
from torch_geometric.data import HeteroData

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
    根据解析的数据构建 PyTorch Geometric 的异构图。
    """
    data = HeteroData()

    # 添加节点
    data['Paper'].x = torch.arange(len(papers)).unsqueeze(1).float()
    data['Author'].x = torch.arange(len(authors)).unsqueeze(1).float()
    data['Venue'].x = torch.arange(len(venues)).unsqueeze(1).float()

    # 添加边
    if edges_cites:
        data['Paper', 'cites', 'Paper'].edge_index = torch.tensor(edges_cites, dtype=torch.long).t().contiguous()
    if edges_written_by:
        data['Paper', 'written_by', 'Author'].edge_index = torch.tensor(edges_written_by, dtype=torch.long).t().contiguous()
    if edges_published_in:
        data['Paper', 'published_in', 'Venue'].edge_index = torch.tensor(edges_published_in, dtype=torch.long).t().contiguous()

    # 添加分类标签（以会议为标签）
    labels = torch.zeros(len(papers), dtype=torch.long)
    for paper_id, paper in papers.items():
        venue = paper['venue']
        if venue and venue in venues:
            labels[paper_id] = venues[venue]
    data['Paper'].y = labels

    # 添加训练/验证/测试掩码
    years = torch.tensor([paper['year'] if paper['year'] else 0 for paper in papers.values()])
    train_mask = years < 2000
    val_mask = (years >= 2000) & (years < 2010)
    test_mask = years >= 2010

    data['Paper'].train_mask = train_mask
    data['Paper'].val_mask = val_mask
    data['Paper'].test_mask = test_mask

    return data


if __name__ == "__main__":
    # 文件路径
    file_path = "/Users/willow/Desktop/Experiments/Hete/datasets/outputacm.txt"

    # 解析文件
    papers, authors, venues, edges_cites, edges_written_by, edges_published_in = parse_acm_file(file_path)

    # 调试信息
    print("Number of papers:", len(papers))
    print("Number of authors:", len(authors))
    print("Number of venues:", len(venues))
    print("Number of citation edges:", len(edges_cites))
    print("Number of written_by edges:", len(edges_written_by))
    print("Number of published_in edges:", len(edges_published_in))

    # 构建异构图
    hetero_graph = build_hetero_graph(papers, authors, venues, edges_cites, edges_written_by, edges_published_in)

    # 打印图信息
    print(hetero_graph)