import torch

import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def cosine_similarity(A, B):
    # 计算余弦相似度
    dot_product = torch.dot(A, B)
    norm_A = torch.norm(A)
    # 存在 norm_B 为 0 的情况
    norm_B = torch.norm(B)
    return dot_product / ((norm_A + 1e-10) * (norm_B + 1e-10))

def cosine_similarity_stat(tensor):
    num_vectors = tensor.shape[0]
    total_similarity = 0
    count = 0
    sim_list = []
    
    # 两两比较
    for i in range(num_vectors):
        for j in range(i + 1, num_vectors):
            A = tensor[i]
            B = tensor[j]
            cur_sim = cosine_similarity(A, B)
            if (torch.isnan(cur_sim)):
                cosine_similarity(A, B)
                print(A, B)
            total_similarity += cur_sim
            sim_list.append(cur_sim)
            count += 1
    
    # 计算平均值
    return total_similarity / count, sim_list

# 定义一个简单的MLP
class MLP(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim):
    super(MLP, self).__init__()
    self.fc1 = nn.Linear(input_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, output_dim)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

# 测试MLP
def test_mlp():
  input_dim = 1256 * 4  # 输入维度
  hidden_dim = 128   # 隐藏层维度
  output_dim = 128    # 输出维度

  vectors = []
  np.random.seed()
  for i in range(128):
    # 创建一个稀疏向量
    sparse_vector = np.zeros(input_dim)
    # 在稀疏向量中随机设置一些值
    random_num = np.random.randint(1, input_dim)

    size = np.random.randint(1, input_dim + 1)
    random_index = np.random.choice(range(input_dim), size=size, replace=False)

    for index in random_index:
      sparse_vector[index] = np.random.uniform(-1, 1)

    vectors.append(torch.tensor(sparse_vector))
  
  vectors = torch.stack(vectors)
  sim, sim_list = cosine_similarity_stat(vectors)

  # 初始化MLP
  mlp = MLP(input_dim, hidden_dim, output_dim)

  vectors = vectors.to(torch.float64)
  mlp = mlp.to(torch.float64)

  # 将稀疏向量通过MLP
  output = mlp(vectors)

  sim, sim_list = cosine_similarity_stat(output)

  print("Output:", output)

if __name__ == "__main__":
  test_mlp()