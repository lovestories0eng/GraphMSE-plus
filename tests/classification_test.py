import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn.functional as F

# 模拟数据
np.random.seed(42)
torch.manual_seed(42)

num_instances = 10000
embedding_dim = 128
num_types = 8

# 随机生成实例 embedding 和标签
X = torch.tensor(np.random.randn(num_instances, embedding_dim), dtype=torch.float32)
y = torch.tensor(np.random.randint(0, num_types, num_instances), dtype=torch.long)

# 随机生成元路径类型 embedding（原型）
prototypes = torch.nn.Parameter(torch.tensor(np.random.randn(num_types, embedding_dim), dtype=torch.float32))

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 基于余弦相似度的分类器
def prototype_classification(X, prototypes):
    # 计算余弦相似度（X 的 shape：[batch_size, embedding_dim]）
    similarities = F.cosine_similarity(X.unsqueeze(1), prototypes.unsqueeze(0), dim=2)
    return similarities

# 训练模型
optimizer = torch.optim.Adam([prototypes], lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()

num_epochs = 300
for epoch in range(num_epochs):
    optimizer.zero_grad()
    similarities = prototype_classification(X_train, prototypes)
    loss = loss_fn(similarities, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 在测试集上评估
with torch.no_grad():
    similarities_test = prototype_classification(X_test, prototypes)
    y_pred = torch.argmax(similarities_test, dim=1)

accuracy = accuracy_score(y_test.numpy(), y_pred.numpy())
f1 = f1_score(y_test.numpy(), y_pred.numpy(), average='macro')

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Macro F1-Score: {f1:.4f}")
