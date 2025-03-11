import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from store import GlobalStore

global_store = GlobalStore.get_instance(metapath_embedding_store=[])
DEVICE = global_store.DEVICE

class MetapathEncoder(nn.Module):
    def __init__(
        self, 
        metapath_types,
        node_embedding_dim,
        pre_embed_dim,
        mlp_settings, 
        ff_hid_dim, 
        num_heads=4,
        dropout=0.1
    ):
        super(MetapathEncoder, self).__init__()
        assert 'layer_list' in mlp_settings.keys()
        assert 'dropout_list' in mlp_settings.keys()
        assert 'activation' in mlp_settings.keys()
        assert mlp_settings['activation'] in ['sigmoid', 'relu', 'tanh']

        # 元路径列表
        self.metapath_types = metapath_types
        self.node_embedding_dim = node_embedding_dim
        self.pre_embed_dim = pre_embed_dim
        self.output_dim = len(self.metapath_types)
        self.mlp_settings = mlp_settings
        self.metapath_mlps = nn.ModuleDict()

        # Multi-head Attention
        self.attn = nn.MultiheadAttention(pre_embed_dim, num_heads, dropout=dropout)

        # Feed Forward
        self.ff = nn.Sequential(
            nn.Linear(pre_embed_dim, ff_hid_dim),
            nn.ReLU(),
            nn.Linear(ff_hid_dim, pre_embed_dim)
        )

        # Layer Normalization
        self.ln1 = nn.LayerNorm(pre_embed_dim)
        self.ln2 = nn.LayerNorm(pre_embed_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        for metapath in metapath_types:
            self.metapath_mlps[metapath] = self.build_mlp(len(metapath) * node_embedding_dim, pre_embed_dim)

        self.classify_layer = nn.Linear(pre_embed_dim, self.output_dim)

    def build_mlp(self, input_dim, pre_embed_dim):
        layers = []
        layer_dim = self.mlp_settings['layer_list']
        layers.append(nn.Linear(input_dim, layer_dim[0]))
        layers.append(self.get_activation(self.mlp_settings['activation']))
        cur = 1
        for i in range(len(layer_dim) - 1):
            layers.append(nn.Linear(layer_dim[i], layer_dim[i + 1]))
            if i < len(self.mlp_settings['dropout_list']):
                layers.append(nn.Dropout(self.mlp_settings['dropout_list'][i]))
            layers.append(self.get_activation(self.mlp_settings['activation']))
            cur += 1
        layers.append(nn.Linear(layer_dim[-1], pre_embed_dim))
        return nn.Sequential(*layers)

    def get_activation(self, activation_type):
        if activation_type == 'sigmoid':
            return nn.Sigmoid()
        elif activation_type == 'tanh':
            return nn.Tanh()
        elif activation_type == 'relu':
            return nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation type: {activation_type}")

    def metapath_aggregate(self, tensor):
        return tensor.mean(dim=0, keepdim=True)
    
    def forward(self, metapath_features_dict, metapath_index_dict):
        metapath_outputs = []
        metapath_labels = []

        all_metapath_embeddings = []
        embedding_labels = []

        for metapath_type, sampled_metapath_features in metapath_features_dict.items():
            cur_metapath_outputs = []

            for metapath_features in sampled_metapath_features:
                metapath_feature_tensor = torch.tensor(np.array(metapath_features), dtype=torch.float32).to(DEVICE)

                metapath_outputs_tensor = self.metapath_mlps[metapath_type](metapath_feature_tensor)
                
                # Multi-head Attention, Feed Forward, Layer Normalization, Dropout
                '''
                    在数据量少的时候，不建议使用 Multi-head Attention，因为会导致过拟合。
                    但是在数据量大的时候，可以使用，因为可以提高模型的泛化能力，减少过拟合，提高模型的准确率
                '''
                metapath_outputs_tensor, _ = self.attn(metapath_outputs_tensor, metapath_outputs_tensor, metapath_outputs_tensor)
                metapath_outputs_tensor = self.ln1(metapath_outputs_tensor)
                metapath_outputs_tensor = self.ff(metapath_outputs_tensor)
                metapath_outputs_tensor = self.ln2(metapath_outputs_tensor)
                metapath_outputs_tensor = self.dropout(metapath_outputs_tensor)
                cur_metapath_outputs.append(metapath_outputs_tensor)

                metapath_labels.append(metapath_index_dict[metapath_type])

            cur_metapath_embedding = [(self.metapath_aggregate(output)).flatten() for output in cur_metapath_outputs]
            cur_metapath_embedding = torch.stack(cur_metapath_embedding).to(DEVICE)
            all_metapath_embeddings.append(
                (self.metapath_aggregate(cur_metapath_embedding)).flatten()
            )
            embedding_labels.append(metapath_index_dict[metapath_type])
            metapath_outputs.extend(cur_metapath_embedding)

        metapath_outputs = torch.stack(metapath_outputs).to(DEVICE)
        metapath_labels = torch.tensor(metapath_labels).to(DEVICE)
        return metapath_outputs, metapath_labels, all_metapath_embeddings, embedding_labels, self.metapath_mlps

class MetapathClassifier(nn.Module):
    def __init__(
        self, 
        metapath_types, 
        mlp_settings, 
        node_embedding_dim, 
        metapath_embeddings, 
        embedding_labels, 
        use_embedding=False
    ):
        super(MetapathClassifier, self).__init__()

        self.metapath_types = metapath_types
        self.mlp_settings = mlp_settings
        self.node_embedding_dim = node_embedding_dim
        self.metapath_embeddings = metapath_embeddings
        self.embedding_labels = embedding_labels
        self.use_embedding = use_embedding

        self.metapath_mlps = nn.ModuleDict()
        self.embedding_dim = (self.metapath_embeddings[0]).shape[0]

        metapath_len_min = min(len(s) for s in metapath_types)
        metapath_len_max = max(len(s) for s in metapath_types)

        for metapath_len in range(metapath_len_min, metapath_len_max + 1):
            self.metapath_mlps[str(metapath_len)] = nn.Sequential(
                nn.Linear(metapath_len * node_embedding_dim, self.embedding_dim),
                nn.Sigmoid()
            )

        # instance embedding, metapath embedding fusion
        self.fusion_layer = nn.Linear(self.embedding_dim, self.embedding_dim)

        # Initialize the learnable attention weights
        self.attn = nn.Parameter(torch.randn(2)).to(DEVICE)  # random initialization

            
        # [metapath instance embedding, metapath embedding]
        self.classify_layer = nn.Linear(self.embedding_dim, len(self.embedding_labels))

        self.generated_metapath_embeddings = torch.rand(len(self.metapath_types), self.embedding_dim).to(DEVICE)
        nn.init.xavier_uniform_(self.generated_metapath_embeddings)

    def forward(self, metapath_features_dict, metapath_index_dict):
        metapath_outputs = []
        metapath_labels = []

        for metapath_type, metapath_features in metapath_features_dict.items():
            metapath_feature_tensor = torch.tensor(np.array(metapath_features), dtype=torch.float32).to(DEVICE)

            cur_metapath_len = len(metapath_type)
            metapath_outputs_tensor = self.metapath_mlps[str(cur_metapath_len)](metapath_feature_tensor)
            
            cur_label = metapath_index_dict[metapath_type]
            cur_num = len(metapath_features)

            index = self.embedding_labels.index(cur_label)
            if (self.use_embedding):
                cur_metapath_embedding = self.metapath_embeddings[index].detach()
                metapath_embedding_expanded = cur_metapath_embedding.unsqueeze(0).expand(cur_num, -1)
            else:
                cur_metapath_embedding = self.generated_metapath_embeddings[index]
                metapath_embedding_expanded = cur_metapath_embedding.unsqueeze(0).expand(cur_num, -1)

            attn_weights = torch.softmax(self.attn, dim=0)

            metapath_outputs_tensor = attn_weights[0] * metapath_outputs_tensor
            metapath_embedding_expanded = attn_weights[1] * metapath_embedding_expanded
            fused_outputs = self.fusion_layer(metapath_outputs_tensor + metapath_embedding_expanded)

            metapath_outputs.extend(fused_outputs)
            metapath_labels.extend([cur_label] * cur_num)

        metapath_outputs = torch.stack(metapath_outputs).to(DEVICE)
        metapath_labels = torch.tensor(metapath_labels).to(DEVICE)
        return self.classify_layer(metapath_outputs), metapath_labels