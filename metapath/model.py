import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class MetapathClassifier(nn.Module):
    def __init__(self, metapath_types, node_embedding_dim, pre_embed_dim, output_dim, mlp_settings):
        super(MetapathClassifier, self).__init__()
        assert 'layer_list' in mlp_settings.keys()
        assert 'dropout_list' in mlp_settings.keys()
        assert 'activation' in mlp_settings.keys()
        assert mlp_settings['activation'] in ['sigmoid', 'relu', 'tanh']

        # 元路径列表
        self.metapath_types = metapath_types
        self.node_embedding_dim = node_embedding_dim
        self.pre_embed_dim = pre_embed_dim
        self.output_dim = output_dim
        self.metapath_mlp = nn.ModuleDict()
        self.mlp_settings = mlp_settings

        for metapath in metapath_types:
            self.metapath_mlp[metapath] = self.build_mlp(len(metapath) * node_embedding_dim, pre_embed_dim)

        self.classify_layer = nn.Linear(pre_embed_dim, output_dim)

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

    def forward(self, metapath_feature_dict, metapath_index_dict):
        metapath_outputs = []
        metapath_labels = []

        for metapath_type, metapath_features in metapath_feature_dict.items():
            for metapath_feature in metapath_features:
                metapath_feature_tensor = torch.tensor(metapath_feature, dtype=torch.float32).to(DEVICE)
                metapath_output = self.metapath_mlp[metapath_type](metapath_feature_tensor)
                metapath_outputs.append(metapath_output)
                metapath_labels.append(metapath_index_dict[metapath_type])

            
        metapath_outputs_tensor = torch.stack(metapath_outputs).to(DEVICE)
        return self.classify_layer(metapath_outputs_tensor), metapath_labels
    
class MetapathEmbedding(nn.Module):
    def __init__(self, metapath_classifier, pre_embed_dim, ff_hid_dim, num_heads = 8, dropout=0.1):
        super(MetapathEmbedding, self).__init__()

        self.metapath_classifier = metapath_classifier
        self.metapath_mlp = self.metapath_classifier.metapath_mlp
        self.classify_layer = self.metapath_classifier.classify_layer

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

    def metapath_aggregate(self, tensor):
        return tensor.mean(dim=0, keepdim=True)


    def forward(self, metapath_feature_dict, metapath_index_dict):
        metapath_outputs = []
        metapath_labels = []

        for metapath_type, sampled_metapath_features in metapath_feature_dict.items():
            cur_metapath_outputs = []
            for metapath_features in sampled_metapath_features:
                metapath_feature_tensor = torch.tensor(metapath_features, dtype=torch.float32).to(DEVICE)
                cur_metapath_output = self.metapath_mlp[metapath_type](metapath_feature_tensor)
                cur_metapath_outputs.append(cur_metapath_output)
            metapath_outputs.append(self.metapath_aggregate(torch.stack(cur_metapath_outputs)))
            metapath_labels.append(metapath_index_dict[metapath_type])

        metapath_outputs_tensor = torch.stack(metapath_outputs).to(DEVICE)
        # (num_metapath, batch_size, embed_size)
        metapath_outputs_tensor = metapath_outputs_tensor.permute(1, 0, 2)
        metapath_outputs_tensor, _ = self.attn(metapath_outputs_tensor, metapath_outputs_tensor, metapath_outputs_tensor)
        metapath_outputs_tensor = self.ln1(metapath_outputs_tensor)
        metapath_outputs_tensor = self.ff(metapath_outputs_tensor)
        metapath_outputs_tensor = self.ln2(metapath_outputs_tensor)
        metapath_outputs_tensor = self.dropout(metapath_outputs_tensor)
        return self.classify_layer(metapath_outputs_tensor), metapath_labels