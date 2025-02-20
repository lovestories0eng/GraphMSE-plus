import numpy as np
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class MetapathClassifier(nn.Module):
    def __init__(self, metapath_types, node_embedding_dim, output_dim, mlp_settings):
        super(MetapathClassifier, self).__init__()
        assert 'layer_list' in mlp_settings.keys()
        assert 'dropout_list' in mlp_settings.keys()
        assert 'activation' in mlp_settings.keys()
        assert mlp_settings['activation'] in ['sigmoid', 'relu', 'tanh']

        # 元路径列表
        self.metapath_types = metapath_types
        self.node_embedding_dim = node_embedding_dim
        self.output_dim = output_dim
        self.metapath_mlp = nn.ModuleDict()
        self.mlp_settings = mlp_settings

        for metapath in metapath_types:
            self.metapath_mlp[metapath] = self.build_mlp(len(metapath) * node_embedding_dim, output_dim)

        self.classify_layer = nn.Linear(output_dim, len(metapath_types))

    def build_mlp(self, input_dim, output_dim):
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
        layers.append(nn.Linear(layer_dim[-1], output_dim))
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
                # metapath_output = self.metapath_mlp[metapath_type](metapath_feature)
                metapath_feature_tensor = torch.tensor(metapath_feature, dtype=torch.float32).to(DEVICE)
                metapath_output = self.metapath_mlp[metapath_type](metapath_feature_tensor)
                metapath_outputs.append(metapath_output)
                metapath_labels.append(metapath_index_dict[metapath_type])

            
        metapath_outputs_tensor = torch.stack(metapath_outputs).to(DEVICE)
        return self.classify_layer(metapath_outputs_tensor), metapath_labels