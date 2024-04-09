from torch import nn
import torch
from .layers import TransformerDecoder

class Identity(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer = nn.Linear(1,1)
        
    def forward(self, x):
        return x['ArDS']

class MlpClasifier(nn.Module):
    def __init__(self, layers, input_sample, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        input_size = 0
        for k in input_sample[0]:
            input_size += len(input_sample[0][k])
        self.mlp = nn.Sequential()
        self.mlp.append(nn.Linear(in_features=input_size, out_features=layers[0]))
        self.mlp.append(nn.BatchNorm1d(num_features=layers[0]))
        self.mlp.append(nn.ReLU())
        for i in range(len(layers)-1):
            self.mlp.append(nn.Linear(in_features=layers[i], out_features=layers[i+1]))
            self.mlp.append(nn.BatchNorm1d(num_features=layers[i+1]))
            self.mlp.append(nn.ReLU())
            
        self.mlp.append(nn.Linear(in_features=layers[-1], out_features=2))
        #self.mlp.append(nn.ReLU())
        
    def prepare_input(self, x_dict):
        return torch.cat([x_dict[x_k] for x_k in x_dict], dim = -1)    
    
    def forward(self, x):
        #x = torch.cat([x_dict[x_k] for x_k in x_dict], dim = -1)
        x = self.mlp(x)
        return x
    
class MlpRegression(nn.Module):
    def __init__(self, layers, input_sample, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        input_size = 0
        for k in input_sample[0]:
            input_size += len(input_sample[0][k])
        self.mlp = nn.Sequential()
        self.mlp.append(nn.Linear(in_features=input_size, out_features=layers[0]))
        self.mlp.append(nn.BatchNorm1d(num_features=layers[0]))
        self.mlp.append(nn.ReLU())
        for i in range(len(layers)-1):
            self.mlp.append(nn.Linear(in_features=layers[i], out_features=layers[i+1]))
            self.mlp.append(nn.BatchNorm1d(num_features=layers[i+1]))
            self.mlp.append(nn.ReLU())
            
        self.mlp.append(nn.Linear(in_features=layers[-1], out_features=1))
        self.mlp.append(nn.ReLU())
        
    def prepare_input(self, x_dict):
        return torch.cat([x_dict[x_k] for x_k in x_dict], dim = -1)
        
    def forward(self, x):
        #x = torch.cat([x_dict[x_k] for x_k in x_dict], dim = -1)
        x = self.mlp(x)
        return x
    
class TransformerRegression(nn.Module):
    def __init__(self, input_sample, n_layers, d_model, n_head, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        in_dim = 0
        for sample_k in input_sample[0]:
            in_dim += input_sample[0][sample_k].shape[0]
        self.proj = nn.Linear(in_features=in_dim, out_features=d_model)
        self.trans_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                batch_first = True,
                activation = 'gelu'
            )
            for _ in range(n_layers)
        ])
        self.decoder = TransformerDecoder(in_dim=d_model, out_dim=1)
        self.last_activation = nn.ReLU()
        
    def prepare_input(self, x_dict):
        return torch.cat([x_dict[x_k] for x_k in x_dict], dim = -1)
        
    def forward(self, x):
        #x = torch.cat([x_dict[x_k] for x_k in x_dict], dim = -1)
        x = self.proj(x)
        x = torch.unsqueeze(x, -2)
        for layer in self.trans_layers:
            x = layer(x)
        x = self.decoder(x)
        x = self.last_activation(x)
        x = torch.squeeze(x, dim=-1)
        return x