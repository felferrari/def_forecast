from torch import nn
import torch

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
        
        
    def forward(self, x_dict):
        x = torch.cat([x_dict[x_k] for x_k in x_dict], dim = -1)
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
        
        
    def forward(self, x_dict):
        x = torch.cat([x_dict[x_k] for x_k in x_dict], dim = -1)
        x = self.mlp(x)
        return x