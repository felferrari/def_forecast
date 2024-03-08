from .layers import ResUnetDecoder, ResUnetEncoder, ResUnetRegressionClassifier
from torch import nn

class Resunet(nn.Module):
    def __init__(self, in_depth, depths, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = ResUnetEncoder(in_depth, depths)
        self.decoder = ResUnetDecoder(depths)
        self.classifier = ResUnetRegressionClassifier(depths, 1, nn.ReLU)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.classifier(x)
        return x
    
class Mlp(nn.Module):
    def __init__(self, in_depth, layers, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mlp = nn.Sequential()
        self.mlp.append(nn.Linear(in_features=in_depth, out_features=layers[0]))
        self.mlp.append(nn.BatchNorm1d(num_features=layers[0]))
        self.mlp.append(nn.ReLU())
        for i in range(len(layers)-1):
            self.mlp.append(nn.Linear(in_features=layers[i], out_features=layers[i+1]))
            self.mlp.append(nn.BatchNorm1d(num_features=layers[i+1]))
            self.mlp.append(nn.ReLU())
            
        self.mlp.append(nn.Linear(in_features=layers[-1], out_features=1))
        self.mlp.append(nn.ReLU())
        
        
    def forward(self, x):
        x = self.mlp(x)
        return x