from .layers import ResUnetDecoder, ResUnetEncoder, ResUnetRegressionClassifier
from torch import nn

class Resunet(nn.Module):
    def __init__(self, in_depth, depths, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = ResUnetEncoder(in_depth, depths)
        self.decoder = ResUnetDecoder(depths)
        self.classifier = ResUnetRegressionClassifier(depths, 1)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.classifier(x)
        return x