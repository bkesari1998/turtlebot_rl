from torch import nn
import numpy as np

class Model(nn.Module):
    def __init__(self, output_size, output_std):
        super(Model,self).__init__()
        self.net = nn.Sequential(self.layer_init(
                nn.Conv2d(3,32,kernel_size=5,stride=1,padding=2)),#input:(heightxweight) 150x50x3, output: 150x50x32
                nn.ReLU(),
                nn.MaxPool2d(2,2),#output: 75x25
                self.layer_init(
                nn.Conv2d(32,64,kernel_size=5, stride=1,padding=2)),# output: 75x25x64
                nn.ReLU(),
                nn.MaxPool2d(2,2),#output: 38x13x64
                nn.Flatten(),#input: 38x13x64... output: 31_616
                self.layer_init(nn.Linear(40_000, 1024)),#
                nn.ReLU(),
                self.layer_init(nn.Linear(1024, 64)),
                nn.ReLU(),
                self.layer_init(nn.Linear(64, output_size), std=output_std)
            )
        self.frozen_layers = []

    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer


    def forward(self, x):
        # make sure frozen parameters are still frozen
        for i, layer in enumerate(self.net.children()):
            if i in self.frozen_layers:
                for param in layer.parameters():
                    param.requires_grad = False

        # run forward on the network
        return self.net.forward(x)


    # freeze weights of layers in list for transfer learning
    def freeze(self, layers):
        self.frozen_layers = set(layers)

    # re-initialize weights of layers in list
    def reset(self, layers):
        layers = set(layers)
        for i, layer in enumerate(self.net.children()):
            if i in layers:
                self.layer_init(layer)

