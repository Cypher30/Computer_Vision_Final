from turtle import forward
from grpc import intercept_channel
from numpy import isin
import torch
import torch.nn as nn
import math


class AlexNet_mine(nn.Module):
    def __init__(self, num_classes: int = 100, dropout: float = 0.5, localnorm=False) -> None:
        super().__init__()
        if localnorm:
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),  # Output: N * 96 * 55 * 55
                nn.ReLU(inplace=True),
                nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
                nn.MaxPool2d(kernel_size=3, stride=2),  # Output N * 96 * 27 * 27
                nn.Conv2d(96, 256, kernel_size=5, padding=2), # Output: N * 256 * 27 * 27
                nn.ReLU(inplace=True),
                nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
                nn.MaxPool2d(kernel_size=3, stride=2), # Output N * 256 * 13 * 13
                nn.Conv2d(256, 384, 3, padding=1), # Output N * 384 * 13 * 13
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 384, 3, padding=1), # Output N * 384 * 13 * 13
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, 3, padding=1), # Output N * 256 * 13 * 13
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2) # Output N * 256 * 6 * 6 
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),  # Output: N * 96 * 55 * 55
			    nn.BatchNorm2d(96),
                nn.ReLU(inplace=True),
                #nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
                nn.MaxPool2d(kernel_size=3, stride=2),  # Output N * 96 * 27 * 27
                nn.Conv2d(96, 256, kernel_size=5, padding=2), # Output: N * 256 * 27 * 27
		    	nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                #nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
                nn.MaxPool2d(kernel_size=3, stride=2), # Output N * 256 * 13 * 13
                nn.Conv2d(256, 384, 3, padding=1), # Output N * 384 * 13 * 13
		    	nn.BatchNorm2d(384),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 384, 3, padding=1), # Output N * 384 * 13 * 13
			    nn.BatchNorm2d(384),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, 3, padding=1), # Output N * 256 * 13 * 13
		    	nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2) # Output N * 256 * 6 * 6 
            )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes)
        )

        self.init_weight()

    def init_weight(self):
        for layer in self.features:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight.data, nonlinearity='relu')
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight.data, nonlinearity='relu')
        # if localnorm:
        #     for layer in self.features:
        #         if isinstance(layer, nn.Conv2d):
        #             nn.init.normal_(layer.weight, mean=0, std=0.01)
        #             nn.init.constant_(layer.bias, 0)
        #     nn.init.constant_(self.features[4].bias, 1)
        #     nn.init.constant_(self.features[10].bias, 1)
        #     nn.init.constant_(self.features[12].bias, 1)
        # else:   
        #     for layer in self.features:
        #         if isinstance(layer, nn.Conv2d):
        #             nn.init.normal_(layer.weight, mean=0, std=0.01)
        #             nn.init.constant_(layer.bias, 0)
        #     nn.init.constant_(self.features[4].bias, 1)
        #     nn.init.constant_(self.features[11].bias, 1)
        #     nn.init.constant_(self.features[14].bias, 1)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
        
