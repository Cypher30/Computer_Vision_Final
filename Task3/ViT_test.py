import torch
from utils import *
from models.res_mine import *
from models.alex_mine import *
from vit_pytorch import ViT
import argparse
import torch.nn as nn


parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, help='path to the stored model')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='Number of workers for dataloading')

args = parser.parse_args()

model = ViT(
	image_size = 32,
	patch_size = 2,
	num_classes = 100,
	dim = 768,
	depth = 16,
	heads = 12,
	mlp_dim = 1024,
	dropout = 0.1,
	emb_dropout = 0.1
)
model.load_state_dict(torch.load(args.src))


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)

train_data, test_data = dataloading(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], 
                                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]], 
                                    dataset='CIFAR100',
                                    batch_size=256,
                                    num_workers=args.workers)


criterion = nn.CrossEntropyLoss()
temp_loss, temp_correct = test(test_data, 
	                                    model, 
	                                    criterion, 
	                                    1)
