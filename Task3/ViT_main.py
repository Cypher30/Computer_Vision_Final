import argparse
from ast import parse
from ctypes import resize
from curses import meta
import os
import shutil
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.transforms.functional import InterpolationMode
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from utils import *
import numpy as np
import copy
import warnings
from torchsummary import summary
from vit_pytorch import ViT, MAE
from vit_pytorch.distill import DistillableViT, DistillWrapper
from models.res_mine import *

parser = argparse.ArgumentParser(description='Data augmentation of CIFAR100,\
     CIFAR10 with AlexNet and Resnet')
parser.add_argument('--dataset', default='CIFAR100', type=str,
                    help='dataset: CIFAR100, CIFAR10')
parser.add_argument('--net_type', default='vit', type=str,
                    help='networktype: vit, mae, dit')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--num_classes', default=100, type=int, metavar='N',
                    help='number of the classes')
parser.add_argument('--verbose', action='store_true', 
                    help='print the status at the end of every epoch')
parser.add_argument('--lr', default=0.1, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N',
                    help='mini-batch size (default: 64)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of epochs per restart(default 300)')
parser.add_argument('-m', '--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for SGD momentum (default: 0.9)')
parser.add_argument('--weight_decay', default=1e-4, type=float, metavar='W',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--beta', default=0, type=float, 
                    help='hyperparameter beta')
parser.add_argument('--aug_prob', default=0, type=float,
                    help='Data augmentation probability')
parser.add_argument('--aug_type', default='cutmix', type=str,
                    help='data augmentation type: cutmix, mixup, cutout')
parser.add_argument('--model_save', action='store_true',
                    help='store the best model')
parser.add_argument('--save_path', default='./model.pt', type=str,
                    help='path to store the best model')
parser.add_argument('-tp', '--tensorboard_path', default='', type=str,
                    help='tensorboard writer path, store in train and test')
parser.add_argument('-tl', '--tensorboard_label', default='', type=str,
                    help='tensorboard label, Loss and Acc')
parser.add_argument('--scheduler', action='store_true',
                    help='Use cosine scheduler for lr decay')
parser.add_argument('--restart', default=1, type=int, metavar='N',
					help='number of restart phases')
parser.add_argument('--mult', default=1, type=int, metavar='N', 
					help="mult factor of number of epochs after each restart")
parser.add_argument('--dim', default=768, type=int, metavar='N',
					help='Last dimension of output tensor after linear transformation')
parser.add_argument('--depth', default=12, type=int, metavar='N',
					help='Number of transformer blocks')
parser.add_argument('--heads', default=6, type=int, metavar='N',
					help='Number of heads in Multi-head Attention layer')
parser.add_argument('--mlp_dim', default=1024, type=int, metavar='N',
					help='dimension of the MLP layer')
parser.add_argument('--teacher', default=None, type=str,
					help='Path to teacher model for dit')

args = parser.parse_args()

train_writer = SummaryWriter(args.tensorboard_path + '/train')
test_writer = SummaryWriter(args.tensorboard_path + '/test')

if args.net_type == 'dit' or args.net_type == 'vit':
	train_data, test_data = dataloading(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], 
                                        std=[x / 255.0 for x in [63.0, 62.1, 66.7]], 
                                        dataset=args.dataset,
                                        batch_size=args.batch_size, 
                                        num_workers=args.workers)
else:										
	train_data, test_data = dataloading(mean=[0.4914, 0.4822, 0.4465], 
                                    	std=[0.247, 0.2435, 0.2616],
                                    	dataset=args.dataset,
                                    	batch_size=args.batch_size, 
                                    	num_workers=args.workers, 
                                    	resize=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if args.net_type == 'vit':
	model = ViT(
		image_size = 32,
		patch_size = 2,
		num_classes = args.num_classes,
		dim = args.dim,
		depth = args.depth,
		heads = args.heads,
		mlp_dim = args.mlp_dim,
		dropout = 0.1,
		emb_dropout = 0.1
	)
# elif args.net_type == 'mae':
# 	v = ViT(
# 		image_size = 224,
# 		patch_size = 16,
# 		num_classes = args.num_classes,
# 		dim = 768,
# 		depth = 16,
# 		heads = 12,
# 		mlp_dim = 1024,
# 	)
	
# 	model = MAE(
# 		encoder=v,
# 		masking_ratio=0.75,
# 		decoder_dim = 512,
# 		decoder_depth = 6
# 	)
elif args.net_type == 'dit':
	v = DistillableViT(
		image_size = 32,
		patch_size = 2,
		num_classes = args.num_classes,
		dim = args.dim,
		depth = args.depth,
		heads = args.heads,
		mlp_dim = args.mlp_dim,
		dropout = 0.1,
		emb_dropout = 0.1
	)
	# teacher = ResNet_test(Bottleneck, [3, 4, 23, 3], num_classes=args.num_classes)	
	teacher = ResNet_test(Bottleneck, [3, 8, 36, 3], num_classes=args.num_classes)
	teacher.load_state_dict(torch.load(args.teacher))	
	model = DistillWrapper(
		student=v,
		teacher=teacher,
		temperature=4,
		alpha=0.5,
		hard=False
	)
	

net = model.cuda()
if args.net_type == 'dit':
	summary(model.student, input_size=(3, 32, 32))
else:
	summary(model, input_size=(3, 32, 32))

criterion = nn.CrossEntropyLoss().cuda()
#optimizer = optim.Adam(net.parameters(), lr = lr)
#optimizer = optim.Adam(net.parameters(), lr = lr, weight_decay=0.0005)
# train_loss, test_loss, train_acc, test_acc = [], [], [], []
best_acc = 0
epoch_counter = 0
for j in range(args.restart):
	print("===Start of %d restart phase===" % (j + 1))
	if args.net_type == "mae":
		optimizer = optim.AdamW(net.parameters(),
								lr=args.lr,
								weight_decay=args.weight_decay,
								betas=(0.9, 0.999)
								)
	else:
		optimizer = optim.SGD(net.parameters(),
                      	momentum=args.momentum, 
                      	lr = args.lr, 
                      	weight_decay=args.weight_decay, 
                      	nesterov=True)
	# cudnn.benchmark = True 
	if args.scheduler:
		scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * (args.mult ** j))
	for i in range(args.epochs * (args.mult ** j)):
		if not args.scheduler:
			if i == args.epochs // 2 or i == args.epochs * 3 // 4:
				for param_group in optimizer.param_groups:
					param_group['lr'] = param_group['lr'] * 0.1
	
		# print('Mock(1)')
		for param_group in optimizer.param_groups:
			print(param_group['lr'])
	
		# if args.net_type == 'mae':
		# 	temp_loss, temp_correct = train(train_data, 
	    #                                 	net, 
	    #                                 	criterion, 
	    #                                 	optimizer, 
	    #                                 	epoch_counter + 1, 
	    #                                 	args.aug_prob, 
	    #                                 	args.beta, 
	    #                                 	aug_type=args.aug_type, 
	    #                                 	verbose=args.verbose,
		# 									mae=True
		# 									)	
		# else:
		temp_loss, temp_correct = train(train_data, 
	                                    net, 
	                                    criterion, 
	                                    optimizer, 
	                                    epoch_counter + 1, 
	                                    args.aug_prob, 
	                                    args.beta, 
	                                    aug_type=args.aug_type, 
	                                    verbose=args.verbose,
										dit=(args.net_type == "dit"))	
		# train_loss.append((temp_loss/len(train_data.dataset)).item())
		# train_acc.append((temp_correct/len(train_data.dataset)).item())
		train_writer.add_scalar(args.tensorboard_label + 'Loss', temp_loss, epoch_counter + 1)
		train_writer.add_scalar(args.tensorboard_label + 'Acc', temp_correct, epoch_counter + 1)
		# print(net.offset.grad)
		# print(net.mask.grad)
		# if args.net_type == 'mae':
		# 	temp_loss, temp_correct = test(test_data, 
	    #                                 	net, 
	    #                                 	criterion, 
	    #                                 	epoch_counter + 1,
	    #                                 	verbose=args.verbose,
		# 									mae=True
		# 									)
		# else:
		temp_loss, temp_correct = test(test_data, 
	                                   net, 
	                                   criterion, 
	                                   epoch_counter + 1,
	                                   verbose=args.verbose,
									   dit=(args.net_type == "dit"))
		# test_loss.append((temp_loss/len(test_data)).item())
		# test_acc.append((temp_correct/10000).item())
		test_writer.add_scalar(args.tensorboard_label + 'Loss', temp_loss, epoch_counter + 1)
		test_writer.add_scalar(args.tensorboard_label + 'Acc', temp_correct, epoch_counter + 1)
	    
		if best_acc < temp_correct:
			best_acc = temp_correct
			if args.model_save:
				if args.net_type == "dit":
					model_cache = copy.deepcopy(net.student.state_dict())
				else:		
					model_cache = copy.deepcopy(net.state_dict())
	
		if args.scheduler:
			scheduler.step()	
		
		epoch_counter = epoch_counter + 1
	print("===End of %d restart phase===" % (j + 1))
 
if args.model_save:
	torch.save(model_cache, args.save_path)
print("Best accuracy on testing instances: %.2f" % best_acc)
