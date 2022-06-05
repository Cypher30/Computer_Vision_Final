# Computer Vision (DATA130051) Final Projects

This is the final projects for Computer Vision (DATA130051)

Authors: [Zixuan Chen](https://github.com/403forbiddennn), [Yanjun Shao](https://github.com/super-dainiu), [Boyuan Yao](https://github.com/Cypher30/)

## Task1

## Task2
The code is the same with the mid-term Faster R-CNN code, and can be found [here](https://github.com/403forbiddennn/DATA130051-Computer-Vision/tree/main/midterm-faster-rcnn).

All the trained models can be downloaded [here](https://drive.google.com/drive/folders/1_R6Kr9MzCyASmfPM2FoUSkc37CcgpSWG?usp=sharing).

## Task3

In this task we train transformer-based models on CIFAR-100 from scratch, improving data efficiency by data augmentation, distillation mentioned in DeiT and use specially designed models.

To use our trained ViT models, you need to first download the models from [here](), and using ViT_test.py to load the models, e.g.

```bash
python ViT_test.py --src vit_baseline.pt # Indicates the source .pt file
```

To train ViT models, use ```python ViT_train.py --help``` to seek help, here is an example

```bash
python main.py --net_type vit # Net type: ViT or DeiT
			   --workers 16 # Number of workers to load images
			   -b 64 # Batchsize
			   --verbose # Verbose the training process
			   --lr 0.01 # Learning rate
			   --epochs 50 # Number of epochs for base phase
			   --model_save # Save model
			   --aug_type cutmix # Data augmentation types
			   --beta 10 # Hyperparameter for data augmentation
			   --aug_prob 0.5 # Augmentation probability
			   --save_path ./vit_cutmix.pt # Path to save the model
			   -tp ./log/vit_cutmix # Tensorboard log path
			   -tl /vit/ # Tensorboard label
			   --scheduler # Using Cosine Annealing scheduler
			   --restart 2 # Number of restart phaeses (N - 1)
			   --mult 2 # mult factor for number of epochs after each restart
			   --dim 768 # Dimension after embedding
			   --depth 16 # Number of transformer blocks
			   --heads 12 # Number of heads for multi-heads attention
			   --mlp_dim 1024 # MLP dimension
```

