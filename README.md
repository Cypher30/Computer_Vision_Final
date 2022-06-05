# Computer Vision (DATA130051) Final Projects

This is the final projects for Computer Vision (DATA130051)

Authors: [Zixuan Chen](https://github.com/403forbiddennn), [Yanjun Shao](https://github.com/super-dainiu), [Boyuan Yao](https://github.com/Cypher30/)

## Task1

You may also find these steps in the [official documents](https://mmsegmentation.readthedocs.io/en/latest/) of MMSegmentation.

#### Prerequisites

**Step 0.** Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim). (Installing mmcv-full may take about 10 mins)

```bash
pip install -U openmim
mim install mmcv-full
```

**Step 1.** Install [MMSegmentation](https://github.com/open-mmlab/mmsegmentation).

```bash
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -v -e .
```

#### Detect Videos

```python
from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import cv2
import tqdm

# DeepLabv3+
config_file = 'configs/deeplabv3plus/deeplabv3plus_r101-d8_fp16_512x1024_80k_cityscapes.py'
checkpoint_file = 'checkpoints/deeplabv3plus_r101-d8_fp16_512x1024_80k_cityscapes_20200717_230920-f1104f4b.pth'

# SETR
config_file = 'configs/setr/setr_vit-large_pup_8x1_768x768_80k_cityscapes.py'
checkpoint_file = 'checkpoints/setr_pup_vit-large_8x1_768x768_80k_cityscapes_20211122_155115-f6f37b8f.pth'

img_file = 'resources/video.mp4'
out_file = 'resources/output.avi'


# build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

# test a video and show the results
video = mmcv.VideoReader('resources/video.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter(out_file, fourcc, video.fps, (video.width, video.height))

for frame in tqdm.tqdm(video):
    result = inference_segmentor(model, frame)
    output = model.show_result(frame, result, opacity=0.5)
    writer.write(output)
writer.release() 
```

You can find the result video here.

>
链接：https://pan.baidu.com/s/1_EtihmU6v1Gqqs6pT8GwFg 
提取码：0216 
--来自百度网盘超级会员V5的分享

## Task2
The code is the same with the mid-term Faster R-CNN code, and can be found [here](https://github.com/403forbiddennn/DATA130051-Computer-Vision/tree/main/midterm-faster-rcnn).

All the trained models can be downloaded [here](https://drive.google.com/drive/folders/1_R6Kr9MzCyASmfPM2FoUSkc37CcgpSWG?usp=sharing).

## Task3

In this task we train transformer-based models on CIFAR-100 from scratch, improving data efficiency by data augmentation, distillation mentioned in DeiT and use specially designed models.

To use our trained ViT models, you need to first download the models from [here](https://pan.baidu.com/s/1OUfxi0aKknK-VZXkCVdXaA) (password: e6qt), and using ViT_test.py to load the models, e.g.

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
The above codes could be found in folder Task3. We also provide a teacher model ResNet-152 mentioned in our report for you to train your DeiT.

The extra work Swin Transformer with localization can be found [here](https://github.com/403forbiddennn/DATA130051-Computer-Vision/tree/master). We get the code from [this repo](https://github.com/yhlleo/VTs-Drloc) and applied some slight changes to it so that it can be adapted to our Pytorch version. The trained model can be found following this [link](https://drive.google.com/drive/folders/1_R6Kr9MzCyASmfPM2FoUSkc37CcgpSWG?usp=sharing).
