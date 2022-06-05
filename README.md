# Computer Vision (DATA130051) Final Projects

This is the final projects for Computer Vision (DATA130051)

Authors: [Zixuan Chen](https://github.com/403forbiddennn) [Yanjun Shao](https://github.com/super-dainiu) [Boyuan Yao](https://github.com/Cypher30/)

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

```
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

img_file = 'resources/example.png'
out_file = 'resources/output.avi'
skip = 10


# build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results

# img = mmcv.imread(img_file)

# result = inference_segmentor(model, img)
# model.show_result(img, result, out_file=out_file, opacity=0.5)

# test a video and show the results
video = mmcv.VideoReader('resources/video.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter(out_file, fourcc, video.fps, (video.width, video.height))
for i, frame in enumerate(tqdm.tqdm(video)):
    if i > 500:
        break
    result = inference_segmentor(model, frame)
    output = model.show_result(frame, result, opacity=0.5)
    writer.write(output)
writer.release() 
```



## Task2

## Task3