from colossalai.amp import AMP_TYPE

# CvT Base
BATCH_SIZE = 64
DROP_RATE = 0.1
NUM_EPOCHS = 10

# mix precision
fp16 = dict(
    mode=AMP_TYPE.TORCH,
)

gradient_accumulation = 16
clip_grad_norm = 0.1

dali = dict(
    gpu_aug=True,
    mixup_alpha=0.2
)

tf_dir = '../../tf-logs/CvT'
log_dir = './log/history.log'