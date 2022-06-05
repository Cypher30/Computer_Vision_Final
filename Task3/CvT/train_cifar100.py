import os

import colossalai
import torch
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.lr_scheduler import LinearWarmupLR
from colossalai.nn.metric import Accuracy
from colossalai.trainer import Trainer, hooks
from model import CvT, CvT_config
# from cvt import CvT
import config
import torch.optim as optim
from colossalai.registry import LOSSES, HOOKS
from utils import build_cifar, save_checkpoint


@HOOKS.register_module
class MyTensorboardHook(hooks.BaseHook):
    """Specialized hook to record the metric to Tensorboard.

    Args:
        log_dir (str): Directory of log.
        ranks (list): Ranks of processors.
        parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`, optional): Parallel mode used in trainer,
            defaults to colossalai.context.parallel_mode.ParallelMode.GLOBAL.
        priority (int, optional): Priority in the printing, hooks with small priority will be printed in front,
            defaults to 10. If different hooks share same priority, the order of printing would
            depend on the hooks order in the hook list.
    """

    def __init__(
        self,
        log_dir: str,
        ranks = None,
        parallel_mode: ParallelMode = ParallelMode.GLOBAL,
        priority: int = 10,
    ) -> None:
        super().__init__(priority=priority)
        from torch.utils.tensorboard import SummaryWriter

        # create log dir
        if not gpc.is_initialized(ParallelMode.GLOBAL) or gpc.get_global_rank() == 0:
            os.makedirs(log_dir, exist_ok=True)

        # determine the ranks to generate tensorboard logs
        self._is_valid_rank_to_log = False
        if not gpc.is_initialized(parallel_mode):
            self._is_valid_rank_to_log = True
        else:
            local_rank = gpc.get_local_rank(parallel_mode)

            if ranks is None or local_rank in ranks:
                self._is_valid_rank_to_log = True

        # check for
        if gpc.is_initialized(ParallelMode.PIPELINE) and \
                not gpc.is_last_rank(ParallelMode.PIPELINE) and self._is_valid_rank_to_log:
            raise ValueError("Tensorboard hook can only log on the last rank of pipeline process group")

        if self._is_valid_rank_to_log:
            # create workspace on only one rank
            if gpc.is_initialized(parallel_mode):
                rank = gpc.get_local_rank(parallel_mode)
            else:
                rank = 0

            # create workspace
            log_dir = os.path.join(log_dir, f'{parallel_mode}_rank_{rank}')
            os.makedirs(log_dir, exist_ok=True)

            self.writer = SummaryWriter(log_dir=log_dir, filename_suffix=f'_rank_{rank}')

    def _log_by_iter(self, trainer, mode: str):
        for metric_name, metric_calculator in trainer.states['metrics'][mode].items():
            if metric_calculator.epoch_only:
                continue
            val = metric_calculator.get_last_step_value()

            if self._is_valid_rank_to_log:
                self.writer.add_scalar(f'{metric_name}/{mode}', float(val), trainer.cur_step)

    def _log_by_epoch(self, trainer, mode: str):
        for metric_name, metric_calculator in trainer.states['metrics'][mode].items():
            if metric_calculator.epoch_only:
                val = metric_calculator.get_accumulated_value()
                if self._is_valid_rank_to_log:
                    self.writer.add_scalar(f'{metric_name}/{mode}', float(val), trainer.cur_step)

    def after_test_iter(self, trainer, *args):
        self._log_by_iter(trainer, mode='test')

    def after_test_epoch(self, trainer):
        self._log_by_epoch(trainer, mode='test')

    def after_train_iter(self, trainer, *args):
        self._log_by_iter(trainer, mode='train')

    def after_train_epoch(self, trainer):
        self._log_by_epoch(trainer, mode='train')


def main():
    # initialize distributed setting
    parser = colossalai.get_default_parser()
    args = parser.parse_args()
    disable_existing_loggers()

    # launch from torch
    colossalai.launch_from_torch(config=args.config)

    # get logger
    logger = get_dist_logger()
    logger.info("initialized distributed environment", ranks=[0])

    # build model
    model = CvT(CvT_config)
    # model = CvT(224, 3, 100)
    
    # load model
    load_checkpoint(model.stage_1, filename='stage_1.pth.tar')
    load_checkpoint(model.stage_2, filename='stage_2.pth.tar')
    load_checkpoint(model.stage_3, filename='stage_3.pth.tar')
    
    # build dataloader
    root = os.environ.get('DATA', './data')
    train_dataloader, test_dataloader = build_cifar(gpc.config.BATCH_SIZE, root, pad_if_needed=True)

    # build optimizer
    optimizer = colossalai.nn.Lamb(model.parameters(), lr=5e-5, weight_decay=0.1)

    # build loss
    criterion = torch.nn.CrossEntropyLoss()

    # lr_scheduler
    lr_scheduler = LinearWarmupLR(optimizer, warmup_steps=5, total_steps=gpc.config.NUM_EPOCHS)

    engine, train_dataloader, test_dataloader, _ = colossalai.initialize(model, optimizer, criterion, train_dataloader,
                                                                         test_dataloader)
    logger.info("initialized colossalai components", ranks=[0])

    # build trainer
    trainer = Trainer(engine=engine, logger=logger)

    # build hooks
    hook_list = [
        hooks.LossHook(),
        hooks.AccuracyHook(accuracy_func=Accuracy()),
        hooks.LogMetricByEpochHook(logger),
        hooks.LRSchedulerHook(lr_scheduler, by_epoch=True),
        MyTensorboardHook(config.tf_dir, ranks=[0], priority=0),
    ]
    
    # start training
    trainer.fit(train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                epochs=gpc.config.NUM_EPOCHS,
                hooks=hook_list,
                display_progress=True,
                test_interval=1)
    
    save_checkpoint(model.stage_1, filename='stage_1.pth.tar')
    save_checkpoint(model.stage_2, filename='stage_2.pth.tar')
    save_checkpoint(model.stage_3, filename='stage_3.pth.tar')
    save_checkpoint(model.mlp_head, filename='mlp_cifar100.pth.tar')
    logger.log_to_file(config.log_dir)

if __name__ == '__main__':
    main()
