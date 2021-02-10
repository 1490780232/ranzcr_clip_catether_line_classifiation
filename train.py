import os
from torch.backends import cudnn

from config import Config
from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
from solver import make_optimizer, WarmupMultiStepLR
from processor import do_train
import torch
from model.backbones.ibnnet.resnext_ibn import resnext101_ibn_a
if __name__ == '__main__':
    cfg = Config()
    if not os.path.exists(cfg.LOG_DIR):
        os.mkdir(cfg.LOG_DIR)
    logger = setup_logger('{}'.format(cfg.PROJECT_NAME), cfg.LOG_DIR)
    logger.info("Running with config:\n{}".format(cfg.CFG_NAME))
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.DEVICE_ID
    cudnn.benchmark = True

    train_loader, val_loader = make_dataloader(cfg)
    model = make_model(cfg, num_class=11)
    # model.load_param(cfg.FINE_TUNE)
    loss_func = torch.nn.BCEWithLogitsLoss()

    optimizer = make_optimizer(cfg, model)
    scheduler = WarmupMultiStepLR(optimizer, cfg.STEPS, cfg.GAMMA,
                                  cfg.WARMUP_FACTOR,
                                  cfg.WARMUP_EPOCHS, cfg.WARMUP_METHOD)

    do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_func,
    )
