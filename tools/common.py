# -*- coding:utf-8 -*-
# File       : text_classify.py
# Time       8/8/2023 下午 11:55
# Author     ：rain
# Description：
import os
import random
import torch
import numpy as np
import json
import pickle
import logging
import torch.nn as nn
from collections import OrderedDict
from pathlib import Path
import matplotlib.pyplot as plt

logger = logging.getLogger()


def print_config(config):
    info = "Running with the following configs:\n"
    for k, v in config.items():
        info += f"\t{k} : {str(v)}\n"
    print("\n" + info + "\n")
    return


def init_logger(log_file=None, log_file_level=logging.NOTSET):
    if isinstance(log_file, Path):
        log_file = str(log_file)
    log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                   datefmt='%m/%d/%Y %H:%M:%S')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        # file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger


def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def prepare_device(n_gpu_use):
    if not n_gpu_use:
        device_type = 'cpu'
    else:
        n_gpu_use = n_gpu_use.split(",")
        device_type = f"cuda:{n_gpu_use[0]}"
    n_gpu = torch.cuda.device_count()
    if len(n_gpu_use) > 0 and n_gpu == 0:
        logger.warning("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        device_type = 'cpu'
    if len(n_gpu_use) > n_gpu:
        msg = f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are available on this machine."
        logger.warning(msg)
        n_gpu_use = range(n_gpu)
    device = torch.device(device_type)
    list_ids = n_gpu_use
    return device, list_ids


def model_device(n_gpu, model):
    device, device_ids = prepare_device(n_gpu)
    if len(device_ids) > 1:
        logger.info(f"current {len(device_ids)} GPUs")
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    if len(device_ids) == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_ids[0])
    model = model.to(device)
    return model, device


def restore_checkpoint(resume_path, model=None):
    if isinstance(resume_path, Path):
        resume_path = str(resume_path)
    checkpoint = torch.load(resume_path)
    best = checkpoint['best']
    start_epoch = checkpoint['epoch'] + 1
    states = checkpoint['state_dict']
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(states)
    else:
        model.load_state_dict(states)
    return [model, best, start_epoch]


def save_pickle(data, file_path):
    if isinstance(file_path, Path):
        file_path = str(file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(input_file):
    with open(str(input_file), 'rb') as f:
        data = pickle.load(f)
    return data


def save_json(data, file_path):
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    with open(str(file_path), 'w') as f:
        json.dump(data, f)


def save_numpy(data, file_path):
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    np.save(str(file_path), data)


def load_numpy(file_path):
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    np.load(str(file_path))


def load_json(file_path):
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    with open(str(file_path), 'r') as f:
        data = json.load(f)
    return data


def json_to_text(file_path, data):
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    with open(str(file_path), 'w') as fw:
        for line in data:
            line = json.dumps(line, ensure_ascii=False)
            fw.write(line + '\n')


def save_model(model, model_path):
    if isinstance(model_path, Path):
        model_path = str(model_path)
    if isinstance(model, nn.DataParallel):
        model = model.module
    state_dict = model.state_dict()
    for key in state_dict:
        state_dict[key] = state_dict[key].cpu()
    torch.save(state_dict, model_path)


def load_model(model, model_path):
    if isinstance(model_path, Path):
        model_path = str(model_path)
    logging.info(f"loading model from {str(model_path)} .")
    states = torch.load(model_path)
    state = states['state_dict']
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state)
    else:
        model.load_state_dict(state)
    return model


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def plot_img_acc_loss(train_ds, dev_ds, img_type, model_type):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_ds) + 1), train_ds, label=f'Train {img_type}')
    plt.plot(range(1, len(dev_ds) + 1), dev_ds, label=f'Test {img_type}')
    plt.xlabel('Epoch')
    plt.ylabel(f'{img_type}')
    plt.title(f'{img_type} Curve for {model_type}')
    plt.legend()
    plt.savefig(f'outputs/{model_type}_{img_type}.png')


def plot_img_auc(auc, model_type):
    plt.figure(figsize=(10, 5))
    
    plt.plot(range(1, len(auc) + 1), auc, label='Val AUC', color='orange', linestyle='-', marker='o')
    
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.title(f'AUC Curve for {model_type}')
    
    plt.legend()
    
    plt.savefig(f'outputs/{model_type}_auc_curve.png')

