import torch
import random
import logging
import time
import os
import numpy as np
from typing import List, Tuple, Dict, Union

class ModelOutput:
    def __init__(self, logits, labels, loss=None):
        self.logits = logits
        self.labels = labels
        self.loss = loss
        
        
def model_load(model, path, device):
    '''
    加载指定路径的模型
    '''
    model.load_state_dict(torch.load(path, map_location=device)['state_dict'])
    return model


def model_save(model, epoch=0, cfg=None):
    '''
    保存模型，默认使用“模型名字+时间”作为文件名
    '''
    time_prefix = time.strftime('%Y-%m-%d_%H-%M-%S')
    prefix = os.path.join(cfg.cwd, 'checkpoints',time_prefix)
    os.makedirs(prefix, exist_ok=True)
    name = os.path.join(prefix, cfg.model_name + '_' + f'epoch{epoch}' + '.pth')

    torch.save(model.state_dict(), name)
    return name

def load_ckp(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

def save_ckp(state, checkpoint_path):
    torch.save(state, checkpoint_path)

"""
def save_ckp(self, state, is_best, checkpoint_path, best_model_path):
    tmp_checkpoint_path = checkpoint_path
    torch.save(state, tmp_checkpoint_path)
    if is_best:
        tmp_best_model_path = best_model_path
        shutil.copyfile(tmp_checkpoint_path, tmp_best_model_path)
"""


def manual_seed(seed: int = 1) -> None:
    """
        设置seed。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #if torch.cuda.CUDA_ENABLED and use_deterministic_cudnn:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False