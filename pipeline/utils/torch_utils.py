import datetime
import time
import logging
import platform
import subprocess
import os
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def date_modified(path=__file__):
    """ return human-readable file modification date, i.e. '2021-3-26'
    Arguments:
        path (str): file path
    """
    t = datetime.datetime.fromtimestamp(Path(file).stat().st_mtime)
    return "{}-{}-{}".format(t.year, t.month, t.day)

def git_describe(path = Path(__file__).parent):
    """ get human-readable git description
        i.e. v5.0-5-g3e25f1e https://git-scm.com/docs/git-describe
    Arguments:
        path (str): path to git repo
    """
    s = "git -C {} describe --tags --long --always".format(path)
    try:
        return subprocess.check_output(s, shell = True, stderr = subprocess.STDOUT).decode()[:-1]
    except subprocess.CalledProcessError as e:
        # input path is not a git repo
        return ''

def select_device(device = '', batch_size = None):
    """ select device to proceed training/inference
    Arguments:
        device (str): 'cpu' or '0', or '0,1,2,3', default: ''
        batch_size (int): batch size, default: None
    """
    s = "YOLOv3 🚀 {} torch {}".format(git_describe() or date_modified(), torch.__version__)
    cpu = device.lower() == 'cpu'
    if cpu:
        # if in cpu mode, force torch.cuda.is_available() = False
        os.environ['CUDA_VISIBLE_DEVICE'] = -1
    elif device:
        # otherwise non-cpu device requested
        # set environment variable
        os.environ['CUDA_VISIBLE_DEVICE'] = device
        assert torch.cuda.is_available(), "CUDA unavailable, invalid device {} requested".format(device)

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        devices = device.split(',') if device else range(torch.cuda.device_count())
        # number of devices
        n = len(devices)
        # check if batch_size is divisible by device_count
        if n > 1 and batch_size:
            assert batch_size % n == 0, "batch-size {} not multiple of GPU count".format(n)
        space = ' ' * len(s)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += "{}CUDA:{} ({}, {}MB)\n".format('' if i == 0 else space, d, p.name, p.total_memory / 1024 ** 2)
    else:
        s += 'CPU\n'

    logger.info(s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s)
    return torch.device('cuda:0' if cuda else 'cpu')
