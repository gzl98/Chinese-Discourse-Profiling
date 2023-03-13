import os
import random
import sys
import time

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print("set seed", seed, "for random, np.random, torch, torch_cuda")


class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    @staticmethod
    def flush():
        sys.stdout = sys.stdout.terminal

    @staticmethod
    def init(log_file_name=None):
        # 自定义目录存放日志文件
        if log_file_name is None:
            log_path = './Logs'
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            # 日志文件名按照程序运行时间设置
            log_file_name = log_path + '/log-' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.log'
        # 记录正常的 print 信息
        sys.stdout = Logger(log_file_name)
        # # 记录 traceback 异常信息
        # sys.stderr = Logger(log_file_name)