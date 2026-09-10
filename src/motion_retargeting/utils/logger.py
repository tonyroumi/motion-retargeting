from torch.utils.tensorboard import SummaryWriter
import logging
import os

class Logger:   
    def __init__(self, log_dir: str = "logs", verbose: bool = False):
        os.makedirs(log_dir, exist_ok=True)

        self._logger = logging.getLogger()
        self._logger.setLevel(logging.INFO)
        
        self.writer = SummaryWriter(log_dir=log_dir)

        self.verbose = verbose
        self.step_num = 0
        self.epoch_num = 0
    
    def step(self):
        self.step_num += 1
    
    def epoch(self):
        self.epoch_num += 1
    
    def info(self, msg: str):
        self._logger.info(msg)
    
    def warn(self, msg: str):
        self._logger.warning(msg)

    def error(self, msg: str):
        self._logger.error(msg)
    
    def log_metric(self, name: str, value: float):
        if self.verbose:
            self.info(f"{name}, {value}, step: {self.step_num}")
        self.writer.add_scalar(name, value, self.step_num)
    
    def close(self):
        self.writer.close()