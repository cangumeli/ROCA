import os
from collections import defaultdict
from typing import Callable, Dict, List

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


class LogWindow:
    def inject_log_window(self, window: Dict[str, float]):
        self.log_window = window

    def eject_log_window(self):
        if self.logging:
            del self.log_window

    @property
    def logging(self):
        return hasattr(self, 'log_window')

    def log_metrics(self, metric_fn: Callable[[], Dict[str, float]]):
        if not self.logging:
            return
        metrics = metric_fn()
        for k, v in metrics.items():
            self.log_window[k].append(v)


class CustomLogger:
    _instance = None

    def __init__(self, cfg):
        cls = self.__class__
        assert cls._instance is None, '{} is a singleton'.format(cls.__name__)

        base_dir = os.path.join(cfg.OUTPUT_DIR, 'custom_logs')
        self._train_writer = SummaryWriter('{}/train'.format(base_dir))
        self._val_writer = SummaryWriter('{}/eval'.format(base_dir))

        self._train_window = defaultdict(lambda: [])
        self._val_window = defaultdict(lambda: [])

        self.period = 20
        self.iter = 0

    @classmethod
    def get(cls, cfg=None) -> 'CustomLogger':
        if cls._instance is None:
            assert cfg is not None, \
                'No instance of {}, provide cfg to create'.format(
                    cls.__name__
                )
            cls._instance = cls(cfg)
        return cls._instance

    @classmethod
    def remove(cls):
        cls._instance = None

    def log_train(self, loss_dict: Dict[str, torch.Tensor]):
        for k, v in loss_dict.items():
            self._train_window[k].append(v.item())

    def log_val(self, loss_dict: Dict[str, torch.Tensor]):
        for k, v in loss_dict.items():
            self._val_window[k].append(v.item())

    def log_metrics(
        self,
        results: Dict[str, float],
        pref: str,
        to_val: bool = True
    ):
        logger = self._val_writer if to_val else self._train_writer
        for metric, values in results.items():
            for name, value in values.items():
                scalar_name = '{}/{}/{}'.format(pref, metric, name)
                logger.add_scalar(scalar_name, value, self.iter)

    def step(self):
        self.iter += 1
        if self.iter % self.period != 0:
            return
        loggers = [self._train_writer, self._val_writer]
        windows = [self._train_window, self._val_window]
        for logger, window, in zip(loggers, windows):
            for k, v in window.items():
                if len(window) > 0:
                    logger.add_scalar(k, np.median(v), self.iter)
            window.clear()

    def train_context(self, model):
        return _Context(model, self._train_window)

    def val_context(self, model):
        return _Context(model, self._val_window)


class _Context:
    def __init__(self, model, window):
        self.model = model
        self.window = window

    def __enter__(self):
        self.model.inject_log_window(self.window)

    def __exit__(self, *args, **kwargs):
        self.model.eject_log_window()
