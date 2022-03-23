import random
import time
from contextlib import contextmanager

import numpy as np
import torch

from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
    MetadataCatalog,
)
from detectron2.engine import DefaultTrainer, SimpleTrainer
from detectron2.evaluation import DatasetEvaluators
from detectron2.utils.events import CommonMetricPrinter

from roca.data import CADCatalog, Mapper
from roca.evaluation import (
    DepthEvaluator,
    InstanceEvaluator,
    Vid2CADEvaluator,
)
from roca.utils.logging import CustomLogger


class Trainer(DefaultTrainer):
    def __init__(self, cfg):
        seed = cfg.SEED
        if seed > 0:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        super().__init__(cfg)

        # Init a custom tensorboard logger
        self._custom_logger = CustomLogger.get(self.cfg)

        # Configure run step for logging
        self._configure_step()

        # Unfreeze the model
        self.model.requires_grad_()

        # Determine per-iteration val step
        self._init_val_step()

    def resume_or_load(self, resume=True):
        super().resume_or_load(resume=resume)
        if resume:
            self._custom_logger.iter = self.start_iter

    def build_writers(self):
        # Disable tensorboard and json in favor of custom logs
        return [CommonMetricPrinter(self.max_iter)]

    def _init_val_step(self):
        self.do_val_step = self.cfg.SOLVER.EVAL_STEP
        if self.do_val_step:
            test_datasets = self.cfg.DATASETS.TEST
            assert len(test_datasets) == 1, \
                'multiple test datasets not supported'

            dataset = get_detection_dataset_dicts(self.cfg.DATASETS.TEST)
            num_workers = self.cfg.SOLVER.WORKERS
            mapper = Mapper(
                self.cfg,
                is_train=True,
                dataset_name=test_datasets[0]
            )
            self._sample_val_data = build_detection_train_loader(
                mapper=mapper,
                dataset=dataset,
                total_batch_size=self.cfg.SOLVER.IMS_PER_BATCH,
                num_workers=num_workers
            )
            self._sample_val_iter = iter(self._sample_val_data)

    def _configure_step(self):

        # See detectron2 SimpleTrainer.run_step
        def run_step(self):
            assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
            start = time.perf_counter()

            # load data
            data = next(self._data_loader_iter)
            data_time = time.perf_counter() - start

            # Run the model
            loss_dict = self.model(data)
            losses = sum(loss_dict.values())

            # Run step
            self.optimizer.zero_grad()
            losses.backward()
            self._write_metrics(loss_dict, data_time)
            self.optimizer.step()

            # Return losses for custom logging
            return {**loss_dict, 'total_loss': losses}

        # Set the custom run step logic
        assert isinstance(self._trainer, SimpleTrainer)
        self._trainer.run_step = run_step.__get__(self._trainer, SimpleTrainer)

    def run_step(self):
        self._trainer.iter = self.iter
        with self._custom_logger.train_context(self.model):
            loss_dict = self._trainer.run_step()

        # Custom step
        self._custom_logger.iter = self.iter
        with torch.no_grad():
            # Log train loss
            self._custom_logger.log_train(loss_dict)

            # Log a vall loss for per-iter val sampling
            # NOTE: batchnorms are frozen in detectron2!
            # Without this behavior, val sampling introduce error
            # TODO: Handle dropouts?
            if self.do_val_step:
                data = next(self._sample_val_iter)
                with self._custom_logger.val_context(self.model):
                    val_loss_dict = self.model(data)
                val_loss_dict['total_loss'] = sum(val_loss_dict.values())
                self._custom_logger.log_val(val_loss_dict)

            # Inform the custom logger
            self._custom_logger.step()

    @classmethod
    def build_train_loader(cls, cfg):
        datasets = cfg.DATASETS.TRAIN
        assert len(datasets) == 1
        workers = cfg.SOLVER.WORKERS
        mapper = Mapper(cfg, is_train=True, dataset_name=datasets[0])
        seed = cfg.SEED
        if seed > 0:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

        return build_detection_train_loader(
            cfg,
            mapper=mapper,
            num_workers=workers
        )

    @classmethod
    def build_test_loader(cls, cfg, dataset_name: str):
        mapper = Mapper(cfg, is_train=False, dataset_name=dataset_name)
        return build_detection_test_loader(cfg, dataset_name, mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name: str):
        full_annot = MetadataCatalog.get(dataset_name).full_annot
        v2c_eval = Vid2CADEvaluator(
            dataset_name,
            full_annot,
            cfg,
            output_dir=cfg.OUTPUT_DIR
        )
        v2c_eval_ret = Vid2CADEvaluator(
            dataset_name,
            full_annot,
            cfg,
            output_dir=cfg.OUTPUT_DIR,
            exact_ret=True,
            key_prefix='retrieval_'
        )
        ap_eval = InstanceEvaluator(dataset_name, cfg)
        depth_eval = DepthEvaluator(dataset_name, cfg)
        return DatasetEvaluators([ap_eval, depth_eval, v2c_eval, v2c_eval_ret])

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        with cls.cad_context(cfg, model):
            results = super().test(cfg, model, evaluators)
        CustomLogger.get(cfg).log_metrics(results, 'test')
        return results

    @staticmethod
    @contextmanager
    def cad_context(cfg, model):
        retrieval = cfg.MODEL.RETRIEVAL_ON
        wild_retrieval = cfg.MODEL.WILD_RETRIEVAL_ON
        voxel = cfg.INPUT.CAD_TYPE == 'voxel'
        try:
            if retrieval:
                dataset = cfg.DATASETS.TEST[0]
                cad_manager = CADCatalog.get(dataset)
                points, ids = cad_manager.batched_points_and_ids(volumes=voxel)
                scene_data = cad_manager.scene_alignments

                model.set_cad_models(points, ids, scene_data)
                model.embed_cad_models()

            if wild_retrieval:
                train_cads = CADCatalog.get(cfg.DATASETS.TRAIN[0])
                train_points, train_ids = train_cads.batched_points_and_ids(
                    volumes=voxel
                )
                model.set_train_cads(train_points, train_ids)
                model.embed_train_cads()
            yield
        finally:
            if retrieval:
                model.unset_cad_models()
            if wild_retrieval:
                model.unset_train_cads()
