import torch

from torch.utils.data import DataLoader
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import RunningAverage
from ignite.handlers import EarlyStopping
from ignite.contrib.handlers.tqdm_logger import ProgressBar
import numpy as np
import time
from pkg_resources import parse_version
import logging

log = logging.getLogger(__name__)


def pytorch_train(
    train_params,  # type: dict
    mlflow_logging=True,  # type: bool
):
    if mlflow_logging:
        try:
            import mlflow
        except ImportError:
            log.warning("Failed to import mlflow. MLflow logging is disabled.")
            mlflow_logging = False

    if mlflow_logging:
        import ignite

        if parse_version(ignite.__version__) >= parse_version("0.2.1"):
            from ignite.contrib.handlers.mlflow_logger import (
                MLflowLogger,
                OutputHandler,
                global_step_from_engine,
            )
        else:
            from .ignite.contrib.handlers.mlflow_logger import (
                MLflowLogger,
                OutputHandler,
                global_step_from_engine,
            )

    def _pytorch_train(model, train_dataset, val_dataset=None, parameters=None):

        train_data_loader_params = train_params.get("train_data_loader_params", dict())
        val_data_loader_params = train_params.get("val_data_loader_params", dict())
        epochs = train_params.get("epochs")
        progress_update = train_params.get("progress_update", dict())

        optim = train_params.get("optim")
        optim_params = train_params.get("optim_params", dict())
        loss_fn = train_params.get("loss_fn")
        metrics = train_params.get("metrics")

        evaluate_train_data = train_params.get("evaluate_train_data")
        evaluate_val_data = train_params.get("evaluate_val_data")

        early_stopping_params = train_params.get("early_stopping_params")
        time_limit = train_params.get("time_limit")

        seed = train_params.get("seed")
        cudnn_deterministic = train_params.get("cudnn_deterministic")
        cudnn_benchmark = train_params.get("cudnn_benchmark")

        if seed:
            torch.manual_seed(seed)
            np.random.seed(seed)
        if cudnn_deterministic:
            torch.backends.cudnn.deterministic = cudnn_deterministic
        if cudnn_benchmark:
            torch.backends.cudnn.benchmark = cudnn_benchmark

        optimizer = optim(model.parameters(), **optim_params)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        trainer = create_supervised_trainer(
            model, optimizer, loss_fn=loss_fn, device=device
        )

        train_data_loader_params.setdefault("shuffle", True)
        train_data_loader_params.setdefault("drop_last", True)
        train_data_loader_params["batch_size"] = _clip_batch_size(
            train_data_loader_params.get("batch_size", 1), train_dataset, "train"
        )
        train_loader = DataLoader(train_dataset, **train_data_loader_params)

        if evaluate_train_data:
            evaluator_train = create_supervised_evaluator(
                model, metrics=metrics, device=device
            )

        if evaluate_val_data:
            val_data_loader_params["batch_size"] = _clip_batch_size(
                val_data_loader_params.get("batch_size", 1), val_dataset, "val"
            )
            val_loader = DataLoader(val_dataset, **val_data_loader_params)
            evaluator_val = create_supervised_evaluator(
                model, metrics=metrics, device=device
            )
            if early_stopping_params:
                assert isinstance(early_stopping_params, dict)
                metric = early_stopping_params.get("metric")
                assert metric in metrics
                minimize = early_stopping_params.get("minimize")
                patience = early_stopping_params.get("patience", 1)

                def score_function(engine):
                    m = engine.state.metrics.get(metric)
                    return -m if minimize else m

                es = EarlyStopping(
                    patience=patience, score_function=score_function, trainer=trainer
                )
                evaluator_val.add_event_handler(Events.COMPLETED, es)
        elif early_stopping_params:
            log.warning(
                "Early Stopping is disabled because evaluate_val_data is not True."
            )

        if time_limit:
            assert isinstance(time_limit, (int, float))
            tl = TimeLimit(limit_sec=time_limit)
            trainer.add_event_handler(Events.ITERATION_COMPLETED, tl)

        pbar = None
        if isinstance(progress_update, dict):
            RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")

            progress_update.setdefault("persist", True)
            progress_update.setdefault("desc", "")
            pbar = ProgressBar(**progress_update)
            pbar.attach(trainer, ["loss"])

        if evaluate_train_data:

            def log_evaluation_train_data(engine):
                evaluator_train.run(train_loader)
                if pbar:
                    pbar.log_message(
                        _get_report_str(engine, evaluator_train, "Train Data")
                    )

            trainer.add_event_handler(Events.EPOCH_COMPLETED, log_evaluation_train_data)

        if evaluate_val_data:

            def log_evaluation_val_data(engine):
                evaluator_val.run(val_loader)
                if pbar:
                    pbar.log_message(_get_report_str(engine, evaluator_val, "Val Data"))

            trainer.add_event_handler(Events.EPOCH_COMPLETED, log_evaluation_val_data)

        if mlflow_logging:
            mlflow_logger = MLflowLogger()

            logging_params = {
                "train_n_samples": len(train_dataset),
                "val_n_samples": len(val_dataset),
                "optim": optim.__name__,
                "loss_fn": loss_fn.__name__,
                "pytorch_version": torch.__version__,
                "ignite_version": ignite.__version__,
            }
            logging_params.update(_loggable_dict(train_data_loader_params, "train"))
            logging_params.update(_loggable_dict(val_data_loader_params, "val"))
            logging_params.update(_loggable_dict(optim_params))
            mlflow_logger.log_params(logging_params)

            if evaluate_train_data:
                mlflow_logger.attach(
                    evaluator_train,
                    log_handler=OutputHandler(
                        tag="train",
                        metric_names=list(metrics.keys()),
                        global_step_transform=global_step_from_engine(trainer),
                    ),
                    event_name=Events.EPOCH_COMPLETED,
                )
            if evaluate_val_data:
                mlflow_logger.attach(
                    evaluator_val,
                    log_handler=OutputHandler(
                        tag="val",
                        metric_names=list(metrics.keys()),
                        global_step_transform=global_step_from_engine(trainer),
                    ),
                    event_name=Events.EPOCH_COMPLETED,
                )

        trainer.run(train_loader, max_epochs=epochs)

        return model

    return _pytorch_train


def _clip_batch_size(batch_size, dataset, tag=""):
    dataset_size = len(dataset)
    if batch_size > dataset_size:
        log.warning(
            "[{}] batch size ({}) is clipped to dataset size ({})".format(
                tag, batch_size, dataset_size
            )
        )
        return dataset_size
    else:
        return batch_size


def _get_report_str(engine, evaluator, tag=""):
    report_str = "[Epoch: {} | {} | Metrics: {}]".format(
        engine.state.epoch, tag, evaluator.state.metrics
    )
    return report_str


def _loggable_dict(d, prefix=None):
    return {
        ("{}_{}".format(prefix, k) if prefix else k): (
            "{}".format(v) if isinstance(v, (tuple, list, dict, set)) else v
        )
        for k, v in d.items()
    }


class TimeLimit:
    def __init__(self, limit_sec=3600):
        self.limit_sec = limit_sec
        self.start_time = time.time()

    def __call__(self, engine):
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.limit_sec:
            log.warning(
                "Reached the time limit: {} sec. Stop training".format(self.limit_sec)
            )
            engine.terminate()


class PytorchSequential(torch.nn.Sequential):
    def __init__(self, modules):
        super().__init__(*modules)


class PytorchFlatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
