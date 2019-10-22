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
        progress_update = train_params.get("progress_update")

        optimizer = train_params.get("optimizer")
        optimizer_params = train_params.get("optimizer_params", dict())
        scheduler = train_params.get("scheduler")
        scheduler_params = train_params.get("scheduler_params", dict())
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

        optimizer_ = optimizer(model.parameters(), **optimizer_params)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        trainer = create_supervised_trainer(
            model, optimizer_, loss_fn=loss_fn, device=device
        )

        train_data_loader_params.setdefault("shuffle", True)
        train_data_loader_params.setdefault("drop_last", True)
        train_data_loader_params["batch_size"] = _clip_batch_size(
            train_data_loader_params.get("batch_size", 1), train_dataset, "train"
        )
        train_loader = DataLoader(train_dataset, **train_data_loader_params)

        if scheduler:

            class ParamSchedulerSavingAsMetric(
                ParamSchedulerSavingAsMetricMixIn, scheduler
            ):
                pass

            cycle_epochs = scheduler_params.pop("cycle_epochs", 1)
            scheduler_params.setdefault(
                "cycle_size", int(cycle_epochs * len(train_loader))
            )
            scheduler_params.setdefault("param_name", "lr")
            scheduler_ = ParamSchedulerSavingAsMetric(optimizer_, **scheduler_params)
            trainer.add_event_handler(Events.ITERATION_STARTED, scheduler_)

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
        if progress_update:
            if not isinstance(progress_update, dict):
                progress_update = dict()
            progress_update.setdefault("persist", True)
            progress_update.setdefault("desc", "")
            pbar = ProgressBar(**progress_update)
            try:
                RunningAverage(output_transform=lambda x: x, alpha=0.98).attach(
                    trainer, "loss_ema"
                )
                pbar.attach(trainer, ["loss_ema"])
            except Exception as e:
                log.error(e, exc_info=True)

        if evaluate_train_data:

            def log_evaluation_train_data(engine):
                evaluator_train.run(train_loader)
                train_report = _get_report_str(engine, evaluator_train, "Train Data")
                if pbar:
                    pbar.log_message(train_report)
                else:
                    log.info(train_report)

            eval_train_event = (
                Events[evaluate_train_data]
                if isinstance(evaluate_train_data, str)
                else Events.EPOCH_COMPLETED
            )
            trainer.add_event_handler(eval_train_event, log_evaluation_train_data)

        if evaluate_val_data:

            def log_evaluation_val_data(engine):
                evaluator_val.run(val_loader)
                val_report = _get_report_str(engine, evaluator_val, "Val Data")
                if pbar:
                    pbar.log_message(val_report)
                else:
                    log.info(val_report)

            eval_val_event = (
                Events[evaluate_val_data]
                if isinstance(evaluate_val_data, str)
                else Events.EPOCH_COMPLETED
            )
            trainer.add_event_handler(eval_val_event, log_evaluation_val_data)

        if mlflow_logging:
            mlflow_logger = MLflowLogger()

            logging_params = {
                "train_n_samples": len(train_dataset),
                "val_n_samples": len(val_dataset),
                "optim": optimizer.__name__,
                "loss_fn": loss_fn.__name__,
                "pytorch_version": torch.__version__,
                "ignite_version": ignite.__version__,
            }
            logging_params.update(_loggable_dict(train_data_loader_params, "train"))
            logging_params.update(_loggable_dict(val_data_loader_params, "val"))
            logging_params.update(_loggable_dict(optimizer_params))
            mlflow_logger.log_params(logging_params)

            metric_names = []
            RunningAverage(output_transform=lambda x: x, alpha=2 ** (-1022)).attach(
                trainer, "loss"
            )
            metric_names.append("loss")
            if scheduler:
                metric_names.append(scheduler_params.get("param_name"))

            mlflow_logger.attach(
                trainer,
                log_handler=OutputHandler(
                    tag="batch",
                    metric_names=metric_names,
                    global_step_transform=global_step_from_engine(trainer),
                ),
                event_name=Events.ITERATION_COMPLETED,
            )

            if evaluate_train_data:
                mlflow_logger.attach(
                    evaluator_train,
                    log_handler=OutputHandler(
                        tag="train",
                        metric_names=list(metrics.keys()),
                        global_step_transform=global_step_from_engine(trainer),
                    ),
                    event_name=Events.COMPLETED,
                )
            if evaluate_val_data:
                mlflow_logger.attach(
                    evaluator_val,
                    log_handler=OutputHandler(
                        tag="val",
                        metric_names=list(metrics.keys()),
                        global_step_transform=global_step_from_engine(trainer),
                    ),
                    event_name=Events.COMPLETED,
                )

        trainer.run(train_loader, max_epochs=epochs)

        try:
            if pbar and pbar.pbar:
                pbar.pbar.close()
        except Exception as e:
            log.error(e, exc_info=True)

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


class ParamSchedulerSavingAsMetricMixIn:
    """ Base code:
     https://github.com/pytorch/ignite/blob/v0.2.1/ignite/contrib/handlers/param_scheduler.py#L49
     https://github.com/pytorch/ignite/blob/v0.2.1/ignite/contrib/handlers/param_scheduler.py#L163
    """

    def __call__(self, engine, name=None):

        if self.event_index != 0 and self.event_index % self.cycle_size == 0:
            self.event_index = 0
            self.cycle_size *= self.cycle_mult
            self.cycle += 1
            self.start_value *= self.start_value_mult
            self.end_value *= self.end_value_mult

        value = self.get_param()

        for param_group in self.optimizer_param_groups:
            param_group[self.param_name] = value

        if name is None:
            name = self.param_name

        if self.save_history:
            if not hasattr(engine.state, "param_history"):
                setattr(engine.state, "param_history", {})
            engine.state.param_history.setdefault(name, [])
            values = [pg[self.param_name] for pg in self.optimizer_param_groups]
            engine.state.param_history[name].append(values)

        self.event_index += 1

        if not hasattr(engine.state, "metrics"):
            setattr(engine.state, "metrics", {})
        engine.state.metrics[self.param_name] = value  # Save as a metric


class PytorchSequential(torch.nn.Sequential):
    def __init__(self, modules):
        super().__init__(*modules)


class PytorchFlatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
