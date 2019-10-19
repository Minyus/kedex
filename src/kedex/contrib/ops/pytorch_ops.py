import torch

from torch.utils.data import DataLoader
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from pkg_resources import parse_version


def pytorch_train(
    train_params,  # type: dict
    mlflow_logging=True,  # type: bool
):
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

    def _pytorch_train(model, train_dataset, val_dataset, parameters):

        train_batch_size = train_params.get("train_batch_size")
        val_batch_size = train_params.get("val_batch_size")
        epochs = train_params.get("epochs")
        progress_update = train_params.get("progress_update", dict())

        optim = train_params.get("optim")
        optim_params = train_params.get("optim_params", dict())
        loss_fn = train_params.get("loss_fn")
        metrics = train_params.get("metrics")

        optimizer = optim(model.parameters(), **optim_params)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        trainer = create_supervised_trainer(
            model, optimizer, loss_fn=loss_fn, device=device
        )
        evaluator_train = create_supervised_evaluator(
            model, metrics=metrics, device=device
        )
        evaluator_val = create_supervised_evaluator(
            model, metrics=metrics, device=device
        )

        train_loader = DataLoader(
            train_dataset, batch_size=train_batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

        pbar = None
        if isinstance(progress_update, dict):
            RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")

            progress_update.setdefault("persist", True)
            progress_update.setdefault("desc", "")
            pbar = ProgressBar(**progress_update)
            pbar.attach(trainer, ["loss"])

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_evaluation_results(engine):
            evaluator_train.run(train_loader)
            if pbar:
                pbar.log_message(_get_report_str(engine, evaluator_train, "Train Data"))
            evaluator_val.run(val_loader)
            if pbar:
                pbar.log_message(_get_report_str(engine, evaluator_val, "Val Data"))

        if mlflow_logging:
            mlflow_logger = MLflowLogger()
            mlflow_logger.attach(
                evaluator_train,
                log_handler=OutputHandler(
                    tag="Train",
                    metric_names=list(metrics.keys()),
                    global_step_transform=global_step_from_engine(trainer),
                ),
                event_name=Events.EPOCH_COMPLETED,
            )
            mlflow_logger.attach(
                evaluator_val,
                log_handler=OutputHandler(
                    tag="Val",
                    metric_names=list(metrics.keys()),
                    global_step_transform=global_step_from_engine(trainer),
                ),
                event_name=Events.EPOCH_COMPLETED,
            )

        trainer.run(train_loader, max_epochs=epochs)

        return model

    return _pytorch_train


def _get_report_str(engine, evaluator, tag=""):
    report_str = "[Epoch: {} | {} | Metrics: {}]".format(
        engine.state.epoch, tag, evaluator.state.metrics
    )
    return report_str


class PytorchSequential(torch.nn.Sequential):
    def __init__(self, modules):
        super().__init__(*modules)


class PytorchFlatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
