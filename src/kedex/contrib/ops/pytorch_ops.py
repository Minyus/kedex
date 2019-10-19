import torch
from torch.utils.data import DataLoader
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar


def pytorch_train(
    train_params  # type: dict
):
    def _train_model(model, train_dataset, val_dataset, parameters):

        train_batch_size = train_params.get("train_batch_size")
        val_batch_size = train_params.get("val_batch_size")
        epochs = train_params.get("epochs")
        progress_params = train_params.get("progress_params", dict())

        optim = train_params.get("optim")
        optim_params = train_params.get("optim_params", dict())
        loss_fn = train_params.get("loss_fn")
        metrics = train_params.get("metrics")

        optimizer = optim(model.parameters(), **optim_params)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        trainer = create_supervised_trainer(
            model, optimizer, loss_fn=loss_fn, device=device
        )
        evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

        train_loader = DataLoader(
            train_dataset, batch_size=train_batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

        RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")

        progress_params.setdefault("persist", True)
        progress_params.setdefault("desc", "")
        pbar = ProgressBar(**progress_params)
        pbar.attach(trainer, ["loss"])

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(engine):
            evaluator.run(train_loader)
            pbar.log_message(_get_report_str(engine, evaluator, "Train Data"))

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            evaluator.run(val_loader)
            pbar.log_message(_get_report_str(engine, evaluator, "Val Data"))

        trainer.run(train_loader, max_epochs=epochs)

        return model

    return _train_model


def _get_report_str(engine, evaluator, data_desc=""):
    report_str = "[Epoch: {} | {} | Metrics: {}]".format(
        engine.state.epoch, data_desc, evaluator.state.metrics
    )
    return report_str


class PytorchSequential(torch.nn.Sequential):
    def __init__(self, modules):
        super().__init__(*modules)


class PytorchFlatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
