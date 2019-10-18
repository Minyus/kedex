from torch.utils.data import DataLoader
import torch
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator

from tqdm import tqdm


def train_model(
    train_params  # type: dict
):
    def _train_model(model, train_dataset, val_dataset, parameters):

        train_batch_size = train_params.get("train_batch_size")
        val_batch_size = train_params.get("val_batch_size")
        epochs = train_params.get("epochs")
        log_interval = train_params.get("log_interval")
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

        desc = "ITERATION - loss: {:.2f}"
        pbar = tqdm(
            initial=0, leave=False, total=len(train_loader), desc=desc.format(0)
        )

        @trainer.on(Events.ITERATION_COMPLETED)
        def log_training_loss(engine):
            iter = (engine.state.iteration - 1) % len(train_loader) + 1

            if iter % log_interval == 0:
                pbar.desc = desc.format(engine.state.output)
                pbar.update(log_interval)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(engine):
            pbar.refresh()
            evaluator.run(train_loader)
            epoch_dict = dict(epoch=engine.state.epoch)
            metrics_dict = dict(metrics=evaluator.state.metrics)
            report_str = "[Training Results]" + " {} | {}".format(
                epoch_dict, metrics_dict
            )
            tqdm.write(report_str)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            evaluator.run(val_loader)
            epoch_dict = dict(epoch=engine.state.epoch)
            metrics_dict = dict(metrics=evaluator.state.metrics)
            report_str = "[Training Results]" + " {} | {}".format(
                epoch_dict, metrics_dict
            )
            tqdm.write(report_str)

            pbar.n = pbar.last_print_n = 0

        trainer.run(train_loader, max_epochs=epochs)
        pbar.close()

        return model

    return _train_model
