import numbers

import warnings
import torch

from .base_logger import (
    BaseLogger,
    BaseOutputHandler,
    BaseOptimizerParamsHandler,
    global_step_from_engine,
)


__all__ = [
    "MLflowLogger",
    "OutputHandler",
    "OptimizerParamsHandler",
    "global_step_from_engine",
]


class OutputHandler(BaseOutputHandler):
    """Helper handler to log engine's output and/or metrics.

    Examples:

        .. code-block:: python

            from ignite.contrib.handlers.mlflow_logger import *

            # Create a logger
            mlflow_logger = MLflowLogger()

            # Attach the logger to the evaluator on the validation dataset and log NLL, Accuracy metrics after
            # each epoch. We setup `global_step_transform=global_step_from_engine(trainer)` to take the epoch
            # of the `trainer`:
            mlflow_logger.attach(evaluator,
                                 log_handler=OutputHandler(tag="validation",
                                                           metric_names=["nll", "accuracy"],
                                                           global_step_transform=global_step_from_engine(trainer)),
                                 event_name=Events.EPOCH_COMPLETED)

        Example with CustomPeriodicEvent, where model is evaluated every 500 iterations:

        .. code-block:: python

            from ignite.contrib.handlers import CustomPeriodicEvent

            cpe = CustomPeriodicEvent(n_iterations=500)
            cpe.attach(trainer)

            @trainer.on(cpe.Events.ITERATIONS_500_COMPLETED)
            def evaluate(engine):
                evaluator.run(validation_set, max_epochs=1)

            from ignite.contrib.handlers.mlflow_logger import *

            mlflow_logger = MLflowLogger()

            def global_step_transform(*args, **kwargs):
                return trainer.state.iteration

            # Attach the logger to the evaluator on the validation dataset and log NLL, Accuracy metrics after
            # every 500 iterations. Since evaluator engine does not have CustomPeriodicEvent attached to it, we
            # provide a global_step_transform to return the trainer.state.iteration for the global_step, each time
            # evaluator metrics are plotted on MLflow.

            mlflow_logger.attach(evaluator,
                                log_handler=OutputHandler(tag="validation",
                                                          metrics=["nll", "accuracy"],
                                                          global_step_transform=global_step_transform),
                                event_name=Events.EPOCH_COMPLETED)

    Args:
        tag (str): common title for all produced plots. For example, 'training'
        metric_names (list of str, optional): list of metric names to plot or a string "all" to plot all available
            metrics.
        output_transform (callable, optional): output transform function to prepare `engine.state.output` as a number.
            For example, `output_transform = lambda output: output`
            This function can also return a dictionary, e.g `{'loss': loss1, `another_loss`: loss2}` to label the plot
            with corresponding keys.
        another_engine (Engine): Deprecated (see :attr:`global_step_transform`). Another engine to use to provide the
            value of event. Typically, user can provide
            the trainer if this handler is attached to an evaluator and thus it logs proper trainer's
            epoch/iteration value.
        global_step_transform (callable, optional): global step transform function to output a desired global step.
            Input of the function is `(engine, event_name)`. Output of function should be an integer.
            Default is None, global_step based on attached engine. If provided,
            uses function output as global_step. To setup global step from another engine, please use
            :meth:`~ignite.contrib.handlers.mlflow_logger.global_step_from_engine`.

    Note:

        Example of `global_step_transform`:

        .. code-block:: python

            def global_step_transform(engine, event_name):
                return engine.state.get_event_attrib_value(event_name)

    """

    def __init__(
        self,
        tag,
        metric_names=None,
        output_transform=None,
        another_engine=None,
        global_step_transform=None,
    ):
        super(OutputHandler, self).__init__(
            tag, metric_names, output_transform, another_engine, global_step_transform
        )

    def __call__(self, engine, logger, event_name):

        if not isinstance(logger, MLflowLogger):
            raise RuntimeError("Handler 'OutputHandler' works only with MLflowLogger")

        metrics = self._setup_output_metrics(engine)

        global_step = self.global_step_transform(engine, event_name)

        if not isinstance(global_step, int):
            raise TypeError(
                "global_step must be int, got {}."
                " Please check the output of global_step_transform.".format(
                    type(global_step)
                )
            )

        rendered_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, numbers.Number):
                rendered_metrics["{} {}".format(self.tag, key)] = value
            elif isinstance(value, torch.Tensor) and value.ndimension() == 0:
                rendered_metrics["{} {}".format(self.tag, key)] = value.item()
            elif isinstance(value, torch.Tensor) and value.ndimension() == 1:
                for i, v in enumerate(value):
                    rendered_metrics["{} {} {}".format(self.tag, key, i)] = v.item()
            else:
                warnings.warn(
                    "MLflowLogger output_handler can not log "
                    "metrics value type {}".format(type(value))
                )

        # Additionally recheck metric names as MLflow rejects non-valid names with MLflowException
        from mlflow.utils.validation import _VALID_PARAM_AND_METRIC_NAMES

        for key in list(rendered_metrics.keys()):
            if not _VALID_PARAM_AND_METRIC_NAMES.match(key):
                warnings.warn(
                    "MLflowLogger output_handler encountered an invalid metric name '{}' that "
                    "will be ignored and not logged to MLflow".format(key)
                )
                del rendered_metrics[key]

        logger.log_metrics(rendered_metrics, step=global_step)


class OptimizerParamsHandler(BaseOptimizerParamsHandler):
    """Helper handler to log optimizer parameters

    Examples:

        .. code-block:: python

            from ignite.contrib.handlers.mlflow_logger import *

            # Create a logger
            mlflow_logger = MLflowLogger()
            # Optionally, user can specify tracking_uri with corresponds to MLFLOW_TRACKING_URI
            # mlflow_logger = MLflowLogger(tracking_uri="uri")

            # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at each iteration
            mlflow_logger.attach(trainer,
                                 log_handler=OptimizerParamsHandler(optimizer),
                                 event_name=Events.ITERATION_STARTED)

    Args:
        optimizer (torch.optim.Optimizer): torch optimizer which parameters to log
        param_name (str): parameter name
        tag (str, optional): common title for all produced plots. For example, 'generator'
    """

    def __init__(self, optimizer, param_name="lr", tag=None):
        super(OptimizerParamsHandler, self).__init__(optimizer, param_name, tag)

    def __call__(self, engine, logger, event_name):
        if not isinstance(logger, MLflowLogger):
            raise RuntimeError(
                "Handler 'OptimizerParamsHandler' works only with MLflowLogger"
            )

        global_step = engine.state.get_event_attrib_value(event_name)
        tag_prefix = "{} ".format(self.tag) if self.tag else ""
        params = {
            "{}{} group_{}".format(tag_prefix, self.param_name, i): float(
                param_group[self.param_name]
            )
            for i, param_group in enumerate(self.optimizer.param_groups)
        }

        logger.log_metrics(params, step=global_step)


class MLflowLogger(BaseLogger):
    """
    `MLflow <https://mlflow.org>`_ tracking client handler to log parameters and metrics during the training
    and validation.

    This class requires `mlflow package <https://github.com/mlflow/mlflow/>`_ to be installed:

    .. code-block:: bash

        pip install mlflow

    Args:
        tracking_uri (str): MLflow tracking uri. See MLflow docs for more details

    Examples:

        .. code-block:: python

            from ignite.contrib.handlers.mlflow_logger import *

            # Create a logger
            mlflow_logger = MLflowLogger()

            # Log experiment parameters:
            mlflow_logger.log_params(**{
                "seed": seed,
                "batch_size": batch_size,
                "model": model.__class__.__name__,

                "pytorch version": torch.__version__,
                "ignite version": ignite.__version__,
                "cuda version": torch.version.cuda,
                "device name": torch.cuda.get_device_name(0)
            })

            # Attach the logger to the evaluator on the training dataset and log NLL, Accuracy metrics after each epoch
            # We setup `global_step_transform=global_step_from_engine(trainer)` to take the epoch
            # of the `trainer` instead of `train_evaluator`.
            mlflow_logger.attach(train_evaluator,
                                 log_handler=OutputHandler(tag="training",
                                                           metric_names=["nll", "accuracy"],
                                                           global_step_transform=global_step_from_engine(trainer)),
                                 event_name=Events.EPOCH_COMPLETED)

            # Attach the logger to the evaluator on the validation dataset and log NLL, Accuracy metrics after
            # each epoch. We setup `global_step_transform=global_step_from_engine(trainer)` to take the epoch of the
            # `trainer` instead of `evaluator`.
            mlflow_logger.attach(evaluator,
                                 log_handler=OutputHandler(tag="validation",
                                                           metric_names=["nll", "accuracy"],
                                                           global_step_transform=global_step_from_engine(trainer)),
                                 event_name=Events.EPOCH_COMPLETED)
    """

    def __init__(self, tracking_uri=None):
        try:
            import mlflow
        except ImportError:
            raise RuntimeError(
                "This contrib module requires mlflow to be installed. "
                "Please install it with command: \n pip install mlflow"
            )

        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)

        self.active_run = mlflow.active_run()
        if self.active_run is None:
            self.active_run = mlflow.start_run()

    def __getattr__(self, attr):

        import mlflow

        def wrapper(*args, **kwargs):
            return getattr(mlflow, attr)(*args, **kwargs)

        return wrapper

    def close(self):
        import mlflow

        mlflow.end_run()
