from importlib.util import find_spec
from .hatch_dict.hatch_dict import *  # NOQA
from .contrib.decorators.decorators import *  # NOQA
from .contrib.ops.argparse_ops import *  # NOQA

if find_spec("kedro"):
    from .pipeline.pipeline import *  # NOQA
    from .pipeline.sub_pipeline import *  # NOQA
    from .context.catalog_sugar_context import *  # NOQA
    from .context.flexible_context import *  # NOQA
    from .context.keep_output_context import *  # NOQA
    from .context.only_missing_string_runner_context import *  # NOQA
    from .context.pipelines_in_parameters_context import *  # NOQA

    if find_spec("mlflow"):
        from .mlflow_context.mlflow_flexible_context import *  # NOQA

if find_spec("pandas"):
    from .contrib.decorators.pandas_decorators import *  # NOQA
    from .contrib.io.pandas import *  # NOQA
    from .contrib.ops.pandas_ops import *  # NOQA

if find_spec("pandas_profiling"):
    from .contrib.io.pandas_profiling import *  # NOQA

if find_spec("PIL"):
    from .contrib.io.pillow import *  # NOQA

if find_spec("seaborn"):
    from .contrib.io.seaborn import *  # NOQA

if find_spec("torch"):
    from .contrib.ops.pytorch_ops import *  # NOQA

if find_spec("shap"):
    from .contrib.ops.shap_ops import *  # NOQA

if find_spec("sklearn"):
    from .contrib.ops.sklearn_ops import *  # NOQA

if find_spec("allennlp"):
    from .contrib.ops.allennlp_ops import *  # NOQA

__version__ = "0.0.9"
