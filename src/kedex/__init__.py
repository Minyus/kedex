from importlib.util import find_spec
from .hatch_dict.hatch_dict import *  # NOQA
from .decorators.decorators import *  # NOQA
from .ops.argparse_ops import *  # NOQA

if find_spec("kedro"):
    from .pipeline.pipeline import *  # NOQA
    from .pipeline.sub_pipeline import *  # NOQA
    from .context.catalog_sugar_context import *  # NOQA
    from .context.flexible_context import *  # NOQA
    from .context.only_missing_string_runner_context import *  # NOQA
    from .context.pipelines_in_parameters_context import *  # NOQA

    if find_spec("mlflow"):
        from .context.mlflow_context import *  # NOQA

if find_spec("pandas"):
    from .decorators import *  # NOQA
    from .io.pandas.efficient_csv_local import *  # NOQA
    from .io.pandas.pandas_cat_matrix import *  # NOQA
    from .io.pandas.pandas_describe import *  # NOQA
    from .ops.pandas_ops import *  # NOQA
    from .decorators.pandas_decorators import *  # NOQA

if find_spec("pandas_profiling"):
    from .io.pandas_profiling.pandas_profiling import *  # NOQA

if find_spec("PIL"):
    from .io.pillow.images import *  # NOQA

if find_spec("seaborn"):
    from .io.seaborn.seaborn_pairplot import *  # NOQA

if find_spec("torchvision"):
    from .io.torchvision.iterable_images import *  # NOQA

if find_spec("torch"):
    from .ops.pytorch_ops import *  # NOQA

if find_spec("shap"):
    from .ops.shap_ops import *  # NOQA

if find_spec("sklearn"):
    from .ops.sklearn_ops import *  # NOQA

if find_spec("allennlp"):
    from .ops.allennlp_ops import *  # NOQA

__version__ = "0.1.0"
