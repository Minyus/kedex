from .pipelines_in_parameters_context import PipelinesInParametersContext
from .keep_output_context import KeepOutputContext
from .only_missing_string_runner_context import (
    OnlyMissingStringRunnerDefaultOptionContext,
)
from .catalog_sugar_context import CatalogSyntacticSugarContext


class FlexibleContext(
    PipelinesInParametersContext,
    CatalogSyntacticSugarContext,
    OnlyMissingStringRunnerDefaultOptionContext,
    KeepOutputContext,
):
    project_name = "FlexibleKedroProject"
    project_version = "0.15.2"
