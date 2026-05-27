from .ExperimentLogger import ExperimentLogger
from .loader import RunData, load_run, load_runs
from .transform import normalize_run, normalize_runs, merge_runs, forward_fill_users
from .uuid_extractor import extract_uuids_from_filenames
from .aggregations import agg_grs_by_role, compute_state_percentages


def __getattr__(name):
    if name in {"aggregations", "plots", "transform"}:
        import importlib
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ExperimentLogger",
    "RunData",
    "load_run",
    "load_runs",
    "normalize_run",
    "normalize_runs",
    "forward_fill_users",
    "merge_runs",
    "extract_uuids_from_filenames",
    "aggregations",
    "plots",
    "agg_grs_by_role",
    "compute_state_percentages",
]
