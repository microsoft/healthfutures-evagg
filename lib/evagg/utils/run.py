import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from .git import RepoStatus
from .settings import SettingsModel


class RunRecord(SettingsModel):
    name: str
    timestamp: str
    args: List[str]
    git: Dict[str, Any]
    path: Optional[str] = None
    output_file: Optional[str] = None
    elapsed_secs: Optional[float] = None

    @property
    def dir_name(self) -> str:
        return f"run_{self.name}_{self.timestamp}"

    @property
    def start_datetime(self) -> datetime:
        return datetime.strptime(self.timestamp, DATE_FORMAT)

    @property
    def start(self) -> str:
        return self.start_datetime.strftime("%Y-%m-%d %H:%M:%S")


DATE_FORMAT = "%Y%m%d_%H%M%S"
logger = logging.getLogger(__name__)
_output_root = ".out"

# Initialize the current run record from the command-line arguments.
repo = RepoStatus()
_current_run = RunRecord(
    name=os.path.splitext(os.path.basename(sys.argv[1]))[0],
    args=sys.argv[1:],
    timestamp=datetime.now().strftime(DATE_FORMAT),
    git={
        "branch": repo.branch,
        "commit": repo.commit,
        "modified_files": [f.name for f in repo.all_modified_files],
    },
)


def set_output_root(root: str) -> None:
    global _current_run, _output_root
    if _current_run.path is not None:
        raise ValueError("Output root cannot be changed after run output has started.")
    _output_root = root


def get_run_path() -> str:
    global _current_run, _output_root
    # Create the run output path the
    # first time it is accessed.
    if _current_run.path is None:
        _current_run.path = os.path.join(_output_root, _current_run.dir_name)
        os.makedirs(_current_run.path)
        # Write out the RunRecord in json format to "run.json"
        with open(os.path.join(_current_run.path, "run.json"), "w") as f:
            f.write(_current_run.json(indent=4))
    return _current_run.path


def set_run_complete(output_file: Optional[str]) -> None:
    global _current_run
    if _current_run.elapsed_secs is not None:
        raise ValueError("Run already marked complete.")
    if output_file and not _current_run.path:
        raise ValueError("Cannot set output file without a persisted run path.")

    # Write out the final updated RunRecord to "run.json" if the run path is set.
    _current_run.elapsed_secs = round(datetime.now().timestamp() - _current_run.start_datetime.timestamp(), 1)
    _current_run.output_file = os.path.relpath(output_file, _current_run.path) if output_file else None
    if _current_run.path:
        with open(os.path.join(_current_run.path, "run.json"), "w") as f:
            f.write(_current_run.json(indent=4))

    if files := len([f for f in repo.all_modified_files if f.name.startswith("lib/")]):
        logger.warning(f"{files} modified {'file' if files == 1 else 'files'} in 'lib'.")
    logger.info(f"Run complete in {_current_run.elapsed_secs} secs on branch {repo.branch} ({repo.commit[:7]})")


def get_previous_run(name: Optional[str] = None) -> Optional[RunRecord]:
    global _current_run, _output_root
    prefix = f"run_{name or _current_run.name}_"

    # Get the set of matching prior runs and return the most recent, if any.
    if not (runs := [d for d in os.listdir(_output_root) if d.startswith(prefix) and d != _current_run.dir_name]):
        return None

    # Read in the most recent run record.
    run_path = os.path.join(_output_root, sorted(runs, reverse=True)[0])
    with open(os.path.join(run_path, "run.json"), "r") as f:
        return RunRecord.parse_raw(f.read())
