# -*- coding: utf-8 -*-
"""A class containting wrappers for interacting with git from the command line and parsing the output produced.

Classes:
    Repo: Represents a git repository, and contains several methods for performing actions on the repo.
    RepoStatus: Class for obtaining and holding the status of the repo.

Functions:
    is_git_installed: Returns True if git is installed on the system.
    get_git_version: Returns the version string from git.

Revision History:
    =========== =============== ========================================================================================
    Date        Author          Revision
    =========== =============== ========================================================================================
    08-08-2018  gabe            Initial implementation
    08-10-2018  gabe            Added contains() to Repo
    08-13-2018  gabe            Added Repo.fast_forward() and Repo.discard_changes()
    09-25-2018  gabe            Added Repo.get_status_str()
    10-04-2018  gabe            Migrated to python 3
    05-24-2024  gregsmi         Copied in, stripped down, and modified for use in evagg
    =========== =============== ========================================================================================

"""

import os
import os.path
import re
import subprocess  # nosec B404 # allow careful use of subprocess
from typing import List, Optional


class GitError(Exception):
    """Error performing git operation on the command line."""

    pass


class ModifiedFile:
    """Container class with information about a modified, renamed, unmerged, untacked, or ignored file in a git repo.

    Attributes:
        name (str): Name of file. If this is a renamed file, then this is the new name.
        type (str): The type of modified file. It will be one of the following:
            'changed', 'renamed', 'unmerged', 'untracked', 'ignored'
        fields (Dict[str, str]): All fields parsed from status line.
        status_line (str): Full status line returned from 'git status --porcelain=2' command.

    """

    def __init__(self, status_line: str) -> None:
        """Initialize class with a single status line from 'git status --porcelain=2' command.

        Args:
            status_line (str): Single status line from 'git status --porcelain=2' command.

        Raises:
            ValueError: If cannot parse the status line.

        """
        if not status_line:
            raise ValueError(f"Invalid git status line: {status_line}")
        self.status_line = status_line

        # Parse the line based on the type, which is encoded in the first character.
        if status_line.startswith("1"):
            self.type = "changed"
            match = re.match(
                r"1 (?P<XY>\S{2}) (?P<sub>\S{4}) (?P<mH>\S+) (?P<mI>\S+) (?P<mW>\S+) (?P<hH>\S+) (?P<hI>\S+) "
                + r"(?P<path>.+)",
                status_line,
            )
        if status_line.startswith("2"):
            self.type = "renamed"
            match = re.match(
                r"2 (?P<XY>\S{2}) (?P<sub>\S{4}) (?P<mH>\S+) (?P<mI>\S+) (?P<mW>\S+) (?P<hH>\S+) (?P<hI>\S+) "
                + r"(?P<X>[R,C])(?P<score>\d+) (?P<path>[^\t]+)\t(?P<origPath>.+)",
                status_line,
            )
        if status_line.startswith("u"):
            self.type = "unmerged"
            match = re.match(
                r"u (?P<XY>\S{2}) (?P<sub>\S{4}) (?P<m1>\S+) (?P<m2>\S+) (?P<m3>\S+) (?P<mW>\S+) (?P<h1>\S+) "
                + r"(?P<h2>\S+) (?P<h3>\S+) (?P<path>.+)",
                status_line,
            )
        if status_line.startswith("?"):
            self.type = "untracked"
            match = re.match(r"\? (?P<path>.+)", status_line)
        if status_line.startswith("!"):
            self.type = "ignored"
            match = re.match(r"! (?P<path>.+)", status_line)

        # If parse failed, raise an error.
        if not match:
            raise ValueError(f"Unable to parse git status line: {status_line}")

        # Save parsed fields.
        self.fields = match.groupdict()

    @property
    def name(self) -> str:
        """Name of file. If this is a renamed file, then this is the new name."""
        return self.fields["path"]

    def __repr__(self) -> str:
        """Return string representation."""
        return f'ModifiedFile("{self.status_line}")'


class RepoStatus:
    """Class for obtaining and holding the status of the repo.

    Attributes:
        commit (str): Current SHA-1 commit hash.
        branch (str): Current branch. '(detached)' is commit is detached from a branch.
        upstream (str): Current upstream branch. Empty string if no upstream is set.
        upstream_ahead (int): Number of commits ahead of the upstream branch. 0 if no upstream is set.
        upstream_behind (int): Number of commits behind of the upstream branch. 0 if no upstream is set.
        changed_files (List[ModifiedFile]): List of added, deleted, and modified either staged or unstaged.
        renamed_files (List[ModifiedFile]): List of renamed or copied files either staged or unstaged.
        unmerged_files (List[ModifiedFile]): List of unmerged files.
        untracked_files (List[ModifiedFile]): List of untracked files.
        ignored_files (List[ModifiedFile]): List of ignored files.
        all_modified_files (List[ModifiedFile]): List of all modified files, including those in the above lists.
        status_str (str): Full string returned from 'git status --porcelain=2 --branch' command.

    Methods:
        is_clean: Determine whether or not the repo is clean.
        is_up_to_date: Determine whether or not the repo is ahead/behind the upstream branch.

    """

    def __init__(self, repo_root_path: Optional[str] = None) -> None:
        """Get the status of the specified repo and fill out all attributes of the class.

        Args:
            repo_root_path (str): Path to root directory of a Git repository to get the status of. Searches up the
            current directory tree if not specified.

        Raises:
            GitError: If unable to run git status on the repo.

        """
        if not repo_root_path:
            repo_root_path = os.getcwd()
            while not os.path.isdir(os.path.join(repo_root_path, ".git")):
                repo_root_path = os.path.dirname(repo_root_path)
                if repo_root_path == "/":
                    raise GitError("No git repo found in the current directory or any parent directory")
        self._repo_root = repo_root_path

        # Run git status on the repo.
        try:
            status = subprocess.check_output(
                ["git", "status", "--porcelain=2", "--branch"], stderr=subprocess.STDOUT, cwd=self._repo_root
            )  # nosec B603, B607 # no untrusted input/partial path
            self.status_str = status.decode()
        except Exception:
            raise GitError("Unable to run git status on the specified repo")

        # Initialize all attributes to their default values.
        self.commit = ""
        self.branch = ""
        self.upstream = ""
        self.upstream_ahead = 0
        self.upstream_behind = 0
        self.all_modified_files: List[ModifiedFile] = []

        # Parse git status output.
        for line in self.status_str.splitlines():
            # Try to parse commit line.
            match = re.match(r"# branch\.oid (.*)", line)  # suglint: ignore
            if match:
                self.commit = match.group(1)
                continue

            # Try to parse branch line.
            match = re.match(r"# branch\.head (.*)", line)  # suglint: ignore
            if match:
                self.branch = match.group(1)
                continue

            # Try to parse upstream line.
            match = re.match(r"# branch\.upstream (.*)", line)  # suglint: ignore
            if match:
                self.upstream = match.group(1)
                continue

            # Try to parse upstream ahead/behind.
            match = re.match(r"# branch\.ab \+(\d+) -(\d+)", line)  # suglint: ignore
            if match:
                self.upstream_ahead = int(match.group(1))
                self.upstream_behind = int(match.group(2))
                continue

            # The line must represent a modified file.
            self.all_modified_files.append(ModifiedFile(line))

    @property
    def changed_files(self) -> List[ModifiedFile]:
        """List of added, deleted, and modified either staged or unstaged."""
        return [mod_file for mod_file in self.all_modified_files if mod_file.type == "changed"]

    @property
    def renamed_files(self) -> List[ModifiedFile]:
        """List of renamed or copied files either staged or unstaged."""
        return [mod_file for mod_file in self.all_modified_files if mod_file.type == "renamed"]

    @property
    def unmerged_files(self) -> List[ModifiedFile]:
        """List of unmerged files."""
        return [mod_file for mod_file in self.all_modified_files if mod_file.type == "unmerged"]

    @property
    def untracked_files(self) -> List[ModifiedFile]:
        """List of untracked files."""
        return [mod_file for mod_file in self.all_modified_files if mod_file.type == "untracked"]

    @property
    def ignored_files(self) -> List[ModifiedFile]:
        """List of ignored files."""
        return [mod_file for mod_file in self.all_modified_files if mod_file.type == "ignored"]

    def __repr__(self) -> str:
        """Return string representation."""
        return f'RepoStatus("{self._repo_root}"):\n{self.status_str}'

    def is_clean(self, include_untracked_files: bool = True) -> bool:
        """Return True if the repo is clean, and False otherwise.

        Args:
            include_untracked_files (bool): If True, then the presence of untracked files will return False.
                If False, then untracked files will not be considered.

        Returns:
            bool: True if the repo is clean, and False otherwise.

        """
        if include_untracked_files and self.untracked_files:
            return False
        return not self.changed_files and not self.renamed_files and not self.unmerged_files

    def is_up_to_date(self) -> bool:
        """Return True if the repo is not ahead or behind the upstream branch.

        If no upstream branch is set, then this returns True.

        """
        return not self.upstream_ahead and not self.upstream_behind
