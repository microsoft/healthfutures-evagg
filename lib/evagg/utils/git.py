# -*- coding: utf-8 -*-
"""A class containting wrappers for interacting with git from the command line and parsing the output produced.

Classes:
    Repo: Represents a git repository, and contains several methods for performing actions on the repo.
    RepoStatus: Class for obtaining and holding the status of the repo.

Functions:
    is_git_installed: Returns True if git is installed on the system.
    get_git_version: Returns the version string from git.

Revision History:
    =========== =============== ==================================================================================================
    Date        Author          Revision
    =========== =============== ==================================================================================================
    08-08-2018  gabe            Initial implementation
    08-10-2018  gabe            Added contains() to Repo
    08-13-2018  gabe            Added Repo.fast_forward() and Repo.discard_changes()
    09-25-2018  gabe            Added Repo.get_status_str()
    10-04-2018  gabe            Migrated to python 3
    =========== =============== ==================================================================================================

"""

import os
import os.path
import re
import subprocess
from typing import List, Optional, Union


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
                r"1 (?P<XY>\S{2}) (?P<sub>\S{4}) (?P<mH>\S+) (?P<mI>\S+) (?P<mW>\S+) (?P<hH>\S+) (?P<hI>\S+) " + r"(?P<path>.+)",
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

    def __init__(self, repo_root_path: str) -> None:
        """Get the status of the specified repo and fill out all attributes of the class.

        Args:
            repo_root_path (str): Path to root directory of a Git repository to get the status of.

        Raises:
            GitError: If unable to run git status on the repo.

        """
        self._repo_root = repo_root_path

        # Run git status on the repo.
        try:
            self.status_str = subprocess.check_output(
                ["git", "status", "--porcelain=2", "--branch"], stderr=subprocess.STDOUT, cwd=self._repo_root
            ).decode()
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


class Repo:
    """Class representing a git repository.

    Contains several methods for performing actions on the repo.

    Attributes:
        root_path (str): The absolute path to the root of the repo.

    Static Methods:
        clone: Clone the specified repo and returns a Repo object.
        validate_repo: Raise an exception if `root_path` is not the root of a git repo.

    Methods:
        get_status: Return the status of the repo.
        get_status_str: Return a multiline string containing a verbose repo status.
        fetch: Fetch changes from remote repos.
        pull: Update the current branch with any changes on the remotes.
        fast_forward: Advance the current branch pointer ahead to the specified target postition.
        checkout: Checkout a specific branch or commit.
        lookup_commit_hash: Return SHA-1 commit hash associated with the specified reference (branch, tag, or commit).
        contains: Return True if the repo contains the specified reference (branch, tag, or commit).
        discard_changes: Remove pending changes (staged and unstaged), and optionally untracked files.

    """

    @classmethod
    def clone(cls, repo_url: str, dest_dir: str = None, dir_name: str = None) -> "Repo":
        """Clone the specified repo and returns a Repo object.

        Args:
            repo_url: Remote git URL for repo to clone.
            dest_dir: Directory to clone the repo into. If not specified, the current working directory is used.
            dir_name: Name of directory within `dest_dir` containing the repo. If not specified, this will be repo name.

        Returns:
            Repo: Repo object for the new repo clone.

        Raises:
            GitError: If git clone fails.

        """
        cwd = dest_dir or os.getcwd()
        if not os.path.isdir(cwd):
            raise GitError(f"The specified dest_dir does not exist: {cwd}")
        args: List[str] = [repo_url]
        if dir_name:
            args.append(dir_name)
        try:
            output = subprocess.check_output(["git", "clone"] + args, stderr=subprocess.STDOUT, cwd=cwd).decode()
            match = re.search("Cloning into '(.*)'", output)
            if match:
                containing_dir: str = match.group(1)
            else:
                raise GitError("Unable to determine containing directory")
        except Exception as err:
            raise GitError(f"Failed to clone repo: {err}")
        return cls(os.path.join(cwd, containing_dir))

    @staticmethod
    def validate_repo(root_path: str) -> None:
        """Raise an exception if `root_path` is not the root of a git repo."""
        if not os.path.isdir(root_path):
            raise IOError(f"The specified root_path is not a directory: {root_path}")
        if not os.path.isdir(os.path.join(root_path, ".git")):
            raise GitError(f"The specified root_path is not the root of a git repo: {root_path}")

    def __init__(self, root_path: str = ".") -> None:
        """Initialize Repo object with the specified path.

        Args:
            root_path (str): File path to repo root. Defaults to current working directory.

        Raises:
            IOError: If the root_path is not a directory.
            GitError: If git cannot be run or the root_path is not the root of a repo.

        """
        # Check that root path points to an existant git repo.
        Repo.validate_repo(root_path)
        self.root_path = os.path.abspath(root_path)
        """The absolute path to the root of the repo."""

        if not is_git_installed():
            raise GitError("Git is not installed")

        # Test running git status. This will raise a GitError if it fails.
        self.get_status()

    def __repr__(self) -> str:
        """Return string representation."""
        return f'Repo("{self.root_path}")'

    def get_status(self) -> RepoStatus:
        """Return the current status of the repo."""
        return RepoStatus(self.root_path)

    def get_status_str(self, include_staged_diff: bool = False, include_all_diff: bool = False) -> str:
        """Return a multiline string containing a verbose repo status.

        Args:
            include_staged_diff (bool): If True, the diff of staged changes are included. Defaults to False.
            include_all_diff (bool): If True, then the diff of both staged changes and working tree changes are included.
                Defaults to False.

        Raises:
            GitError: If unable to run git status on the repo.

        """
        try:
            args: List[str] = []
            if include_all_diff:
                args.append("-vv")
            elif include_staged_diff:
                args.append("-v")
            return subprocess.check_output(["git", "status"] + args, stderr=subprocess.STDOUT, cwd=self.root_path).decode()
        except:
            raise GitError("Unable to run git status on the specified repo")

    def fetch(self, remotes: Union[List[str], None] = None, all_tags: Union[bool, None] = False) -> str:
        """Fetch the latest repo tree and references from the selected remotes.

        Args:
            remotes (List[str] or None): A list of remotes to update. If None, then all remotes will be fetched.
            all_tags (bool or None): If True all tags are fetched ('--tags' option), if False only tags that point into the
                histories being fetched are downloaded (default option), and if None then no tags are fetched ('--no-tags' option)

        Returns:
            str: Command line output from 'git fetch'.

        Raises:
            GitError: If call to 'get fetch' fails.

        """
        # Determine command line args based on remotes parameter.
        args: List[str] = []
        if not remotes:
            args.append("--all")
        elif len(remotes) == 1:
            args.append(remotes[0])
        else:
            args.append("--multiple")
            args.extend(remotes)

        # Determine command line args based on remotes parameter.
        if all_tags is None:
            args.append("--no-tags")
        elif all_tags:
            args.append("--tags")

        try:
            output = subprocess.check_output(["git", "fetch"] + args, stderr=subprocess.STDOUT, cwd=self.root_path).decode()
        except subprocess.CalledProcessError as err:
            raise GitError(f"Git fetch returned an error code: {err.output}")
        return output

    def pull(self) -> str:
        """Pull latest changes from remote and update the current branch with any changes from remote.

        Returns:
            str: Command line output from 'git pull'.

        Raises:
            GitError: If call to 'git pull' fails.

        """
        try:
            output = subprocess.check_output(["git", "pull"], stderr=subprocess.STDOUT, cwd=self.root_path).decode()
        except subprocess.CalledProcessError as err:
            raise GitError(f"Git pull returned an error code: {err.output}")
        return output

    def fast_forward(self, target: str = "") -> str:
        """Advance the current branch pointer ahead to the specified target position.

        Args:
            target (str): Commit representing the final position after the fast-forward. If not specified, then this will be the
                head of the upstream (remote) branch.

        Returns:
            str: Command line output from 'git merge'.

        Raises:
            GitError: If call to 'git merge' fails.

        """
        try:
            output = subprocess.check_output(
                ["git", "merge", "--ff-only", target], stderr=subprocess.STDOUT, cwd=self.root_path
            ).decode()
        except subprocess.CalledProcessError as err:
            raise GitError(f"Git merge returned an error code: {err.output}")
        return output

    def checkout(self, destination: str) -> str:
        """Checkout the specified branch or commit.

        Args:
            destination (str): Name of branch or commit hash to checkout.

        Returns:
            str: Command line output from 'git checkout'.

        Raises:
            GitError: If call to 'git checkout' fails.

        """
        try:
            output = subprocess.check_output(
                ["git", "checkout", destination], stderr=subprocess.STDOUT, cwd=self.root_path
            ).decode()
        except subprocess.CalledProcessError as err:
            raise GitError(f"Git checkout returned an error code: {err.output}")
        return output

    def lookup_commit_hash(self, ref: str, include_remotes: bool = False) -> str:
        """Return SHA-1 commit hash associated with the specified reference (branch, tag, or commit).

        The reference name must be the complete name. For example if `ref`='my-branch' and that branch name does not exist,
            but 'feature/my-branch' does exist, then an exception will be raised since the specified reference does not exist.

        Using the `include_remotes` option, branches present on the remotes will also be searched. For example, if
            `ref`='feature/my-branch' and 'feature/my-branch' does not exist, but 'origin/feature/my-branch' does exist,
            the behavior is based on the value of `include_remotes`. If `include_remotes` is True, then a SHA-1 hash is returned,
            but if `include_remotes` is False (default), then an exception is raised since the specified reference name does not
            exist in the local repo.

        Args:
            ref (str): Reference to lookup. This can be a branch, tag, or SHA-1 commit hash.
            include_remotes (bool): If True, the remotes are also searched for the specified branch name. Defaults to False.

        Returns:
            str: SHA-1 commit hash associated with the specified reference (branch, tag, or commit).

        Raises:
            GitError: If the specified reference cannot be found in the repo.

        """
        # First search the local repo only.
        commit = self._get_commit_hash(ref)

        # If not found and we're allowed to search remotes, then search remotes to find full name of branch.
        if not commit and include_remotes:
            try:
                branch_names = subprocess.check_output(
                    ["git", "rev-parse", "--symbolic-full-name", f"--glob=*/{ref}"], stderr=subprocess.STDOUT, cwd=self.root_path
                ).decode()

                # Parse output from 'git rev-parse' to find the branch name that exactly matches the reference plus a remote name.
                for branch in branch_names.splitlines():
                    if re.match(f"refs/remotes/[^/]+/{ref}", branch):
                        commit = self._get_commit_hash(branch)
                        break

            except subprocess.CalledProcessError:  # Git search for similar branch names failed.
                commit = ""

        if not commit:
            raise GitError(f"The specified reference cannot be found in the repo: {ref}")
        return commit

    def contains(self, ref: str, include_remotes: bool = False) -> bool:
        """Return True if the repo contains the specified reference (branch, tag, or commit).

        The reference name must be the complete name. For example if `ref`='my-branch' and that branch name does not exist,
            but 'feature/my-branch' does exist, then False is returned since the specified reference does not exist.

        Using the `include_remotes` option, branches present on the remotes will also be searched. For example, if
            `ref`='feature/my-branch' and 'feature/my-branch' does not exist, but 'origin/feature/my-branch' does exist,
            the behavior is based on the value of `include_remotes`. If `include_remotes` is True, then True is returned,
            but if `include_remotes` is False (default), then False is returned since the specified reference name does not
            exist in the local repo.

        Args:
            ref (str): Reference to lookup. This can be a branch, tag, or SHA-1 commit hash.
            include_remotes (bool): If True, the remotes are also searched for the specified branch name. Defaults to False.

        Returns:
            bool: True if the repo contains the specified reference (branch, tag, or commit), False otherwise.

        """
        try:
            self.lookup_commit_hash(ref, include_remotes)
        except GitError:
            return False
        return True

    def discard_changes(
        self,
        remove_untracked_files: bool = False,
        remove_untracked_dirs: bool = False,
        remove_ignored_files: bool = False,
    ) -> None:
        """Remove pending changes (staged and unstaged), and optionally untracked files and/or ignore files.

        Args:
            remove_untracked_files (bool): If True, all untracked files will also be removed. Defaults to False.
            remove_untracked_dirs (bool): If True, all untracked directories will also be removed.
                If True, `remove_untracked_files` must also be True. Defaults to False.
            remove_ignored_files (bool): If True, all files ignored by .gitignore will also be removed. Defaults to False.

        Raises:
            ValueError: If invalid arguments are given.
            GitError: If call to 'git reset' or 'git clean' fails.

        """
        if remove_untracked_dirs and not remove_untracked_files:
            raise ValueError("remove_untracked_dirs cannot be True if remove_untracked_files is False")

        # There's nothing to do if the repo is already clean and we're not cleaning ignored files.
        if not remove_ignored_files and self.get_status().is_clean(include_untracked_files=remove_untracked_files):
            return

        # Run 'git reset' to discard staged and unstaged changes.
        try:
            subprocess.check_output(["git", "reset", "--hard"], stderr=subprocess.STDOUT, cwd=self.root_path).decode()
        except subprocess.CalledProcessError as err:
            raise GitError(f"Git reset returned an error code: {err.output}")

        # Remove all untracked files if requested.
        if remove_untracked_files or remove_ignored_files:
            try:
                flags = "-f"
                if remove_untracked_dirs:
                    flags += "d"
                if remove_ignored_files:
                    flags += "x" if remove_untracked_files else "X"
                subprocess.check_output(["git", "clean", flags], stderr=subprocess.STDOUT, cwd=self.root_path).decode()
            except subprocess.CalledProcessError as err:
                raise GitError(f"Git clean returned an error code: {err.output}")

        assert self.get_status().is_clean(include_untracked_files=remove_untracked_dirs)

    def _get_commit_hash(self, ref: str) -> str:
        """Convert a reference (branch, tag, or commit) into an SHA-1 commit hash.

        Args:
            ref (str): Reference to lookup. This can be a branch, tag, or SHA-1 commit hash.

        Returns:
            str: SHA-1 commit hash associated with the specified reference if the reference is found in the local repo.
                Empty string is the reference is not found.

        """
        try:
            # To determine if ref is can be mapped to a valid commit, use the following git command:
            #   git rev-parse --verify "$VAR^{commit}" where $VAR is the commit-ish reference to check.
            #   See: --verify option of rev-parse: https://git-scm.com/docs/git-rev-parse.
            commit = subprocess.check_output(
                ["git", "rev-parse", "--verify", ref + "^{commit}"], stderr=subprocess.STDOUT, cwd=self.root_path
            ).decode()
        except subprocess.CalledProcessError:
            commit = ""
        return commit.strip()


def is_git_installed() -> bool:
    """Return True if git is installed on system, False otherwise."""
    return len(get_git_version()) > 0


def get_git_version() -> str:
    """Return the version string from git.

    An empty string is returned if git is not installed or cannot be run.

    """
    try:
        version = subprocess.check_output(["git", "--version"]).decode()
    except:
        version = ""
    return version
