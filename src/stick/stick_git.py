import subprocess
import tempfile
import os
from typing import Optional, Any
import warnings
from stat import S_ISREG

import git

EIGHT_MEBIBYTES = 8 * 2**20
STICK_CHECKPOINTS_BRANCH = "stick-checkpoints"


def checkpoint_repo(
    run_dir: str, launcher_path: Optional[str] = None
) -> dict[str, Any]:
    """Find the git repo, starting from __main__ or launcher_path (if
    provided), and create a new git commit on the stick-checkpoints branch.

    Also writes out run_dir/from_head.diff and
    run_dir/from_last_checkpoint.diff, for easily visualizing changes present
    in this experiment launch.

    Returns a dictionary of metadata, which will be empty on failure.
    """

    git_root_path = get_git_root(start_path=launcher_path)
    if git_root_path is None:
        warnings.warn("Could not find git root path. " "Code will not be checkpointed.")
        return {}
    repo = git.Repo(git_root_path)
    if STICK_CHECKPOINTS_BRANCH in repo.heads:
        checkpoint_branch = repo.heads[STICK_CHECKPOINTS_BRANCH]
    else:
        checkpoint_branch = repo.create_head(STICK_CHECKPOINTS_BRANCH)
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Ask gitpython to create an index for us in this temporary directory.
        index_path = os.path.join(tmp_dir, 'stick_git_index')
        index_file = git.IndexFile(repo, index_path)
        # There appears to be no way to add all modified files to an IndexFile
        # using gitpython.
        # Is this really better than just moving the index file and calling
        # `git add -a`?
        for f in get_modified_file_list(git_root_path):
            index_file.add(f)
        checkpoint_commit = index_file.commit(
            f"stick checkpoint of {run_dir}",
            [checkpoint_branch.commit], head=False, skip_hooks=True
        )
        checkpoint_branch.reference = checkpoint_commit
        print(
            f"Saved code to git commit f{checkpoint_commit.hexsha} on "
            f"branch {STICK_CHECKPOINTS_BRANCH}"
        )
        git_hash = checkpoint_commit.hexsha
        subprocess.run(
            (
                "git",
                "diff",
                "HEAD",
                checkpoint_commit.hexsha,
                "--histogram",
                "--output",
                os.path.join(run_dir, "from_head.diff"),
            ),
            cwd=git_root_path,
        )
        subprocess.run(
            (
                "git",
                "diff",
                f"{STICK_CHECKPOINTS_BRANCH}~1",
                checkpoint_commit.hexsha,
                "--histogram",
                "--output",
                os.path.join(run_dir, "from_last_checkpoint.diff"),
            ),
            cwd=git_root_path,
        )
        return {
            "git_root_path": git_root_path,
            "git_checkpoint_hash": git_hash,
        }


def get_git_root(start_path: Optional[str] = None) -> Optional[str]:
    if start_path is None:
        import __main__ as main

        start_path = getattr(main, "__file__", None)
    if start_path is None:
        start_path = os.getcwd()
    assert start_path is not None
    start_path = os.path.abspath(start_path)
    try:
        git_root_path = subprocess.check_output(
            ("git", "rev-parse", "--show-toplevel"),
            cwd=os.path.dirname(start_path),
            stderr=subprocess.DEVNULL,
        )
        git_root_path = git_root_path.strip()
    except subprocess.CalledProcessError:
        return None
    return git_root_path.decode()


def get_modified_file_list(git_root_path: str) -> list[str]:
    # List all non-excluded files
    # This is basically all files listed by `git status`, but separated by nul
    # characters.
    git_files = subprocess.check_output(
        ("git", "ls-files", "--exclude-standard", "--cached", "--others", "-z"),
        cwd=git_root_path,
    ).strip()
    repo_size = 0
    files_to_archive = []
    file_list = [os.path.join(git_root_path, f.decode()) for f in git_files.split(b"\0")]
    for f in file_list:
        try:
            file_stats = os.stat(f)
            file_size = file_stats.st_size
            repo_size += file_size
            if file_size < EIGHT_MEBIBYTES and S_ISREG(file_stats.st_mode):
                files_to_archive.append(f)
        except FileNotFoundError:
            pass
    if repo_size >= EIGHT_MEBIBYTES:
        warnings.warn(
            "Checkpointing over 8MiB of files. This may be "
            "slow. Set create_git_checkpoint=False in init_extra "
            "to disable this feature."
        )
    return files_to_archive
