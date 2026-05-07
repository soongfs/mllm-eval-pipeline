import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def read_git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
