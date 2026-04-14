import shutil
import subprocess
from pathlib import Path

from invoke import Context, task


# ================ Helper Functions ================= #
@task
def lint(c: Context) -> None:
    """Run linting checks."""
    result = 0
    print("Running Ruff check...")
    check_command = subprocess.run("uv run ruff check --fix src/.", shell=True)
    if check_command.returncode != 0:
        result += check_command.returncode
    print("\nRunning Ruff format check...")
    format_command = subprocess.run("uv run ruff format src/.", shell=True)
    if format_command.returncode != 0:
        result += format_command.returncode
    print("\nRunning mypy...")
    mypy_command = subprocess.run("uv run mypy src/.", shell=True)
    if mypy_command.returncode != 0:
        result += mypy_command.returncode
    if result != 0:
        print("❌ Linting issues found!")
    else:
        print("🔎 Linting Task Done!")


# ================ Tasks ================= #
CLEAN_DIRS: list[str] = [
    "build",  # Artéfacts de build
    "dist",  # Distributions packagées
    ".pytest_cache",  # Cache de pytest
    ".coverage",  # Données de couverture
    ".ruff_cache",  # Cache de Ruff
    ".mypy_cache",  # Cache de mypy (si utilisé)
    "__pycache__",  # Cache Python (racine)
]


@task
def clean(c: Context) -> None:
    """Remove build artifacts and caches."""
    for directory in CLEAN_DIRS:
        path: Path = Path(directory)
        if path.exists():
            print(f"  - Removing {directory}")
            shutil.rmtree(path, ignore_errors=True)

    # Nettoyer récursivement les __pycache__
    for path in Path(".").rglob("__pycache__"):
        print(f"  - Removing {path}")
        shutil.rmtree(path, ignore_errors=True)

    # Nettoyer les *.egg-info
    for path in Path(".").rglob("*.egg-info"):
        print(f"  - Removing {path}")
        shutil.rmtree(path, ignore_errors=True)

    # Nettoyer les fichiers .pyc
    for path in Path(".").rglob("*.pyc"):
        print(f"  - Removing {path}")
        path.unlink(missing_ok=True)

    print("🗑 Clean task Done!")
