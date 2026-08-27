import shutil
import subprocess
import webbrowser
from pathlib import Path

from invoke.context import Context
from invoke.tasks import task

# ================ Config ================= #
RELEASE_BRANCH = "main"  # branche de référence pour les releases
CLEAN_DIRS: list[str] = [
    "build",  # Artéfacts de build
    "dist",  # Distributions packagées
    "htmlcov",  # Rapport de couverture HTML (pytest-cov)
    "docs/build",  # Build Sphinx
    "docs/code/api",  # Pages d'API générées par sphinx-apidoc
    ".pytest_cache",  # Cache de pytest
    ".ruff_cache",  # Cache de Ruff
    ".mypy_cache",  # Cache de mypy
    "__pycache__",  # Cache Python (racine)
]
# Fichiers à supprimer séparément : rmtree échoue sur un fichier, et
# ignore_errors=True masquerait l'échec sans rien nettoyer.
CLEAN_FILES: list[str] = [
    ".coverage",  # Données de couverture
    "coverage.xml",  # Rapport de couverture XML
]


@task
def clean(c: Context) -> None:
    """Remove build artifacts, generated docs and caches."""
    for directory in CLEAN_DIRS:
        path: Path = Path(directory)
        if path.is_dir():
            print(f"  - Removing {directory}/")
            shutil.rmtree(path, ignore_errors=True)

    for filename in CLEAN_FILES:
        file_path = Path(filename)
        if file_path.is_file():
            print(f"  - Removing {filename}")
            file_path.unlink(missing_ok=True)

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


@task
def lint(c: Context) -> None:
    """Run all quality checks via pre-commit (source unique de vérité)."""
    print("🔎 Running pre-commit on all files...")
    result = subprocess.run("uv run pre-commit run --all-files", shell=True)
    if result.returncode != 0:
        print("❌ Linting issues found!")
        raise SystemExit(result.returncode)
    print("✅ Linting Task Done!")


@task
def precommit_install(c: Context) -> None:
    """Install the pre-commit hooks (stages pre-commit et commit-msg).

    Les deux stages sont nécessaires : le hook commitizen ne se déclenche
    que sur commit-msg, qui n'est pas installé par défaut par pre-commit.
    """
    print("🔧 Installing pre-commit hooks...")
    result = subprocess.run(
        "uv run pre-commit install --install-hooks -t pre-commit -t commit-msg",
        shell=True,
    )
    if result.returncode != 0:
        print("❌ pre-commit install failed!")
        raise SystemExit(result.returncode)
    print("✅ pre-commit hooks installed!")


@task
def test(c: Context, verbose: bool = False, coverage: bool = True) -> None:
    """Run the pytest test suite."""
    print("🧪 Running test suite...")

    cmd = "uv run python -m pytest"
    if verbose:
        cmd += " -v"
    if not coverage:
        # pyproject.toml active la couverture par défaut via addopts
        cmd += " --no-cov"

    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print("❌ Tests failed!")
        raise SystemExit(result.returncode)

    print("✅ All tests passed!")
    html_report = Path("htmlcov") / "index.html"
    if html_report.exists():
        print(f"📊 HTML report: {html_report.resolve()}")


@task
def docs(c: Context, open_browser: bool = False) -> None:
    """Build the Sphinx documentation as HTML.

    Les pages d'API sont (re)générées automatiquement par le hook
    `builder-inited` déclaré dans docs/code/conf.py : un nouveau module sous
    src/ est ainsi documenté sans qu'aucun fichier .rst n'ait à être écrit ou
    maintenu à la main. Le dossier généré (docs/code/api/) n'est pas versionné.

    Cette tâche n'appelle donc PAS sphinx-apidoc elle-même : la génération vit
    dans conf.py pour que la CI, qui invoque `sphinx-build` directement,
    produise exactement la même documentation qu'en local.

    `-W --keep-going` reproduit le comportement de la CI : tout avertissement
    devient une erreur, mais le build va au bout pour tous les remonter.
    """
    src = Path("docs/code")
    out = Path("docs/build") / "html"
    out.mkdir(parents=True, exist_ok=True)

    print("📖 Building Sphinx documentation...")
    result = subprocess.run(
        f'uv run sphinx-build -W --keep-going -b html "{src}" "{out}"',
        shell=True,
    )
    if result.returncode != 0:
        print("❌ Sphinx build failed!")
        raise SystemExit(result.returncode)

    index_html = out / "index.html"
    print(f"✅ Documentation built: {index_html.resolve()}")

    if open_browser:
        webbrowser.open(index_html.as_uri())


@task
def build(c: Context) -> None:
    """Build the package (wheel + sdist) via uv."""
    print("📦 Building package...")
    result = subprocess.run("uv build", shell=True)
    if result.returncode != 0:
        print("❌ Build failed!")
        raise SystemExit(result.returncode)
    print("✅ Build Done! Artifacts in dist/")


@task
def release(
    c: Context,
    part: str = "patch",
    dry_run: bool = False,
    skip_tests: bool = False,
) -> None:
    """Orchestrate a release: bump version, changelog, commit, tag, push.

    Sequence:
      1. Verify we are on the main branch with a clean working tree
      2. Run lint + tests (safety before publishing) — unless --skip-tests
      3. Bump the version via commitizen (patch | minor | major)
      4. Create a release commit + a signed vX.Y.Z tag
      5. Push the commit AND the tag to origin/main

    Usage:
        inv release                  # bump patch (0.1.0 → 0.1.1)
        inv release --part=minor     # bump minor (0.1.0 → 0.2.0)
        inv release --part=major     # bump major (0.1.0 → 1.0.0)
        inv release --dry-run        # simulate without modifying anything
        inv release --skip-tests     # for local development only
    """

    def _run(cmd: str, check: bool = True) -> int:
        """Execute a shell command, display the result, return the exit code."""
        if dry_run:
            print(f"[dry-run] {cmd}")
            return 0
        result = subprocess.run(cmd, shell=True)
        if check and result.returncode != 0:
            print(f"\n❌ Failed: {cmd}")
            raise SystemExit(result.returncode)
        return result.returncode

    print("\n🔍 Release workflow")
    print(f"  Bump type  : {part}")
    print(f"  Dry run    : {dry_run}")
    print(f"  Branch     : {RELEASE_BRANCH}")

    # ── Step 0: pre-flight checks ─────────────────────────────────────────────
    print("\n📦 Step 1/5: pre-flight checks...")

    current_branch = subprocess.run(
        "git rev-parse --abbrev-ref HEAD", shell=True, capture_output=True, text=True
    ).stdout.strip()
    if not dry_run and current_branch != RELEASE_BRANCH:
        print(f"❌ You are on '{current_branch}', not on '{RELEASE_BRANCH}'.")
        print(f"   Run 'git checkout {RELEASE_BRANCH}' before releasing.")
        raise SystemExit(1)

    dirty = subprocess.run(
        "git status --porcelain", shell=True, capture_output=True, text=True
    ).stdout.strip()
    if not dry_run and dirty:
        print("❌ Working tree is not clean. Commit or stash first.")
        print(dirty)
        raise SystemExit(1)

    print("✅ Branch and working tree OK")

    # ── Step 1: lint + tests ──────────────────────────────────────────────────
    if not skip_tests:
        print("\n📦 Step 2/5: lint + tests...")
        _run("uv run pre-commit run --all-files")
        _run("uv run python -m pytest -q")
        print("✅ Lint and tests OK")
    else:
        print("\n⚠️  Step 2/5: lint + tests skipped (--skip-tests)")

    # ── Step 2: bump version + changelog ─────────────────────────────────────
    print(f"\n📦 Step 3/5: bump version ({part})...")
    bump_cmd = f"uv run cz bump --increment {part.upper()}"
    if dry_run:
        bump_cmd += " --dry-run"
        # cz bump --dry-run est lui-même non destructif, on l'exécute donc
        # directement plutôt que de laisser _run l'absorber — c'est ce qui
        # affiche réellement la version cible + l'aperçu du changelog.
        preview = subprocess.run(bump_cmd, shell=True)
        if preview.returncode != 0:
            print(f"\n❌ Failed: {bump_cmd}")
            raise SystemExit(preview.returncode)
    else:
        _run(bump_cmd)
    print("✅ Version bumped + CHANGELOG.md updated")

    # En dry-run rien n'a réellement été bump, donc c'est toujours la version
    # courante — ne pas la présenter comme la nouvelle version.
    new_version = subprocess.run(
        "uv run cz version --project", shell=True, capture_output=True, text=True
    ).stdout.strip()
    if not dry_run:
        print(f"   New version: {new_version}")

    # ── Step 3: push commit + tag ─────────────────────────────────────────────
    print("\n📦 Step 4/5: push commit + tag...")
    _run(f"git push origin {RELEASE_BRANCH}")
    _run(f"git push origin v{new_version}")
    print("✅ Commit + tag pushed")

    # ── Step 4: summary ───────────────────────────────────────────────────────
    print("\n📦 Step 5/5: summary...")
    print(f"🎉 Release v{new_version} launched!")
