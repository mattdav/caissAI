# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

from sphinx.application import Sphinx

# Le package est exposé via src/ : c'est ce dossier qu'il faut ajouter au
# path pour qu'autodoc puisse importer ``caissAI`` et ses sous-modules.
sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "caissAI"
copyright = "2025, Matthieu Daviaud"
author = "Matthieu Daviaud"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.githubpages",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

# Docstrings au format Google
napoleon_google_docstring = True
napoleon_numpy_docstring = False

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []


# Pour gérer __main__ spécifiquement
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": "__init__",
}

# maia2/einops/torch sont des dépendances lourdes non requises pour builder
# la documentation (ex. workflow docs.yml) : mockées pour éviter de les
# installer uniquement pour générer les pages d'API.
autodoc_mock_imports = ["maia2", "einops", "torch"]


# -- Génération automatique des pages d'API ----------------------------------
#
# `docs/code/api/` est généré, donc non versionné. Si la génération n'était
# appelée que depuis `inv docs`, la CI — qui invoque `sphinx-build`
# directement — publierait une documentation amputée de toute son API, sans
# échouer pour autant. Brancher la génération sur l'événement `builder-inited`
# garantit que TOUT build la déclenche, d'où qu'il vienne. C'est aussi la
# raison pour laquelle `inv docs` n'appelle plus `sphinx-apidoc` lui-même :
# une seule source de vérité.


def _run_apidoc(app: Sphinx) -> None:
    """Génère les pages d'API avant chaque build, en local comme en CI."""
    from pathlib import Path

    from sphinx.ext.apidoc import main

    package = Path(__file__).parent.parent.parent / "src" / "caissAI"
    output = Path(__file__).parent / "api"
    main(["--force", "--separate", "--module-first", "-o", str(output), str(package)])


def setup(app: Sphinx) -> None:
    """Enregistre la génération d'API sur l'événement `builder-inited`."""
    app.connect("builder-inited", _run_apidoc)
