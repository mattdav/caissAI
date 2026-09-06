"""Résolution des chemins runtime de caissAI.

Le programme n'écrit jamais dans le package installé : une réinstallation
ou un ``uv tool upgrade`` remplacerait le contenu du package et effacerait
les PGN traités et les journaux applicatifs.

Tous les emplacements suivent la même précédence :
``argument CLI > variable d'environnement > défaut utilisateur``.

Variables reconnues :
    CAISSAI_PGN        Fichier PGN traité par défaut (équivaut à --pgn)
    CAISSAI_DATA_DIR   Dossier data/ (input/ et output/), hors du package installé
    CAISSAI_LOG_DIR    Dossier des journaux applicatifs
    CAISSAI_MAIA2_DIR  Dossier des poids Maia2 (``blitz_model.pt``)

``config/lichess_eco.parquet`` reste packagé : c'est une donnée de référence
livrée avec le code, jamais écrite à l'exécution, donc hors du périmètre de
ce module (voir ``utils.get_folder_path``).
"""

import os
from pathlib import Path


def default_state_dir() -> Path:
    """Dossier d'état par défaut, situé hors du package installé.

    Returns:
        ``%LOCALAPPDATA%/caissAI`` sous Windows,
        ``~/.local/share/caissAI`` ailleurs.

    Examples:
        >>> default_state_dir().name
        'caissAI'
    """
    local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
    if local_appdata:
        return Path(local_appdata) / "caissAI"
    return Path.home() / ".local" / "share" / "caissAI"


def resolve_path(cli_value: str | None, env_var: str, default: Path) -> Path:
    """Résout un chemin selon la précédence CLI > environnement > défaut.

    S'applique indifféremment à un dossier ou à un fichier : la fonction
    ne fait que choisir une valeur et normaliser ``~``, sans rien créer
    ni vérifier l'existence.

    Args:
        cli_value: Valeur passée en argument CLI (prioritaire), ou None.
        env_var: Nom de la variable d'environnement à consulter ensuite.
        default: Valeur de repli si ni l'un ni l'autre n'est renseigné.

    Returns:
        Chemin résolu, avec ``~`` développé.

    Examples:
        >>> resolve_path("/tmp/cli", "UNSET_VAR_XYZ", Path("/tmp/def")).name
        'cli'
        >>> resolve_path(None, "UNSET_VAR_XYZ", Path("/tmp/def")).name
        'def'
        >>> resolve_path("", "UNSET_VAR_XYZ", Path("/tmp/def")).name
        'def'
    """
    if cli_value:
        return Path(cli_value).expanduser()
    env_value = os.environ.get(env_var, "").strip()
    if env_value:
        return Path(env_value).expanduser()
    return default


def resolve_data_dir(cli_value: str | None = None) -> Path:
    """Résout le dossier ``data/`` (sous-dossiers ``input/`` et ``output/``).

    Args:
        cli_value: Valeur passée en argument CLI, ou None.

    Returns:
        Chemin résolu (peut ne pas encore exister).

    Examples:
        >>> resolve_data_dir().name
        'data'
    """
    return resolve_path(cli_value, "CAISSAI_DATA_DIR", default_state_dir() / "data")


def resolve_log_dir(cli_value: str | None = None) -> Path:
    """Résout le dossier des journaux applicatifs.

    Args:
        cli_value: Valeur passée en argument CLI, ou None.

    Returns:
        Chemin résolu (peut ne pas encore exister).

    Examples:
        >>> resolve_log_dir().name
        'log'
    """
    return resolve_path(cli_value, "CAISSAI_LOG_DIR", default_state_dir() / "log")


def resolve_maia2_dir(cli_value: str | None = None) -> Path:
    """Résout le dossier des poids Maia2 (``blitz_model.pt``).

    ``maia2.model.from_pretrained`` télécharge ses poids dans ``save_root``,
    dont le défaut ``./maia2_models`` est **relatif au répertoire courant**.
    Sans cette résolution explicite, un lancement depuis un autre dossier
    provoque un nouveau téléchargement de plusieurs centaines de Mo.

    Le défaut reproduit le comportement de maia2 (``./maia2_models``) pour
    rester compatible avec les installations existantes.

    Args:
        cli_value: Valeur passée en argument CLI, ou None.

    Returns:
        Chemin résolu (peut ne pas encore exister ; maia2 le crée au besoin).

    Examples:
        >>> resolve_maia2_dir().name
        'maia2_models'
    """
    return resolve_path(cli_value, "CAISSAI_MAIA2_DIR", Path("maia2_models"))


def resolve_pgn_path(cli_value: str | None) -> Path | None:
    """Résout le fichier PGN à traiter (mode ``--pgn``).

    Précédence : ``--pgn`` > ``CAISSAI_PGN`` > None (mode dossier
    input/output classique, voir :func:`resolve_data_dir`).

    Args:
        cli_value: Valeur de ``--pgn``, ou None.

    Returns:
        Chemin résolu, ou None si ni l'argument ni la variable
        d'environnement ne sont renseignés.

    Examples:
        >>> resolve_pgn_path("/tmp/games.pgn").name
        'games.pgn'
        >>> resolve_pgn_path(None) is None
        True
    """
    if cli_value:
        return Path(cli_value).expanduser()
    env_value = os.environ.get("CAISSAI_PGN", "").strip()
    if env_value:
        return Path(env_value).expanduser()
    return None
