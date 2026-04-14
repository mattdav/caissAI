from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, cast

import chess.pgn
import pandas as pd
from beartype import beartype
from chess import Board, Move
from chess.engine import PovScore, SimpleEngine
from chess.pgn import Game, GameNode
from maia2 import inference
from maia2.main import MAIA2Model
from memoization import cached
from openai import OpenAI

from .utils import (
    classify_epd,
    clean_previous_variations,
    get_advantage,
    get_eco_file,
    get_es,
    get_nag_advantage,
)

# ---------------------------------------------------------------------------
# Module-level constants (#13, #19)
# ---------------------------------------------------------------------------

# Maia2 uses an ELO range offset: ratings passed to inference are shifted
# by this constant to account for the model's internal calibration.
MAIA2_ELO_OFFSET: int = 200


@beartype
def classify_opening(game: Game, eco_path: Path) -> GameNode:
    """Takes a game and adds the ECO code an description in its header as well as
    a comment of the last theoretical game.

    Args:
        game (Game): The game to classify.
        eco_path (Path): The path to the Lichess ECO database.

    Returns:
        GameNode: Returns the game positioned at the last theoretical move.
    """
    df_eco = get_eco_file(eco_path)
    opening_node: GameNode = game.root()
    node: GameNode = game.end()
    while not node == game.root():
        assert node.parent is not None
        prev_node = node.parent
        classification = classify_epd(node.board().epd(), df_eco)
        if classification["code"] != "":
            # Add some comments classifying the opening
            game.headers["ECO"] = classification["code"]
            game.headers["Opening"] = classification["desc"]
            node.comment = "{} {}".format(
                classification["code"], classification["desc"]
            )
            # Remember this position so we don't analyze the moves
            # preceding it later
            opening_node = node
            # Break (don't classify previous positions)
            break
        node = prev_node
    return opening_node


@beartype
def extract_game_data(opening_node: GameNode) -> pd.DataFrame:
    """Extract nodes and ratings of player and opponent for each
    move of the game.

    Args:
        opening_node (GameNode): The last theoretical move
        of the game.

    Returns:
        pd.DataFrame: A dataframe with the infos.
    """
    node: GameNode | None = opening_node.parent
    assert node is not None
    w_elo = int(cast(Game, node.root()).headers["WhiteElo"])
    b_elo = int(cast(Game, node.root()).headers["BlackElo"])
    node_list = []
    move_list = []
    player_elo_list = []
    is_forced_list = []
    is_zeroing_list = []
    opponent_elo_list = []
    # Cache board per iteration to avoid replaying from root repeatedly (#8)
    board = node.board()
    while node is not None and not board.is_game_over():
        assert node.parent is not None
        parent_board = node.parent.board()
        node_list.append(node)
        assert node.move is not None
        move_list.append(node.move)
        is_forced_list.append(parent_board.legal_moves.count() == 1)
        is_zeroing_list.append(parent_board.is_zeroing(node.move))
        player_elo_list.append(b_elo if board.turn else w_elo)
        opponent_elo_list.append(w_elo if board.turn else b_elo)
        node = node.next()
        board = node.board() if node is not None else board
    game_data = {
        "node": node_list,
        "played_move": move_list,
        "is_forced": is_forced_list,
        "is_zeroing": is_zeroing_list,
        "player_elo": player_elo_list,
        "opponent_elo": opponent_elo_list,
    }
    df_game = pd.DataFrame.from_dict(game_data)
    return df_game


@beartype
def get_move_probs(df_game: pd.DataFrame, model: MAIA2Model) -> pd.DataFrame:
    """Get likeliest move of each node with its probability.

    Args:
        df_game (pd.DataFrame): The dataframe with the game infos.
        model (MAIA2Model): The Maia2 model (https://www.maiachess.com/).

    Returns:
        pd.DataFrame: The dataframe with added infos.
    """
    prepared = inference.prepare()
    likeliest_move_list = []
    main_proba_list = []
    played_move_prob_list = []
    for node, played_move, player_elo, opponent_elo in df_game[
        ["node", "played_move", "player_elo", "opponent_elo"]
    ].values:
        move_probs, win_prob = inference.inference_each(
            model,
            prepared,
            node.parent.board().fen(),
            player_elo + MAIA2_ELO_OFFSET,
            opponent_elo + MAIA2_ELO_OFFSET,
        )
        likeliest_move_list.append(Move.from_uci(list(move_probs.keys())[0]))
        main_proba_list.append(list(move_probs.values())[0])
        played_move_prob_list.append(move_probs[str(played_move)])
    df_game["likeliest_move"] = likeliest_move_list  # type: ignore[assignment]
    df_game["likeliest_move_proba"] = main_proba_list
    df_game["played_move_proba"] = played_move_prob_list
    return df_game


@beartype
def get_position_depth(
    board: Board,
    elo: int,
    engine: chess.engine.SimpleEngine,
    times: list[float] | None = None,
) -> int:
    """Perform to separate Stockfish analysis, a quick one of time 0.1 and
    a longer one of time 1.
    The more different are the two analysis, the more complex the position is.
    We set the time needed to fully appreciate the position in our analysis
    accordingly.

    Args:
        board (Board): The position to assess.
        elo (int): The player's ELO.
        engine (chess.engine.SimpleEngine): The UCI engine to use.
        times (list, optional): The times to perform the two analysis.
        Defaults to [0.3, 1].

    Returns:
        int: The defined depth to use for further analysis.
    """
    if times is None:
        times = [0.3, 1.0]
    evals = []
    best_moves = []
    depth: list[int] = []
    for time in times:
        info = engine.analyse(board, chess.engine.Limit(time=time))
        eval_score = get_es(elo, info["score"].white().score(mate_score=10000))
        best_move = info["pv"][0]
        evals.append(eval_score)
        best_moves.append(best_move)
        depth.append(info["depth"])
    # Indicateur 1 : Changement d'évaluation entre depths
    delta_evals = abs(evals[1] - evals[0])
    # Indicateur 2 : Changement de meilleur coup
    changement_coup = best_moves[0] != best_moves[1]
    if delta_evals > 0.02 and changement_coup:
        return int(depth[1] * 1.1)
    elif delta_evals > 0.02 or changement_coup:
        return depth[1]
    else:
        return depth[0]


@beartype
def get_eval(
    board: Board,
    engine: chess.engine.SimpleEngine,
    depth: int,
) -> dict[str, Any]:
    """Compute the position's evaluation and the best variation
    according to the engine and the position's complexity.

    Args:
        board (Board): The position to analyse.
        engine (chess.engine.SimpleEngine): The engine to use.
        depth (int): The depth to perform the analysis.

    Returns:
        dict: a dictionary containing the two best variations and their score.
    """
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    eval_result = {"variation": info["pv"], "score": info["score"]}
    return eval_result


@beartype
def evaluate_position(
    board: Board, elo: int, engine: chess.engine.SimpleEngine
) -> dict[str, Any]:
    """Assess a position according to its complexity.

    Args:
        board (Board): The position to analyse.
        elo (int): The player's ELO.
        engine (chess.engine.SimpleEngine): The engine to use.

    Returns:
        dict: A dictionary containing the absolute two best variations and their score.
    """
    depth = get_position_depth(board, elo, engine)
    eval_result = get_eval(board, engine, depth)
    return eval_result


@beartype
def process_row(
    row_data: tuple[GameNode, int, Move, Move], engine_path: Path
) -> dict[str, Any]:
    """Get position and likeliest position evaluations.

    Args:
        row_data (tuple): List with needed infos.
        engine_path (Path): Path to the engine.

    Returns:
        dict[str, Any]: Dictionary with the two evaluations.
    """
    node, player_elo, played_move, likeliest_move = row_data
    engine = SimpleEngine.popen_uci(str(engine_path))
    engine.configure({"Threads": 2, "Hash": 256})
    evals: dict[str, Any] = {}
    evals["played_move_eval"] = evaluate_position(node.board(), player_elo, engine)
    evals["likeliest_move_eval"] = evals["played_move_eval"]
    if played_move != likeliest_move:
        # Build the likeliest board by copying the parent board and pushing the
        # move — avoids mutating the shared game tree from multiple threads (#4)
        assert node.parent is not None
        likeliest_board = node.parent.board()
        likeliest_board.push(likeliest_move)
        evals["likeliest_move_eval"] = evaluate_position(
            likeliest_board, player_elo, engine
        )
    engine.quit()
    return evals


@beartype
def evaluate_game(
    df_game: pd.DataFrame, engine_path: Path, workers: int
) -> pd.DataFrame:
    """Process each node of the game to get its evaluations.

    Args:
        df_game (pd.DataFrame): Dataframe of the game.
        engine_path (Path): Path to the engine.
        workers (int): Number of CPU workers to use in parallel.

    Returns:
        pd.DataFrame: The dataframe with evaluations added.
    """
    # itertuples is significantly faster than iterrows for row iteration (#6)
    row_data = [
        (row.node, row.player_elo, row.played_move, row.likeliest_move)
        for row in df_game[
            ["node", "player_elo", "played_move", "likeliest_move"]
        ].itertuples(index=False)
    ]
    process_func = partial(process_row, engine_path=engine_path)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        evals: list[dict[str, Any]] = list(executor.map(process_func, row_data))
    df_evals = pd.json_normalize(evals)
    df_game = pd.concat([df_game, df_evals], axis=1)
    return df_game


@cached
@beartype
def get_nags(
    is_forced: bool,
    is_zeroing: bool,
    two_moves_before_score: PovScore,
    before_move_score: PovScore,
    played_move_proba: float,
    after_move_score: PovScore,
    after_likeliest_move_score: PovScore,
    next_likeliest_move_proba: float,
    next_likeliest_move_score: PovScore | None,
    player_elo: int,
    opponent_elo: int,
) -> dict[str, Any]:
    """Compare the player's move performance with the best one and the likeliest one.
    We use the two following references to classify a move.
    First the expected score delta between two moves as defined by chess.com :
    https://support.chess.com/en/articles/8572705-how-are-moves-classified-what-is-a-blunder-or-brilliant-etc
    Then, we add a criteria of advantage magnitude delta as defined there :
    https://chessdream.app/blogs/advanced-stockfish-analysis
    A dubious move reduces the player's advantage by a magnitude of at least 1 and
    its expected score by at least 0.05.
    A mistake reduces the player's advantage by a magnitude of at least 2 and
    its expected score by at least 0.1.
    A blunder reduces the player's advantage by a magnitude of 3 or more and
    its expected score by at least 0.2.
    A move is defined as brillant if :

        - the player's expected score compared with the likeliest's move expected score
          is greater than 0.2.
        - the player's avantage compared with the likeliest's move advantage
          is better by at least 3 magnitude.
        - the resulting position is good (expected score > 0.5).
        - the move had a probability of being played of less than 0.2.
        - A capture or a move pawn (zeroing).

    A move is defined as good if :

        - the player's expected score improved by at least 0.1.
        - the player's avantage magnitude improved by at least 2.

    A move is defined as interesting if :

        - the resulting position is good (expected score > 0.5).
        - the resulting position's likeliest move for the opponent is a mistake
          or a blunder and has a probability of being played higher than 0.33.

    Args:
        is_forced (bool): True if the move is forced.
        is_zeroing (bool): True if the move is a capture or a move pawn.
        two_moves_before_score (PovScore): The position's evaluation last time
        the current player played.
        before_move_score (PovScore): The position's evaluation before the
        current player moved.
        played_move_proba (float): The probability of the player's move
        after_move_score (PovScore): The position's evaluation after the
        current player moved.
        after_likeliest_move_score (PovScore): The position's evaluation if the
        current player had played the likeliest move.
        next_likeliest_move_proba (float): The probability the current player's
        opponent has to play the likeliest next move.
        next_likeliest_move_score (PovScore): The position's evaluation if the
        next player plays the likeliest move.
        player_elo (int): The player's ELO.
        opponent_elo (int): His opponent ELO.

    Returns:
        dict: A dictionary with all needed informations.
    """
    color_player = bool(before_move_score.turn)  # True = white (#11)
    adv_after_move = get_advantage(after_move_score.white().score(mate_score=100000))
    nag_after_move = get_nag_advantage(adv_after_move)
    es_after_move = get_es(
        player_elo, after_move_score.white().score(mate_score=100000)
    )
    adv_after_likeliest_move = get_advantage(
        after_likeliest_move_score.white().score(mate_score=100000)
    )
    es_after_likeliest_move = get_es(
        player_elo, after_likeliest_move_score.white().score(mate_score=100000)
    )
    adv_before_move = get_advantage(before_move_score.white().score(mate_score=100000))
    es_before_move = get_es(
        player_elo, before_move_score.white().score(mate_score=100000)
    )
    adv_two_moves_before = get_advantage(
        two_moves_before_score.white().score(mate_score=100000)
    )
    es_two_moves_before = get_es(
        player_elo, two_moves_before_score.white().score(mate_score=100000)
    )
    if next_likeliest_move_score is not None:  # explicit None check (#12)
        adv_next_likeliest_move = get_advantage(
            next_likeliest_move_score.white().score(mate_score=100000)
        )
        es_next_likeliest_move = get_es(
            opponent_elo, after_likeliest_move_score.white().score(mate_score=100000)
        )
    else:
        adv_next_likeliest_move = adv_after_move
        es_next_likeliest_move = es_after_move

    if not color_player:
        es_after_move = 1 - es_after_move
        es_after_likeliest_move = 1 - es_after_likeliest_move
        es_next_likeliest_move = 1 - es_next_likeliest_move
        es_before_move = 1 - es_before_move
        es_two_moves_before = 1 - es_two_moves_before
    es_after_before_delta = es_after_move - es_before_move
    es_after_two_moves_before_delta = es_after_move - es_two_moves_before
    es_played_likeliest_delta = es_after_move - es_after_likeliest_move
    es_next_likeliest_before_delta = es_after_likeliest_move - es_before_move
    adv_after_before_delta = abs(adv_after_move - adv_before_move)
    adv_after_two_moves_before_delta = abs(adv_after_move - adv_two_moves_before)
    adv_played_likeliest_delta = abs(adv_after_move - adv_after_likeliest_move)
    adv_next_likeliest_before_delta = abs(adv_next_likeliest_move - adv_before_move)
    nag_dict: dict[str, Any] = {
        "nag": [],
        "es_after_two_moves_before_delta": es_after_two_moves_before_delta,
        "es_after_before_delta": es_after_before_delta,
        "es_after_move": es_after_move,
        "es_played_likeliest_delta": es_played_likeliest_delta,
        "es_next_likeliest_before_delta": es_next_likeliest_before_delta,
    }
    if is_forced:
        nag_dict["nag"] = [chess.pgn.NAG_SINGULAR_MOVE]
    else:
        if es_after_before_delta <= -0.2 and adv_after_before_delta >= 3:
            nag_dict["nag"] = [chess.pgn.NAG_BLUNDER]
        elif es_after_before_delta <= -0.2 or (
            es_after_before_delta <= -0.1 and adv_after_before_delta >= 2
        ):
            nag_dict["nag"] = [chess.pgn.NAG_MISTAKE]
        elif es_after_before_delta <= -0.05 and adv_after_before_delta >= 1:
            nag_dict["nag"] = [chess.pgn.NAG_DUBIOUS_MOVE]
        elif (
            es_played_likeliest_delta >= 0.2
            and adv_played_likeliest_delta >= 3
            and es_after_move > 0.5
            and played_move_proba <= 0.2
            and is_zeroing
        ):
            nag_dict["nag"] = [chess.pgn.NAG_BRILLIANT_MOVE]
        elif (
            adv_after_two_moves_before_delta >= 2
            and es_after_two_moves_before_delta >= 0.1
        ):
            nag_dict["nag"] = [chess.pgn.NAG_GOOD_MOVE]
        elif (
            es_after_move > 0.5
            and es_next_likeliest_before_delta <= 0.1
            and adv_next_likeliest_before_delta >= 2
            and next_likeliest_move_proba >= 0.33
        ):
            nag_dict["nag"] = [chess.pgn.NAG_SPECULATIVE_MOVE]
        if len(nag_dict["nag"]) > 0:
            nag_dict["nag"].append(nag_after_move)
    return nag_dict


@cached
@beartype
def get_comment(
    node: GameNode,
    player_elo: int,
    played_move_proba: float,
    played_move_score: PovScore,
    likeliest_move: Move,
    next_likeliest_move: Move | None,
    next_likeliest_move_proba: float,
    nag_dict: dict[str, Any],
) -> str:
    """Generate a comment for an annotated move according to its specificities.

    Args:
        node (GameNode): The node to comment.
        player_elo (int): The player's ELO.
        played_move_proba (float): The probability the player had to play
        his move.
        played_move_score (PovScore): The score position after the move.
        likeliest_move (str): The likeliest move the player had to play.
        next_likeliest_move (str): The likeliest move his opponent had to play.
        next_likeliest_move_proba (float): The probability his opponent had to
        play the next move.
        nag_dict (dict): The dictionary containing the NAG infos.

    Returns:
        str: The resulting comment.
    """
    comment = ""
    player_color = "noirs" if played_move_score.turn else "blancs"
    next_player_color = "blancs" if played_move_score.turn else "noirs"
    assert node.parent is not None
    if chess.pgn.NAG_BRILLIANT_MOVE in nag_dict["nag"]:
        comment = (
            f"Un coup brillant qui n'avait que {100 * played_move_proba:.2f}% "
            "de probabilité d'être joué et qui débouche sur une position avec une "
            "espérance de gain de +"
            f"{100 * nag_dict['es_played_likeliest_delta']:.2f}%"
            f" pour les {player_color} "
            f"par rapport au coup {node.parent.board().san(likeliest_move)} qui était "
            f"le plus probable dans cette position pour un joueur de {player_elo} ELO."
        )
    elif chess.pgn.NAG_GOOD_MOVE in nag_dict["nag"]:
        comment = (
            f"Un très bon coup qui améliore l'espérance de gain des {player_color} "
            f"de +{100 * nag_dict['es_after_two_moves_before_delta']:.2f}%."
        )
    elif chess.pgn.NAG_SPECULATIVE_MOVE in nag_dict["nag"] and next_likeliest_move is not None:
        comment = (
            "Un coup intéressant. Dans la position résultante, "
            f"les {next_player_color} ont une probabilité de "
            f"{100 * next_likeliest_move_proba:.2f}% de jouer le mauvais coup "
            f"{node.board().san(next_likeliest_move)} avec une espérance de gain "
            f"de {100 * nag_dict['es_next_likeliest_before_delta']:.2f}%."
        )
    elif any(  # dubious, mistake and blunder share the same comment format (#10)
        nag in nag_dict["nag"]
        for nag in (
            chess.pgn.NAG_DUBIOUS_MOVE,
            chess.pgn.NAG_MISTAKE,
            chess.pgn.NAG_BLUNDER,
        )
    ):
        comment = (
            f"Espérance de gain pour les {player_color} : "
            f"{100 * nag_dict['es_after_before_delta']:.2f}%."
        )
    return comment


@beartype
def comment_end(end_node: GameNode, last_eval: float, last_player_color: str) -> str:
    """Analyse the game's end nature and provide an adapted comment.

    Args:
        end_node (chess.pgn.ChildNode): The last node of the game.
        last_eval (float): Last expected score of the game.
        last_player_color (str): Color of the last player of the game.

    Returns:
        str: The comment of the end node.
    """
    if end_node.board().outcome():
        if end_node.board().is_stalemate():
            return "Stalemate"
        elif end_node.board().is_insufficient_material():
            return " Insufficient material to mate"
        elif end_node.board().can_claim_fifty_moves():
            return " Fifty move rule"
        elif end_node.board().can_claim_threefold_repetition():
            return " Three-fold repetition"
        elif end_node.board().is_checkmate():
            return " Checkmate"
        return ""
    else:
        end_comment = (
            f"Espérance de gain pour les {last_player_color} : {100 * last_eval:.2f}%"
        )
        return end_comment


@beartype
def df_to_pgn(df: pd.DataFrame) -> Game:
    """Convert a dataframe to a PGN game.

    Args:
        df (pd.DataFrame): The dataframe with all the infos.

    Returns:
        Game: A PGN formatted game.
    """
    node = df.iloc[-1:]["node"].values[0]
    last_eval = df.iloc[-1:]["nag_dict"].values[0]["es_after_move"]
    last_player_color = (
        "noirs" if df.iloc[-1:]["played_move_eval.score"].values[0].turn else "blancs"
    )
    node = clean_previous_variations(node)
    for _index, row in df[::-1].iterrows():
        node.nags = row["nag_dict"]["nag"]
        node.comment = row["comment"]
        if (
            len(row["nag_dict"]["nag"]) > 0
            and row["nag_dict"]["nag"][0]
            in [
                chess.pgn.NAG_DUBIOUS_MOVE,
                chess.pgn.NAG_MISTAKE,
                chess.pgn.NAG_BLUNDER,
            ]
            and row["best_move_variation"][0] != row["played_move"]
        ):
            assert node.parent is not None
            node.parent.add_line(
                moves=row["best_move_variation"], comment="Meilleure variante."
            )
        assert node.parent is not None
        node = node.parent
    node.end().comment = comment_end(node.end(), last_eval, last_player_color)
    return cast(Game, node.root())


@beartype
def summarise_game(game: Game, client: OpenAI, openai_model: str) -> Game:
    """Prompt GPT for a synthesis of the entire game
    to insert as final comment.

    Args:
        game (Game): The game to summarized.
        client (OpenAI): OpenAI API's client.
        openai_model (str): OpenAI model name to use.

    Returns:
        chess.pgn.Game: The game with the synthesis as final comment.
    """
    response = client.responses.create(
        model=openai_model,
        instructions="Tu es un coach d'échecs de niveau 3200 ELO en charge de "
        f"produire une analyse éclairante de la partie {game}.",
        input="Répond en trois phrases. Une pour mettre en lumière les moments "
        "clés de la partie. Une pour résumer les réussites et les erreurs des "
        "blancs ainsi que ce qu'ils doivent améliorer. Enfin, une dernière pour "
        "résumer les réussites et les erreurs de noirs ainsi que ce qu'ils "
        "doivent améliorer.",
    )
    game.end().comment += f" - {response.output_text}"
    return game


@beartype
def _build_analysis_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Align each row with the context it needs for NAG/comment computation:
    previous evaluations (shift forward) and opponent's next move (shift back).

    Args:
        df (pd.DataFrame): Fully evaluated game DataFrame.

    Returns:
        pd.DataFrame: Copy of df starting from move index 2, with context
        columns added.
    """
    df = df.copy()
    df["best_move_variation"] = df.shift(1)["played_move_eval.variation"]
    df["before_move_score"] = df.shift(1)["played_move_eval.score"]
    df["two_moves_before_score"] = df.shift(2)["played_move_eval.score"]
    df["next_likeliest_move"] = df.shift(-1)["likeliest_move"]
    df["next_likeliest_move_score"] = df.shift(-1)["likeliest_move_eval.score"]
    df["next_likeliest_move_proba"] = df.shift(-1)["likeliest_move_proba"]
    return df.iloc[2:].copy()  # first two rows lack the required context (#5)


@beartype
def process_game(
    game: Game,
    config_path: Path,
    engine_path: Path,
    maia2_model: MAIA2Model,
    n_workers: int,
    openai_client: OpenAI,
    comment: bool,
    openai_model: str = "gpt-4.1",
) -> Game:
    """Iterate through a game from the last theoretical move to the end.
    For each move, detect if it is a critical one.
    If so, we annonate it with the appropriates NAGs and we provide
    a comment.
    Finally, we provide a comment to explain why the game ended and
    we ask GPT for a game analysis.

    Args:
        game (Game): The game to comment.
        config_path (Path): Path to the config folder.
        engine_path (Path): Path to the engine.
        maia2_model (MAIA2Model): The Maia2 model (https://www.maiachess.com/).
        n_workers (int): Number of CPU workers to use.
        openai_client (OpenAI): The OpenAI API client to use.
        comment (bool): Whether to ask GPT for a comment of the game or not.
        openai_model (str): OpenAI model name to use for game summarisation.
            Defaults to "gpt-4.1". Read from OPENAI_MODEL env var in __main__.py.

    Returns:
        Game: The annotated game.
    """
    opening_node = classify_opening(game, config_path / "lichess_eco.parquet")
    df_game = extract_game_data(opening_node)
    df_game = get_move_probs(df_game, maia2_model)
    df_game = evaluate_game(df_game, engine_path=engine_path, workers=n_workers)
    # Build context columns and drop the first two rows that lack prior context
    df_game_to_comment = _build_analysis_columns(df_game)
    # Use named kwargs to make the column→parameter mapping explicit (#9)
    df_game_to_comment["nag_dict"] = df_game_to_comment.apply(
        lambda row: get_nags(
            is_forced=row["is_forced"],
            is_zeroing=row["is_zeroing"],
            two_moves_before_score=row["two_moves_before_score"],
            before_move_score=row["before_move_score"],
            played_move_proba=row["played_move_proba"],
            after_move_score=row["played_move_eval.score"],
            after_likeliest_move_score=row["likeliest_move_eval.score"],
            next_likeliest_move_proba=row["next_likeliest_move_proba"],
            next_likeliest_move_score=row["next_likeliest_move_score"],
            player_elo=row["player_elo"],
            opponent_elo=row["opponent_elo"],
        ),
        axis=1,
    )
    df_game_to_comment["comment"] = df_game_to_comment.apply(
        lambda row: get_comment(
            node=row["node"],
            player_elo=row["player_elo"],
            played_move_proba=row["played_move_proba"],
            played_move_score=row["played_move_eval.score"],
            likeliest_move=row["likeliest_move"],
            next_likeliest_move=row["next_likeliest_move"],
            next_likeliest_move_proba=row["next_likeliest_move_proba"],
            nag_dict=row["nag_dict"],
        ),
        axis=1,
    )
    commented_game = df_to_pgn(df_game_to_comment)
    if comment:
        commented_game = summarise_game(commented_game, openai_client, openai_model)
    return commented_game
