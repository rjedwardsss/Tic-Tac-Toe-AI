"""
Tic-Tac-Toe AI — minimax with alpha-beta pruning.

Pure Python, no third-party dependencies. Board is a length-9 list, indices 0–8
(row-major: 0 1 2 / 3 4 5 / 6 7 8). Empty cells are None; marks are "X" or "O".

The AI (maximizer) maximizes a numeric score; the opponent (minimizer) minimizes
it. Terminal scores reward wins for the AI, penalize losses, and treat draws as
neutral, with depth used to prefer quicker wins and slower losses.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple, Union

Mark = Literal["X", "O"]
Cell = Optional[Mark]
Board = List[Cell]
# Game over: one side wins, or full board with no winner.
Outcome = Union[Mark, Literal["D"]]

# All three-in-a-row lines (indices on a 3×3 board).
LINES: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)

# Move expansion order: center and corners first — helps alpha-beta prune earlier.
ORDER: Tuple[int, ...] = (4, 0, 2, 6, 8, 1, 3, 5, 7)


def other(mark: Mark) -> Mark:
    """
    Return the opponent's mark.

    Args:
        mark: Either "X" or "O".

    Returns:
        The other mark.
    """
    return "O" if mark == "X" else "X"


def winner(board: Board) -> Optional[Outcome]:
    """
    Determine whether the game has ended.

    Args:
        board: Current 3×3 board as nine cells.

    Returns:
        The winning Mark if three in a row, "D" if the board is full with no
        winner, or None if the game is still in progress.
    """
    for a, b, c in LINES:
        if board[a] is not None and board[a] == board[b] == board[c]:
            return board[a]
    if all(cell is not None for cell in board):
        return "D"
    return None


def terminal_score(board: Board, ai_mark: Mark, depth: int) -> Optional[int]:
    """
    Numeric score at terminal positions only (game over).

    Uses a small depth adjustment so the AI prefers faster wins and slower losses.
    Non-terminal positions return None so minimax keeps searching.

    Args:
        board: Current board.
        ai_mark: The mark controlled by the AI (the maximizing player).
        depth: Ply count from the start of this search branch.

    Returns:
        Score for the AI if terminal, else None.
    """
    w = winner(board)
    if w is None:
        return None
    if w == "D":
        return 0
    if w == ai_mark:
        return 10 - depth
    return -10 + depth


def minimax(
    board: Board,
    ai_mark: Mark,
    maximizing: bool,
    depth: int,
    alpha: float,
    beta: float,
) -> int:
    """
    Minimax value with alpha-beta pruning for the current position.

    The AI is the maximizing player: when ``maximizing`` is True, we try every
    legal AI move and take the maximum child value. When False, we try opponent
    moves and take the minimum. Alpha (best score the maximizer can guarantee so
    far) and beta (best the minimizer can guarantee) let us skip subtrees that
    cannot affect the root choice.

    The board is temporarily modified during recursion and always restored, so
    the caller's board is unchanged when this returns.

    Args:
        board: Current board (mutated in place, then restored).
        ai_mark: The AI's mark.
        maximizing: True if it is the AI's turn to move at this node.
        depth: Search depth from the root of this minimax call.
        alpha: Lower bound for the maximizing player along this path.
        beta: Upper bound for the minimizing player along this path.

    Returns:
        Minimax score from this node's perspective (higher is better for the AI).
    """
    ts = terminal_score(board, ai_mark, depth)
    if ts is not None:
        return ts

    legal = [i for i in ORDER if board[i] is None]

    if maximizing:
        # AI picks the move that maximizes the score.
        value = float("-inf")
        for i in legal:
            board[i] = ai_mark
            # After AI moves, it is the opponent's turn → minimizing node.
            value = max(
                value,
                minimax(board, ai_mark, False, depth + 1, alpha, beta),
            )
            board[i] = None
            alpha = max(alpha, value)
            # Beta cutoff: minimizer already has a line ≤ beta; they will not pick this branch.
            if alpha >= beta:
                break
        return int(value)

    human_mark = other(ai_mark)
    # Opponent picks the move that minimizes the AI's score.
    value = float("inf")
    for i in legal:
        board[i] = human_mark
        # After opponent moves, it is the AI's turn → maximizing node.
        value = min(
            value,
            minimax(board, ai_mark, True, depth + 1, alpha, beta),
        )
        board[i] = None
        beta = min(beta, value)
        # Alpha cutoff: maximizer already has a line ≥ alpha; this branch is too good for the minimizer to allow.
        if alpha >= beta:
            break
    return int(value)


def best_move_minimax(board: Board, ai_mark: Mark) -> Optional[int]:
    """
    Return the best empty square index (0–8) for the AI, or None if no moves exist.

    If the AI can win in one move, that move is returned immediately (same as
    the original implementation). Otherwise every legal AI move is evaluated with
    minimax starting from the opponent's response (depth 1 in the outer loop).

    Args:
        board: Current board; not permanently altered (temporary placements undone).
        ai_mark: The mark the AI plays.

    Returns:
        Index of the chosen cell, or None if the board is full.
    """
    for i in ORDER:
        if board[i] is not None:
            continue
        board[i] = ai_mark
        if winner(board) == ai_mark:
            board[i] = None
            return i
        board[i] = None

    best: Optional[int] = None
    best_score = float("-inf")
    alpha = float("-inf")
    beta = float("inf")

    for i in ORDER:
        if board[i] is not None:
            continue
        board[i] = ai_mark
        # Opponent to move next → evaluate with minimizing at depth 1.
        score = minimax(board, ai_mark, False, 1, alpha, beta)
        board[i] = None
        if score > best_score:
            best_score = score
            best = i
        alpha = max(alpha, score)

    return best


# ---------------------------------------------------------------------------
# Example usage (run: python tic_tac_toe_ai.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Human (X) played center; AI (O) selects an optimal reply.
    sample_board: Board = [None, None, None, None, "X", None, None, None, None]
    ai = "O"
    move = best_move_minimax(sample_board, ai)
    print("Sample board (None = empty, indices 0-8):")
    print(sample_board)
    print(f"AI mark: {ai}")
    print(f"AI chooses index: {move}")
