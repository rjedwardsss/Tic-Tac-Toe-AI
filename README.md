# Tic-Tac-Toe AI (Minimax)

## Overview

This project is a small, dependency-free Python module that plays perfect Tic Tac Toe using the **minimax** algorithm. Given any legal board position and which side the AI plays (X or O), it computes an **optimal** move: with correct play from both sides, the AI never loses (wins if possible, otherwise forces a draw).

Minimax is a classic decision rule for two-player, zero-sum games: it searches ahead, assuming both players choose moves that are best for themselves, and picks the move that leads to the best guaranteed outcome for the AI.

## How It Works

- **Minimax** — The AI is the **maximizing** player: it tries to maximize a numeric score. The opponent is the **minimizing** player: they try to minimize that same score. The algorithm builds a game tree (or explores it implicitly), evaluates terminal positions (win, loss, draw), and propagates values up so each side picks the best reply at every turn.
- **Alpha-beta pruning** — While searching, **alpha** is the best score the maximizer can already force; **beta** is the best the minimizer can force. If a branch cannot improve the result, it is **pruned** (skipped). The chosen move stays the same as plain minimax, but fewer positions are evaluated.
- **Scoring** — Wins for the AI score positively (with a small bonus for **shorter** wins), losses score negatively (with a preference for **longer** losses), and draws score zero. Non-terminal positions are not scored directly; search continues until the game ends along that line of play.

## Features

- **Optimal play** — The AI never loses against perfect opposition.
- **Alpha-beta pruning** — Faster search without changing the optimal move.
- **Clean board representation** — A simple length-9 list: indices `0–8` in row-major order, `None` for empty cells, `"X"` / `"O"` for marks.

## Usage

From the directory containing `tic_tac_toe_ai.py`:

```bash
python tic_tac_toe_ai.py
```

To use the AI in your own code, import `best_move_minimax` (and optionally `winner`, `Board`) and pass a board list plus the AI’s mark (`"X"` or `"O"`).

## Example

The built-in demo uses a board where **X** has taken the center (index `4`) and the AI plays **O**. The script prints the board, the AI’s mark, and the chosen index (e.g. a corner), which is one of several optimal replies in that position.

## Why I Built This

I wanted a compact reference for **adversarial search** and **optimal decision-making**—the same ideas that underpin stronger game AI and many planning problems. Minimax with alpha-beta is a clear bridge from classic CS to modern systems that reason about opponents and outcomes.
