import chess
import chess.engine
import numpy as np
import os
import time

from utils import material_balance, is_endgame
from time_manager import TimeManager


class LinUCB:

    def __init__(self, n_arms, n_features, alpha=1.5):

        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha

        self.A_inv = [
            np.identity(n_features)
            for _ in range(n_arms)
        ]

        self.b = [
            np.zeros((n_features, 1))
            for _ in range(n_arms)
        ]

    def select_arm(self, x):

        p_values = []

        for arm in range(self.n_arms):

            theta = self.A_inv[arm] @ self.b[arm]

            p = (
                theta.T @ x
                + self.alpha
                * np.sqrt(x.T @ self.A_inv[arm] @ x)
            )

            p_values.append(p.item())

        return int(np.argmax(p_values))

    def update(self, arm, x, reward):

        num = (self.A_inv[arm] @ x) @ (x.T @ self.A_inv[arm])
        den = 1.0 + (x.T @ self.A_inv[arm] @ x).item()
        
        self.A_inv[arm] -= num / den

        self.b[arm] += reward * x


class ChessMAB:

    def __init__(
        self,
        engine,
        model_path="models/final_model.npz"
    ):

        self.engine = engine

        self.model_path = model_path

        self.n_features = 7

        self.n_arms = 4

        self.bandit = LinUCB(
            self.n_arms,
            self.n_features
        )

        self.time_manager = TimeManager()

        self.load()

    # ---------------------------------
    # Save / Load
    # ---------------------------------

    def save(self):

        temp_path = self.model_path + ".tmp.npz"

        np.savez(temp_path, A_inv=self.bandit.A_inv, b=self.bandit.b)

        os.replace(temp_path, self.model_path)

    def load(self):

        if not os.path.exists(self.model_path):

            print("New model initialized.")

            return

        try:

            data = np.load(self.model_path)

            self.bandit.A_inv = list(data['A_inv'])
            self.bandit.b = list(data['b'])

            print(f"Loaded model: {self.model_path}")

        except Exception:

            print("Corrupted model file. Reinitializing.")

    # ---------------------------------
    # Evaluation
    # ---------------------------------

    def evaluate(self, board, depth=8):

        info = self.engine.analyse(
            board,
            chess.engine.Limit(depth=depth)
        )

        score = info["score"].white()

        if score.is_mate():

            mate_moves = score.mate()
            eval_cp = 10000 if mate_moves > 0 else -10000

        else:
            
            eval_cp = score.score()
            
        wdl = 1.0 / (1.0 + 10.0 ** (-eval_cp / 400.0))

        return wdl

    # ---------------------------------
    # Features
    # ---------------------------------

    def extract_features(self, board, clock):

        legal_moves = len(list(board.legal_moves))
        captures = len([m for m in board.legal_moves if board.is_capture(m)])

        features = np.array([

            legal_moves / 50.0,

            clock.ratio(),

            board.fullmove_number / 100.0,

            material_balance(board) / 39.0,

            int(is_endgame(board)),

            int(board.is_check()),
            
            captures / 10.0

        ])

        return features.reshape(-1, 1)

    # ---------------------------------
    # Reward
    # ---------------------------------

    def compute_reward(
        self,
        board,
        move,
        elapsed
    ):

        score_before = self.evaluate(board, depth=8)

        board.push(move)

        score_after = self.evaluate(board, depth=8)

        board.pop()

        delta_wdl = score_after - score_before

        quality = delta_wdl * 10.0

        reward = (
            2.0 * quality
            - 0.05 * elapsed
        )

        return reward

    # ---------------------------------
    # Play
    # ---------------------------------

    def play(
        self,
        board,
        clock,
        training=True
    ):

        x = self.extract_features(
            board,
            clock
        )

        arm = self.bandit.select_arm(x)

        legal_moves = len(list(board.legal_moves))

        budget = self.time_manager.compute_time_budget(
            arm=arm,
            remaining_time=clock.time_left,
            legal_moves=legal_moves,
            move_number=board.fullmove_number,
            endgame=is_endgame(board)
        )

        start = time.time()

        result = self.engine.play(
            board,
            chess.engine.Limit(time=budget)
        )

        elapsed = time.time() - start

        move = result.move

        reward = 0.0

        # ---------------------------------
        # Training only
        # ---------------------------------

        if training:

            reward = self.compute_reward(
                board,
                move,
                elapsed
            )
            
            # Penalize agent heavily if it flags on time
            if clock.time_left - elapsed <= 0.0:
                reward -= 10.0

            self.bandit.update(
                arm,
                x,
                reward
            )

        clock.spend(elapsed)

        return (
            move,
            arm,
            reward,
            elapsed,
            budget
        )
