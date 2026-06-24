"""
Phase 2 - Step 3 (Week 3): DQN training for Connect Four via stable-baselines3.

Trains a DQN agent against a random opponent first (to learn basic legal-move
play and winning patterns), then evaluates against:
  - random opponent
  - Minimax depth 1 / 3 / 5 opponents

This mirrors the evaluation style used in the Connect Four RL literature
(e.g., win rate vs. fixed-strength opponents), giving real, defensible
numbers for the paper's RL section.

Run from the Connect-Four-Project folder:
    python train_dqn.py
"""

import os
import json
import time
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from connect_four_env import ConnectFourEnv

OUTPUT_DIR = "models"
TOTAL_TIMESTEPS = 100_000          # adjust up/down depending on time budget
EVAL_EPISODES = 200
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


class ProgressCallback(BaseCallback):
    """Lightweight progress logger so long training runs show life signs."""
    def __init__(self, print_every=5000, verbose=0):
        super().__init__(verbose)
        self.print_every = print_every
        self._t0 = time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps % self.print_every == 0:
            elapsed = time.time() - self._t0
            print(f"  [{self.num_timesteps}/{TOTAL_TIMESTEPS}] timesteps, {elapsed:.1f}s elapsed")
        return True


def evaluate(model, opponent, opponent_depth, n_episodes):
    env = ConnectFourEnv(opponent=opponent, opponent_depth=opponent_depth)
    wins, losses, draws, invalids = 0, 0, 0, 0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < 42:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            if info.get("invalid"):
                invalids += 1

        result = info.get("result")
        if result == "win":
            wins += 1
        elif result == "loss":
            losses += 1
        elif result == "draw":
            draws += 1

    n = wins + losses + draws
    return {
        "episodes": n_episodes,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / n_episodes,
        "invalid_move_attempts": invalids,
    }


def main():
    print(f"Training DQN for {TOTAL_TIMESTEPS} timesteps against a random opponent...")

    train_env = Monitor(ConnectFourEnv(opponent="random"))

    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=1e-3,
        buffer_size=50_000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=500,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=[128, 128]),
        verbose=0,
        seed=RANDOM_SEED,
    )

    t0 = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=ProgressCallback(print_every=10_000))
    train_time = time.time() - t0
    print(f"Training finished in {train_time:.1f}s")

    model.save(os.path.join(OUTPUT_DIR, "dqn_connect_four"))
    print(f"Model saved to {OUTPUT_DIR}/dqn_connect_four.zip")

    print("\nEvaluating trained agent...")
    results = {}

    print("  vs Random opponent...")
    results["vs_random"] = evaluate(model, "random", None, EVAL_EPISODES)
    print(f"    Win rate: {results['vs_random']['win_rate']*100:.1f}%")

    for depth in [1, 3, 5]:
        print(f"  vs Minimax depth {depth}...")
        n_eps = EVAL_EPISODES if depth <= 3 else 50  # depth 5 is slow, use fewer episodes
        results[f"vs_minimax_depth{depth}"] = evaluate(model, "minimax", depth, n_eps)
        print(f"    Win rate: {results[f'vs_minimax_depth{depth}']['win_rate']*100:.1f}%")

    summary = {
        "total_timesteps": TOTAL_TIMESTEPS,
        "training_time_seconds": train_time,
        "evaluation_episodes_per_opponent": EVAL_EPISODES,
        "results": results,
    }

    with open(os.path.join(OUTPUT_DIR, "dqn_training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Final Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved to {os.path.abspath(OUTPUT_DIR)}/dqn_training_summary.json")


if __name__ == "__main__":
    main()
