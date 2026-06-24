"""
Phase 2 - Step 3 (Week 3, v2): Curriculum DQN training for Connect Four.

Instead of training only against a random opponent, this trains in stages
of increasing difficulty:
  Stage 1: vs random        (learn legal moves, basic winning patterns)
  Stage 2: vs minimax depth 1  (learn to handle weak-but-purposeful play)
  Stage 3: vs minimax depth 3  (learn to handle real strategic resistance)

The same DQN model is carried forward across stages (continued training),
so it builds on what it already learned rather than starting over.

Run from the Connect-Four-Project folder:
    python train_dqn_curriculum.py
"""

import os
import json
import time
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from connect_four_env import ConnectFourEnv

OUTPUT_DIR = "models"
EVAL_EPISODES = 200
RANDOM_SEED = 42

# Curriculum stages: (opponent, opponent_depth, timesteps)
STAGES = [
    ("random", None, 60_000),
    ("minimax", 1, 60_000),
    ("minimax", 3, 40_000),
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


class ProgressCallback(BaseCallback):
    def __init__(self, stage_name, total_steps, print_every=10_000, verbose=0):
        super().__init__(verbose)
        self.stage_name = stage_name
        self.total_steps = total_steps
        self.print_every = print_every
        self._t0 = time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps % self.print_every == 0:
            elapsed = time.time() - self._t0
            print(f"  [{self.stage_name}] {self.num_timesteps}/{self.total_steps} steps, {elapsed:.1f}s elapsed")
        return True


def evaluate(model, opponent, opponent_depth, n_episodes):
    env = ConnectFourEnv(opponent=opponent, opponent_depth=opponent_depth)
    wins, losses, draws, invalids = 0, 0, 0, 0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        info = {}
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

    return {
        "episodes": n_episodes,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / n_episodes,
        "invalid_move_attempts": invalids,
    }


def main():
    print("=== Curriculum DQN Training ===")
    print(f"Stages: {STAGES}\n")

    model = None
    stage_results = []
    total_train_time = 0

    for i, (opponent, depth, timesteps) in enumerate(STAGES, start=1):
        stage_name = f"Stage{i}_{opponent}{'_d'+str(depth) if depth else ''}"
        print(f"--- {stage_name}: training {timesteps} steps vs {opponent}"
              f"{f' (depth {depth})' if depth else ''} ---")

        train_env = Monitor(ConnectFourEnv(opponent=opponent, opponent_depth=depth))

        if model is None:
            # First stage: create the model fresh
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
        else:
            # Subsequent stages: swap in the new (harder) environment,
            # keep the learned weights/replay buffer behavior continuing
            model.set_env(train_env)

        t0 = time.time()
        model.learn(
            total_timesteps=timesteps,
            callback=ProgressCallback(stage_name, timesteps),
            reset_num_timesteps=False,  # keep a continuous step counter across stages
        )
        stage_time = time.time() - t0
        total_train_time += stage_time
        print(f"  {stage_name} finished in {stage_time:.1f}s")

        # Quick eval after each stage vs the stage's own opponent
        eval_result = evaluate(model, opponent, depth, n_episodes=50)
        print(f"  Post-stage win rate vs {opponent}"
              f"{f' depth {depth}' if depth else ''}: {eval_result['win_rate']*100:.1f}%\n")

        stage_results.append({
            "stage": stage_name,
            "opponent": opponent,
            "opponent_depth": depth,
            "timesteps": timesteps,
            "training_time_seconds": stage_time,
            "post_stage_eval": eval_result,
        })

        # Save checkpoint after each stage
        model.save(os.path.join(OUTPUT_DIR, f"dqn_connect_four_{stage_name}"))

    # Final model
    model.save(os.path.join(OUTPUT_DIR, "dqn_connect_four_curriculum_final"))
    print(f"Final curriculum model saved to {OUTPUT_DIR}/dqn_connect_four_curriculum_final.zip")

    # Full evaluation suite on final model
    print("\n=== Final Evaluation (curriculum-trained model) ===")
    final_results = {}

    print("  vs Random opponent...")
    final_results["vs_random"] = evaluate(model, "random", None, EVAL_EPISODES)
    print(f"    Win rate: {final_results['vs_random']['win_rate']*100:.1f}%")

    for depth in [1, 3, 5]:
        print(f"  vs Minimax depth {depth}...")
        n_eps = EVAL_EPISODES if depth <= 3 else 50
        final_results[f"vs_minimax_depth{depth}"] = evaluate(model, "minimax", depth, n_eps)
        print(f"    Win rate: {final_results[f'vs_minimax_depth{depth}']['win_rate']*100:.1f}%")

    summary = {
        "approach": "curriculum",
        "stages": stage_results,
        "total_training_time_seconds": total_train_time,
        "evaluation_episodes_per_opponent": EVAL_EPISODES,
        "final_results": final_results,
    }

    with open(os.path.join(OUTPUT_DIR, "dqn_curriculum_training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Full Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved to {os.path.abspath(OUTPUT_DIR)}/dqn_curriculum_training_summary.json")


if __name__ == "__main__":
    main()
