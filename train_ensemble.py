"""
Phase 2 - Week 4: RF + XGBoost Ensemble
Connect Four AI - move prediction ensemble

Loads the trained Random Forest and XGBoost models from Week 1 and combines
their predictions via probability averaging (soft voting). Evaluates the
ensemble's agreement with the true Minimax-chosen move, alongside each
individual model, on the same held-out test set used in Week 1.

This produces a real, measured "ensemble accuracy" number to replace the
fabricated "98.1% Dynamic Ensemble" claim in the original paper draft.

Run from the Connect-Four-Project folder (after train_rf_xgboost.py has
already been run, so models/ contains the saved RF/XGBoost artifacts):
    python train_ensemble.py
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, top_k_accuracy_score, classification_report

# ---------------------------------------------------------------------------
# Config (mirrors train_rf_xgboost.py so the train/test split is reproducible)
# ---------------------------------------------------------------------------
SELF_PLAY_CSV = "datasets/continuous_gameplay_log.csv"
REAL_CSV = "datasets/continuous_gameplay_log_real.csv"
MODELS_DIR = "models"
RANDOM_STATE = 42

ROW_COUNT = 6
COLUMN_COUNT = 7


# ---------------------------------------------------------------------------
# Data loading + feature engineering (identical to train_rf_xgboost.py)
# ---------------------------------------------------------------------------
def load_and_merge(self_play_path, real_path):
    self_play = pd.read_csv(self_play_path)
    self_play["data_source"] = "self_play"
    if real_path and os.path.exists(real_path):
        real = pd.read_csv(real_path)
        real["data_source"] = "real_gameplay"
        merged = pd.concat([self_play, real], ignore_index=True)
    else:
        merged = self_play
    return merged


def filter_ai_decisions(df):
    df = df.copy()
    df["minimax_score"] = pd.to_numeric(df["minimax_score"], errors="coerce")
    return df[df["minimax_score"].notna()].reset_index(drop=True)


def parse_board(board_flat_str):
    return np.array([int(x) for x in board_flat_str.split(",")], dtype=np.int8)


def engineer_features(df, reference_columns=None):
    boards = np.vstack(df["board_flat"].apply(parse_board).values)
    cell_cols = [f"cell_{i}" for i in range(42)]
    feat_df = pd.DataFrame(boards, columns=cell_cols)

    board_grid = boards.reshape(-1, ROW_COUNT, COLUMN_COUNT)
    col_heights = (board_grid != 0).sum(axis=1)
    for c in range(COLUMN_COUNT):
        feat_df[f"col_height_{c}"] = col_heights[:, c]

    feat_df["player_piece_count"] = (boards == 1).sum(axis=1)
    feat_df["ai_piece_count"] = (boards == 2).sum(axis=1)
    feat_df["total_pieces"] = (boards != 0).sum(axis=1)

    feat_df["game_mode"] = df["game_mode"].fillna("UNKNOWN")
    feat_df["difficulty"] = df["difficulty"].fillna("UNKNOWN")
    feat_df["data_source"] = df["data_source"]

    feat_df = pd.get_dummies(feat_df, columns=["game_mode", "difficulty", "data_source"])

    # Align columns to the original training feature set (handles any
    # category that doesn't appear in this particular split)
    if reference_columns is not None:
        feat_df = feat_df.reindex(columns=reference_columns, fill_value=0)

    target = df["chosen_column"].astype(int)
    return feat_df, target


# ---------------------------------------------------------------------------
# Ensemble evaluation
# ---------------------------------------------------------------------------
def evaluate_ensemble(rf_model, xgb_model, X_test, y_test, num_classes, rf_weight=0.5):
    rf_proba = rf_model.predict_proba(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)

    # Align RF's class ordering (RF only learns classes it has seen;
    # XGBoost's predict_proba covers all num_classes by construction)
    rf_full_proba = np.zeros((X_test.shape[0], num_classes))
    for idx, cls in enumerate(rf_model.classes_):
        rf_full_proba[:, cls] = rf_proba[:, idx]

    ensemble_proba = rf_weight * rf_full_proba + (1 - rf_weight) * xgb_proba
    ensemble_pred = np.argmax(ensemble_proba, axis=1)

    acc = accuracy_score(y_test, ensemble_pred)
    top3 = top_k_accuracy_score(y_test, ensemble_proba, k=3, labels=np.arange(num_classes))

    print(f"\n=== RF+XGBoost Ensemble (rf_weight={rf_weight}) ===")
    print(f"Top-1 Accuracy: {acc:.4f}")
    print(f"Top-3 Accuracy: {top3:.4f}")
    print(classification_report(y_test, ensemble_pred, zero_division=0))

    return {"top1_accuracy": acc, "top3_accuracy": top3, "rf_weight": rf_weight}


def evaluate_individual(model, X_test, y_test, num_classes, name, is_xgb=False):
    proba = model.predict_proba(X_test)
    if not is_xgb:
        full_proba = np.zeros((X_test.shape[0], num_classes))
        for idx, cls in enumerate(model.classes_):
            full_proba[:, cls] = proba[:, idx]
        proba = full_proba
    pred = np.argmax(proba, axis=1)
    acc = accuracy_score(y_test, pred)
    top3 = top_k_accuracy_score(y_test, proba, k=3, labels=np.arange(num_classes))
    print(f"{name}: Top-1={acc:.4f}, Top-3={top3:.4f}")
    return {"top1_accuracy": acc, "top3_accuracy": top3}


def main():
    print("Loading saved models from Week 1...")
    rf_model = joblib.load(os.path.join(MODELS_DIR, "random_forest_model.joblib"))
    feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.joblib"))

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(os.path.join(MODELS_DIR, "xgboost_model.json"))

    print("Rebuilding the same train/test split used in Week 1...")
    merged = load_and_merge(SELF_PLAY_CSV, REAL_CSV)
    ai_rows = filter_ai_decisions(merged)
    X, y = engineer_features(ai_rows, reference_columns=feature_columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    num_classes = int(y.max()) + 1
    print(f"Test set: {X_test.shape[0]} rows\n")

    print("--- Individual model performance (sanity check vs Week 1) ---")
    rf_individual = evaluate_individual(rf_model, X_test, y_test, num_classes, "Random Forest", is_xgb=False)
    xgb_individual = evaluate_individual(xgb_model, X_test, y_test, num_classes, "XGBoost", is_xgb=True)

    print("\n--- Ensemble (soft-voting, equal weight) ---")
    ensemble_equal = evaluate_ensemble(rf_model, xgb_model, X_test, y_test, num_classes, rf_weight=0.5)

    # Sweep a couple of weight options for transparency/robustness reporting
    print("\n--- Weight sensitivity sweep ---")
    sweep_results = {}
    for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
        r = evaluate_ensemble(rf_model, xgb_model, X_test, y_test, num_classes, rf_weight=w)
        sweep_results[f"rf_weight_{w}"] = r

    best_weight = max(sweep_results, key=lambda k: sweep_results[k]["top1_accuracy"])
    print(f"\nBest RF weight by top-1 accuracy: {best_weight} -> {sweep_results[best_weight]}")

    summary = {
        "test_set_size": int(X_test.shape[0]),
        "individual_models": {
            "random_forest": rf_individual,
            "xgboost": xgb_individual,
        },
        "ensemble_equal_weight": ensemble_equal,
        "weight_sweep": sweep_results,
        "best_weight_config": best_weight,
    }

    with open(os.path.join(MODELS_DIR, "ensemble_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved ensemble summary to {os.path.abspath(MODELS_DIR)}/ensemble_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
