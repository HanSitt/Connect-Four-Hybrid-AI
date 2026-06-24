# Connect-Four-Hybrid-AI

A hybrid AI framework for Connect Four integrating classical search, supervised machine learning, and reinforcement learning — developed as a graduate class project for **AIE503 Introduction to AI Engineering** at Istanbul Okan University, and submitted to **UBMK 2026 (11th International Conference on Computer Science and Engineering)**.

---

## 📄 Paper

**"Hybrid Connect-Four AI: Minimax with Alpha-Beta Pruning, Multi-Model Ensemble, and Explainable Analytics"**  
Han Sitt Aung, Nyibong George, Abdulrahman Warsamah, Sina Alp  
UBMK 2026 — Istanbul Beykent University, September 16–18, 2026

---

## 🧠 System Overview

The system combines three AI paradigms:

| Component | Method | Key Result |
|---|---|---|
| Classical Search | Minimax + Alpha-Beta Pruning (depth 5) | 100% win rate (Hard mode) |
| Supervised Ensemble | Random Forest + XGBoost (soft-voting) | 91.2% top-1 move accuracy |
| Reinforcement Learning | DQN (curriculum training, 3 stages) | 100% win rate vs Minimax depth-3 |

**Notable finding:** Catastrophic forgetting observed across curriculum stages — depth-1 win rate dropped from 100% (after Stage 2) to 0% (after Stage 3), consistent with sequential curriculum learning limitations.

---

## 📁 Repository Structure

```
Connect-Four-Hybrid-AI/
├── connect_4.py                  # Main game engine + Pygame GUI (Görkem Önder)
├── video.py                      # Intro video renderer
├── generate_selfplay_data.py     # Headless self-play data generator
├── train_rf_xgboost.py           # Random Forest + XGBoost training pipeline
├── connect_four_env.py           # Custom Gymnasium environment for DQN
├── train_dqn.py                  # DQN training (vs random opponent)
├── train_dqn_curriculum.py       # DQN curriculum training (3 stages)
├── train_ensemble.py             # RF+XGBoost soft-voting ensemble evaluation
├── datasets/                     # CSV gameplay logs (auto-generated)
├── models/                       # Saved trained models (auto-generated)
├── images/                       # GUI assets
├── sounds/                       # Background music and sound effects
└── videos/                       # Intro and gameplay videos
```

---

## ⚙️ Requirements

```
Python 3.8
pygame==2.6.1
numpy==1.24.4
opencv-python-headless==4.13.0
scikit-learn==1.3.2
xgboost==2.1.4
stable-baselines3==2.4.1
torch==2.4.1
gymnasium==1.0.0
pandas==2.0.3
joblib==1.4.2
```

Install dependencies:
```bash
pip install pygame opencv-python-headless scikit-learn xgboost stable-baselines3 torch gymnasium pandas joblib
```

---

## 🚀 How to Run

### 1. Play the game
```bash
python connect_4.py
```

### 2. Generate self-play training data
```bash
python generate_selfplay_data.py
```
Outputs: `datasets/continuous_gameplay_log.csv` (2,159 AI-decision moves)

### 3. Train Random Forest + XGBoost ensemble
```bash
python train_rf_xgboost.py
```
Results: RF 90.4%, XGBoost 91.0%, Ensemble 91.2% top-1 accuracy

### 4. Train DQN agent (curriculum)
```bash
python train_dqn_curriculum.py
```
Trains across 3 stages: random → Minimax depth-1 → Minimax depth-3  
Total: 160,000 environment steps (~45 min)

### 5. Evaluate ensemble
```bash
python train_ensemble.py
```

---

## 📊 Key Results

### Supervised Move-Prediction (488 test instances, 80/20 split)

| Model | Top-1 Accuracy | Top-3 Accuracy |
|---|---|---|
| Random Forest | 90.4% | 96.1% |
| XGBoost | 91.0% | 97.3% |
| **RF+XGBoost Ensemble** | **91.2%** | **96.7%** |

### DQN Curriculum Agent (post-training evaluation)

| Opponent | Win Rate | Episodes |
|---|---|---|
| Random | 65.5% | 200 |
| Minimax depth-1 | 0.0% | 200 |
| Minimax depth-3 | **100.0%** | 200 |
| Minimax depth-5 | 0.0% | 50 |

---

## 🏗️ Architecture

Four-layer modular design:
1. **Game Engine** — board state management, move validation, win detection
2. **Classical AI Engine** — Minimax + Alpha-Beta pruning + iterative deepening
3. **Machine Learning Layer** — RF, XGBoost, soft-voting ensemble + DQN agent
4. **Analytics & XAI Layer** — real-time CSV logging, decision explanations

---

## 👥 Authors

| Name | Student ID | Level |
|---|---|---|
| Han Sitt Aung | 253307001 | MSc |
| Nyibong George | 253008008 | MSc |
| Abdulrahman Warsamah | 210218356 | BSc |

**Supervisor:** Assist. Prof. Sina Alp  
**Institution:** Istanbul Okan University, Faculty of Engineering and Natural Sciences  
**Course:** AIE503 Introduction to AI Engineering

> Note: Görkem Önder and Doğa Aslan contributed to the original class project codebase (`connect_4.py`) but are not included in the conference submission by their own decision.

---

## 📚 References

1. Madiyarova et al., "Connect-4 AI: A Comprehensive Taxonomy and Critical Review," Preprints.org, 2026
2. Sheoran et al., "Solving Connect 4 Using Optimized Minimax and MCTS," AAMS, 2022
3. Taylor & Stella, "An Evolutionary Framework for Connect-4," arXiv:2405.16595, 2024
4. Nasa et al., "Alpha-Beta Pruning in Mini-Max Algorithm," IRJET, 2018
5. Nayak et al., "Parallelizing MCTS for Connect 4 Using CPU and GPU," Procedia CS, 2025
6. Deptuła, "Application of Game Tree Structures," Silesian Univ. of Technology, 2020
7. Świechowski et al., "Monte Carlo Tree Search: A Review," AI Review, 2023
8. O'Neill, "AI: Connect Four Agent," ACM EngageCSEdu, 2022
9. P.K.G et al., "A Dataset of Gameplay Videos for Connect Four," Data in Brief, 2026
10. Lin et al., "Near-Optimal Algorithms for Minimax Optimization," PMLR, 2020

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.
