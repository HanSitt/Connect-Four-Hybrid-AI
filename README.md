<div align="center">
  <img src="https://github.com/user-attachments/assets/8d256564-aad3-4339-9309-5c05e1f1af3a" alt="Connect Wars Gameplay" width="700" />
  
  <h1>Connect Wars: 4 🏰</h1>
  
  <p><b>Advanced Connect-4 AI with Explainable Decision Making (XAI)</b></p>


**Connect Wars** is a high-performance Connect-4 battle simulation powered by a custom-built AI engine. The project focuses on the implementation of the **Minimax Algorithm** optimized with **Alpha-Beta Pruning**, combined with a real-time **Explainable AI (XAI)** system that logs every strategic decision.

---

## 🚀 Key Features

* **Minimax AI Engine:** Calculates the most optimal moves up to 5 steps ahead using deep tree searching.(Unbeatable AI Logic)
* **Different Level Templates:** EASY, MEDIUM and HARD level options for client.(Target depth had determined rowly 1, 3 and 5.)
* **Win & Conquer Mode:** This is a tournament-based game mode.
* **Alpha-Beta Pruning:** A high-efficiency optimization that significantly reduces computational cost by pruning unnecessary branches of the game tree.
* **Explainable AI (XAI):** The AI doesn't just play; it explains. Every move is logged in the terminal with a human-readable strategic explanation (e.g., "AI detects a strong offensive opportunity").
* **Automatic Dataset Generation:** For every session is recorded as a time-stamped one CSV file in the `datasets/` directory, capturing board states and Minimax scores for potential Machine Learning training.
* **Cinematic Experience:** Features a high-quality intro video, dynamic sound effects, and a custom-designed medieval warfare UI.

---

## 🛠️ Technical Stack

* **Language:** Python 3.x
* **GUI & Audio:** [Pygame](https://www.pygame.org/)
* **Matrix Operations:** [NumPy](https://numpy.org/)
* **Computer Vision (Intro):** [OpenCV](https://opencv.org/)
* **Data Structure:** CSV (One dataset for training)

---

## 🎮 How to Play

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/onder17/Connect-Four-AI-Game.git
    ```
2.  **Install Dependencies:**
    ```bash
    pip install pygame numpy opencv-python
    ```
3.  **Run the Game:**
    ```bash
    python connect_4.py
    ```

### Controls:
* **Mouse Move:** Navigate your piece.
* **Left Click:** Drop the piece.
* **F11:** Toggle Fullscreen.
* **ESC/SPACE/ENTER:** Skip intro video.

---

## 🧠 AI Strategy & Logic

The AI evaluates the board using a heuristic scoring system that prioritizes:
1.  **Center Control:** Pieces in the middle column are weighted 3x higher.
2.  **Window Scoring:** Scanning 4-slot "windows" horizontally, vertically, and diagonally.
3.  **Winning/Blocking:** Absolute priority (score: 10^14) given to immediate wins or blocking the player's winning moves.

---

## 📂 Project Structure

* `main.py`: The core game engine and AI logic.
* `images/`: High-resolution UI assets and backgrounds.
* `sounds/`: SFX including rock falling and background music.
* `videos/`: Cinematic intro video.
* `datasets/`: Auto-generated CSV file for every game.

---

## 👥 Contributors

* **Görkem Önder** - *Lead Developer*
* **Group 5 Team Members:** Nyibong George, Han Sitt Aung, Doğa Aslan, Abdulrahman Warsamah.

---
*Developed as a Computer Engineering Senior Design Project.*
