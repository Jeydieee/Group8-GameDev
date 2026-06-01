# PentagoAI — Group 8

A two-player strategy board game built with Python and Pygame, featuring an AI opponent powered by iterative-deepening alpha-beta search.

---

## Requirements

- **Python 3.12**
- **pip** (comes bundled with Python)

---

## Installing Python

### If Python is already installed

Open a terminal (Command Prompt or PowerShell on Windows, Terminal on macOS/Linux) and verify:

```bash
python --version
```

If the version shown is **3.10 or higher**, skip to [Installing Libraries](#installing-libraries).

---

### If Python is NOT installed

#### Windows

1. Go to [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. Click **Download Python 3.x.x** (the latest stable version).
3. Run the installer.
4. **Important:** On the first screen, check **"Add Python to PATH"** before clicking Install Now.
5. Click **Install Now** and wait for it to finish.
6. Open Command Prompt and confirm: `python --version`

#### macOS

1. Go to [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. Download the latest macOS installer (`.pkg` file).
3. Open the `.pkg` file and follow the on-screen steps.
4. Open Terminal and confirm: `python3 --version`

#### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install python3 python3-pip
```

---

## Downloading the Game

### Option A — With Git installed

```bash
git clone https://github.com/Jeydieee/Group8-GameDev.git
cd Group8-GameDev
```

### Option B — Without Git

1. Go to [https://github.com/Jeydieee/Group8-GameDev](https://github.com/Jeydieee/Group8-GameDev)
2. Click the green **Code** button → **Download ZIP**
3. Extract the ZIP file to a folder of your choice.
4. Open a terminal and navigate to that folder:
   ```bash
   cd path/to/Group8-GameDev
   ```

---

## Installing Libraries

The game requires two libraries: **pygame** and **numpy**.

```bash
pip install pygame numpy
```

> On macOS or Linux, you may need to use `pip3` instead of `pip`.

To verify the installations:

```bash
python -c "import pygame; import numpy; print('All good!')"
```

---

## Running the Game

From inside the `Group8-GameDev` folder, run:

```bash
python Pentago.py
```

> On macOS or Linux: `python3 Pentago.py`

A game window (1060×820) will open and you will land on the **Main Menu**.

---

## How to Play

### Overview

Pentago is a two-player abstract strategy game played on a 6×6 board divided into four 3×3 quadrants. The goal is to get **five of your marbles in a row** — horizontally, vertically, or diagonally — before your opponent does.

You play against an AI opponent.

---

### Game Setup (Main Menu)

Before a match starts, configure three things:

| Setting | Options |
|---|---|
| **Player Name** | Type your name in the text box (up to 15 characters) |
| **Color** | White (moves first) or Black (moves second) |
| **Series Format** | Best of 1, Best of 3, or Best of 5 |

Click **START SERIES** to begin.

---

### Each Turn — Two Steps

Every turn has exactly two mandatory steps:

**Step 1 — Place a marble**
Click any empty cell on the board to place your marble there.

**Step 2 — Rotate a quadrant**
After placing, eight rotation buttons appear below the board. Each button represents one quadrant (Q1–Q4) rotated either clockwise (↻) or counterclockwise (↺). You **must** rotate a quadrant — you cannot skip this step.

> **Quadrant layout:**
> | Q1 (top-left) | Q2 (top-right) |
> |---|---|
> | **Q3 (bottom-left)** | **Q4 (bottom-right)** |

The AI automatically takes its turn after yours, with a thinking animation shown while it calculates.

---

### Winning

A player wins by getting **5 marbles in a row** in any direction. This is checked **after** the rotation step, so a rotation can create — or destroy — a winning line.

- If both players form five-in-a-row simultaneously, the round is a **draw**.
- If the board fills up with no winner, the round is also a **draw**.

---

### Series Play

In a Best of 3 or Best of 5 series, colors swap after each round. The current score is shown at the top of the screen. The series ends when one side reaches the required number of wins.

---

### In-Game Buttons

| Button | Action |
|---|---|
| **Surrender** | Concede the current round; the AI wins it |
| **Main Menu** | Return to setup screen mid-match |
| **Next Match** | Start the next round in a series |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'pygame'`**
Run `pip install pygame` again, making sure you are using the same Python installation you run the game with.

**Window does not open / crashes immediately**
Ensure your system has a display available. Running over SSH without a display is not supported.