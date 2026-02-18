# NFL Red Zone Efficiency Analysis (2024)

**Author:** Suhas Makineni  
**Project:** Project 2 — NFL Red Zone Efficiency Analysis

## Overview
This project analyzes the **2024 NFL play-by-play dataset** to measure **red zone efficiency**—how often plays inside the red zone result in touchdowns—and to understand which situational factors are most associated with scoring.

The project has two parts:
1. **Statistical analysis** of touchdown probability in the red zone by field position, down, and play type.
2. **Predictive modeling** to estimate touchdown likelihood using logistic regression and a random forest classifier.

## Data
- **Source file:** `pbp-2024.csv`
- Each row represents an offensive play with context such as:
  - down and distance
  - field position
  - play result (yards gained, touchdown indicator)
  - play type (rush/pass)
  - penalty information

## Tools & Libraries
- Python: `pandas`, `numpy`
- Visualization: `matplotlib`, `seaborn`
- Statistics: `scipy.stats`
- Modeling: `scikit-learn`

## Methodology

### 1) Data Cleaning
Steps performed:
- Dropped fully-null columns
- Filtered to **offensive plays only**: `PlayType ∈ {RUSH, PASS}`
- Kept only relevant variables for analysis:
  - game context: quarter/time, offense/defense
  - situation: down, yards-to-go
  - field position: yard line variables
  - outcomes: yards gained, touchdown
  - play type + penalty fields

### 2) Feature Engineering
Created the following derived variables:
- `yardline_100`: standardized field position variable (see note below)
- `red_zone`: binary flag for red zone plays
- `td`: touchdown indicator as an integer (0/1)

> **Important note on field position**
>
> Red zone can be defined depending on how `yardline_100` is coded:
> - If `yardline_100` means **yards from opponent end zone**, red zone is `yardline_100 <= 20`
> - If `yardline_100` means **field position where 100 is the opponent end zone**, red zone is `yardline_100 >= 80`
>
> This repo assumes the definition used in the analysis notebook/code you run. Verify this once by checking a few sample rows.

### 3) Exploratory Data Analysis (EDA)
Key variables explored:
- `Down`
- `ToGo`
- `yardline_100`
- `Yards`
- `td`

Visualizations include:
- Histograms for down, distance, field position, and yards gained
- Touchdown probability by field position bins (showing sharp increase in/near red zone)

### 4) Hypothesis Testing (Red Zone Only)
Three hypotheses were evaluated using **chi-square tests** on contingency tables:

#### Hypothesis 1 — Field position affects TD probability
- **H0:** TD rates are equal across red zone field-position bins  
- **H1:** TD rates differ across bins  
- **Result:** Reject H0 (p < 0.05). TD probability increases near the goal line (0–5 yards).

#### Hypothesis 2 — Down affects TD probability
- **H0:** TD probability is equal across downs  
- **H1:** Some downs have different TD probabilities  
- **Result:** Reject H0 (p < 0.05). 4th down shows higher TD probability, likely due to selection effects (teams go for it near the goal line).

#### Hypothesis 3 — Play type affects TD probability
- **H0:** Rush and pass plays have equal TD probability  
- **H1:** TD probability differs by play type  
- **Result:** Fail to reject H0 (p > 0.05). Rush vs pass efficiency is similar in the red zone at the aggregate level.

### 5) Predictive Modeling (TD Classification)
Goal: Predict whether a red zone play results in a touchdown.

**Features:**
- `Down`, `ToGo`, `yardline_100`, `IsRush`, `IsPass`, `IsPenalty`

**Models:**
- Logistic Regression  
- Random Forest (300 trees)

**Evaluation metrics:**
- Accuracy (reported but not emphasized due to class imbalance)
- ROC AUC (primary metric)

**Results (from analysis run):**
- Logistic Regression: **AUC ≈ 0.746**
- Random Forest: **AUC ≈ 0.727**

Accuracy was very high for both models but is **misleading** because touchdowns are relatively rare (even in the red zone). AUC provides a more meaningful measure of discrimination between TD vs non-TD plays.

Feature importance from the Random Forest indicated:
- `yardline_100` is the strongest predictor
- `Down` and `ToGo` contribute moderately
- play type and penalties contribute minimally

## Key Findings
- **Field position is the strongest driver** of red zone touchdown probability.
- **Down appears significant**, but much of the effect is due to situational context (4th downs happen closer to the goal line).
- **Rush vs pass** does not significantly change touchdown probability in the red zone at an aggregate level.
- A **simple logistic regression** performs competitively with a random forest for the selected features.

## How to Run

### 1) Create environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
