# 🏈 NFL Red Zone Efficiency Analysis (2024)
Author: Suhas Makineni

What makes a red zone play turn into a touchdown? Using every offensive play from the **2024 NFL season**, this project tests which situational factors (field position, down, play type) drive touchdown probability inside the 20, then builds models to predict it.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)

---

## 🔑 Key Findings

| | |
|---|---|
| 📍 **Field position dominates** | TD rate climbs from **7%** at 15–20 yards out to **33%** inside the 5 |
| 4️⃣ **4th down scores most (29%)**, but it's selection bias | Teams mostly go for it on 4th down when they're right at the goal line |
| 🎯 **Passing beats rushing** | Pass plays score **21.3%** of the time vs **17.1%** for rushes (p = 0.0001) |
| 🌲 **Random forest edges out logistic regression** | ROC-AUC **0.73** vs **0.69** |

---

## 📦 Data

- **Source:** `pbp-2024.csv`, the 2024 NFL play-by-play data (**53,283 plays**, 45 columns)
- **Scope:** offensive plays only (`RUSH`, `PASS`), giving **5,546 red zone plays**
- Each row includes down, distance, field position, yards gained, play type, penalty info, and a touchdown flag

---

## 🔬 Methodology

### 1. Cleaning & feature engineering
- Dropped fully null columns and kept only rush and pass plays
- `yardline_100`: field position, where **100 = the opponent's goal line**
- `red_zone`: `1` if `yardline_100 >= 80` (inside the opponent's 20)
- `td`: touchdown indicator (0/1)

### 2. Exploratory analysis
Distributions of down, yards to go, field position, and yards gained, plus touchdown probability by field position bin. The bins show TD rate jumping sharply once a team reaches the red zone.

### 3. Hypothesis testing: chi-square tests on red zone plays

| Hypothesis | Result | Takeaway |
|---|:---:|---|
| **H1:** TD rate differs by field position | ✅ Reject H₀ (p < 0.001) | The closer to the goal line, the higher the TD rate |
| **H2:** TD rate differs by down | ✅ Reject H₀ (p < 0.001) | Driven by situation: 4th down attempts happen closest to the goal line |
| **H3:** TD rate differs by play type | ✅ Reject H₀ (p = 0.0001) | Passes convert more often than rushes in the red zone |

### 4. Predictive modeling
**Goal:** predict whether a red zone play results in a touchdown.
**Features:** `Down`, `ToGo`, `yardline_100`, `IsRush`, `IsPass`, `IsPenalty` · **Split:** 80/20 train/test

| Model | ROC-AUC |
|---|:---:|
| Logistic Regression | 0.69 |
| **Random Forest** (300 trees) | **0.73** |

**Why AUC and not accuracy?** Only about 19% of red zone plays score, so a model that always predicts "no touchdown" is already 81% accurate. ROC-AUC measures how well each model *ranks* TD plays above non-TD plays, which is what matters here.

**Feature importance (random forest):** field position (55%) > yards to go (30%) > down (9%) > play type and penalties (< 5% combined)

---

## 🚀 Run it locally

```bash
git clone https://github.com/skmakineni3/NFL-rezone-efficiency-analysis.git
cd NFL-rezone-efficiency-analysis
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
pip install -r requirements.txt
jupyter notebook Project_2.ipynb
```

---

## 🗂️ Project structure

```
├── Project_2.ipynb      # Full analysis: cleaning, EDA, hypothesis tests, models
├── pbp-2024.csv         # 2024 NFL play-by-play data
├── requirements.txt
└── README.md
```

---

## 🔭 Next Steps

- Add game context (score differential, time remaining) to the models
- Try gradient boosting and calibrate predicted probabilities
- Compare red zone efficiency across teams and play-callers

---

## 📬 Contact

**Suhas Makineni** · [GitHub](https://github.com/skmakineni3) · [skmakineni@gmail.com](mailto:skmakineni@gmail.com)
