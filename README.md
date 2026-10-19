# Predicting U.S. Domestic Flight Arrival Delays

This project is a part of the ADS-505 (Applied Data Science for Business) course in the Applied Data Science Program at the University of San Diego.

**Project Status: Completed**

## Installation

The project runs in Python 3.11+ (developed on Python 3.14 with scikit-learn 1.9). All analysis is in one Jupyter notebook. The 30,000-row modeling sample is committed to the repository, so the notebook runs without downloading any raw data.

```bash
# 1. clone the repository
git clone https://github.com/aros-usd/ads-505-final-project.git
cd ads-505-final-project

# 2. create and activate a virtual environment
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

# 3. install dependencies
pip install -r requirements.txt

# 4. open the notebook from the repository root and choose "Run All"
jupyter notebook Technical_Notebook-Team10.ipynb
```

The notebook reads `Data/flights_sample_2025H1.csv` using a path relative to the repository root, so run the notebook from there. All randomness is fixed with `RANDOM_STATE = 123`, so results reproduce exactly.

**Optional - Rebuild the Sample:** The six raw monthly files are about 250 MB each. That is over GitHub's 100 MB limit, so they are excluded by `.gitignore`. To regenerate the sample:

1. Open the BTS download page (see Data Sources below). For each month from January to June 2025, select the year and period, tick `Prezipped File`, and download.
2. Unzip each file and save the CSVs as `Data/January.csv`, `Data/February.csv`, …, `Data/June.csv`.
3. From inside the `Data/` folder, run `python build_sample.py`. It writes `flights_sample_2025H1.csv` next to the script. Change `ROWS_PER_FILE` at the top of the script to draw a larger or smaller sample.

## Project Intro/Objective

The main purpose of this project is to predict, using only information known before departure, whether a scheduled domestic flight will arrive 15 or more minutes late. About one in five U.S. domestic flights arrives 15 or more minutes late. These delays cost airlines crew time, fuel, and gate capacity, and they ripple into missed connections and later flights. 

A model like this could help an airline's operations control center rank tomorrow's flights by delay risk. Staff could then focus low-cost early actions, such as standby crews, protected connections, and early passenger notices, on the flights most likely to need them.

## Partner(s)/Contributor(s)

| Team member | Contact | Contributions |
|---|---|---|
| **Cameron Moran** | cameronmoran@sandiego.edu | Notebook Sections 1-4: problem statement and data description, exploratory data analysis, data pre-processing and feature engineering, and data splitting. |
| **Christopher Guillen** | cguillen@sandiego.edu | Notebook Sections 5-6: model building strategy, model training, and hyperparameter tuning. |
| **Allen Ros** | aros@sandiego.edu | Notebook Sections 7-8: results and final model selection, discussion and conclusion, and the Model Brief. |

## Methods Used

- Exploratory Data Analysis
- Data Manipulation and Feature Engineering
- Predictive Modeling (binary classification)
- Machine Learning
- Hyperparameter Tuning (stratified k-fold cross-validation)
- Data Visualization

## Technologies

- Python
- Jupyter Notebook
- pandas
- scikit-learn
- Matplotlib, seaborn

## Project Description

### Dataset

- **Source:** U.S. Department of Transportation, Bureau of Transportation Statistics (BTS), Reporting Carrier On-Time Performance (1987-present). It has one row per non-stop domestic flight leg and 109 fields.
- **Size:** January-June 2025, about 3.5 million flights.
- **Sample:** `Data/build_sample.py` keeps 17 columns known before departure plus the target. It removes cancelled and diverted flights, which have no arrival delay to predict, and randomly samples 5,000 flights per month, giving 30,000 rows × 17 columns.
- **Target:** `ArrDel15`, which is 1 if the flight arrived 15 or more minutes late. About 21.9% of flights in the sample are delayed.

### Questions Explored

1. Can schedule-only information rank flights by delay risk better than a no-model baseline?
2. Which flight and schedule characteristics (time of day, day, month, carrier, airport) are most associated with arrival delay?
3. Which classification method performs best, and how should its output be used in practice?

### Approach

- **EDA:** Includes delay rates by day of week, month, and scheduled departure hour. Also the daily delay rate over time and a correlation heatmap.
- **Pre-processing:**
  - The data has no missing values.
  - Outliers in `Distance` and `CRSElapsedTime` were checked with the IQR rule and kept, because they are real longer flights.
  - Highly correlated columns were removed (`DistanceGroup`, `CRSElapsedTime`).
  - Engineered features: `departure_hour`, `arrival_hour`, `is_weekend`, and `is_holiday` (within 3 days of a major holiday).
- **Split:** stratified 60% training, 20% validation, 20% test.
- **Models:** an always on time baseline (`DummyClassifier`), Logistic Regression, Random Forest, and Histogram Gradient Boosting. Each was tuned with stratified 5-fold cross-validation on ROC AUC.

### Results

The tuned Histogram Gradient Boosting model was selected on validation ROC AUC.

| Model | Validation ROC AUC |
|---|---|
| Histogram Gradient Boosting (selected) | 0.655 |
| Random Forest | 0.654 |
| Logistic Regression | 0.638 |
| Always-on-time baseline | 0.500 |

On the untouched test set, the selected model reached ROC AUC 0.652, recall 61.3%, and precision 30.8% at the default threshold. Schedule information gives moderate, useful ranking of delay risk, but it cannot predict individual delays with high confidence.

### Challenges

- **Weak signal from schedule data:** Daily delay rates swing from about 4% to 46%, driven by weather and air traffic control events that schedule data can't see.
- **Class imbalance:** About 22% of flights are delayed, so models are judged on ROC AUC, recall, and precision rather than accuracy.
- **Data size:** The raw monthly files are too large for GitHub, so a reproducible sample is used.

### Data Source

Bureau of Transportation Statistics. (2025). Reporting carrier on-time performance (1987-present). U.S. Department of Transportation. https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr