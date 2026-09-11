# Machine Learning Portfolio

Machine learning projects by Benjamin Junkins. Each project folder holds the analysis notebook,
the data it runs on, and a write-up covering the method, the results, and what those results
actually support.

## Projects

| Project | Problem | Techniques | Headline result |
| --- | --- | --- | --- |
| [**Student Score Prediction**](projects/student-score-prediction)<br>*Course project* | Predict secondary school final grades, and test whether it can be done before term grades exist | scikit-learn pipelines, custom transformers, cross-validation, GridSearchCV, Tableau | RMSE 2.09 (R² 0.79) with prior grades, R² 0.05 without — early prediction does not hold up |

Currently working on: a spam filter (text classification).

## Skills

Data cleaning and imputation, feature engineering, custom scikit-learn transformers, pipeline
and column-transformer construction, cross-validation, hyperparameter tuning, evaluating models
against a business objective rather than a score alone, and Tableau for exploratory analysis.

## Running the code

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then open the notebook in whichever project folder you want. Each notebook keeps the outputs
from its original run, so it can be read through without executing anything.
