# Machine Learning Portfolio

Machine learning projects by Benjamin Junkins. Each project folder holds the code, the data it
runs on (or a script to download it), and a write-up covering the method, the results, and what
those results actually support.

## Projects

| Project                                                                               | Problem                                                                                               | Techniques                                                                                                       | Headline result                                                                                       |
| ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| [**Student Score Prediction**](projects/student-score-prediction)<br>_Course project_ | Predict secondary school final grades, and test whether it can be done before term grades exist       | scikit-learn pipelines, custom transformers, cross-validation, GridSearchCV, Tableau                             | RMSE 2.09 (R² 0.79) with prior grades, R² 0.05 without — early prediction does not hold up            |
| [**Spam Filter**](projects/spam-filter)<br>_Personal project_                         | Classify emails as spam or ham, served as a web API and as a desktop popup that scores a pasted email | Email parsing, text cleanup, TF-IDF, Random Forest, RandomizedSearchCV, FastAPI, Docker, GitHub Actions, tkinter | 97.4% test accuracy with 1.1% of real emails marked as spam (SpamAssassin corpus, duplicates removed) |

## Skills

Data cleaning and imputation, feature engineering, custom scikit-learn transformers, pipeline
and column-transformer construction, text preprocessing and TF-IDF features, cross-validation,
hyperparameter tuning, evaluating models against a business objective rather than a score alone,
Tableau for exploratory analysis, wrapping a trained model in a small desktop app, and serving
one as a tested web API in a Docker container.

## Running the code

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then open the notebook in whichever project folder you want. Each notebook keeps the outputs
from its original run, so it can be read through without executing anything.

The spam filter is a Python package rather than a notebook and has its own requirements, so
follow the setup in [its README](projects/spam-filter#setup) instead.
