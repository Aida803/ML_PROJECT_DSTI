# Book Rating Prediction

An end-to-end machine learning project predicting a book's average rating from its
metadata. Covers data cleaning, exploratory analysis, feature engineering, model
comparison, and a Streamlit web application for live predictions.


## Project Structure

```text
ML_PROJECT_DSTI/
|-- App/
|   |-- app.py                              # Streamlit interface
|   `-- features.py                         # raw inputs -> 21 model features
|-- Data/
|   |-- Raw/books.csv                       # original dataset
|   |-- Clean/clean_books.csv               # malformed rows repaired
|   `-- Processed/
|       |-- clean_books_features.csv        # model-ready, numeric only
|       `-- clean_books_with_text.csv       # features plus title/author/publisher
|-- Model/book_rating_random_forest_model.joblib
|-- Notebooks/
|   |-- 01_ml_pipeline.ipynb                # cleaning, EDA, training, evaluation
|   `-- 02_feature_pipeline_prototype.ipynb # building and testing features.py
|-- requirements.txt
`-- README.md
```

## Method

1. Repair four rows where an unescaped comma in the title shifted every column right.
2. Strip whitespace, check duplicates, drop rows with a zero average rating.
3. Reduce `publication_date` to a year; flag audiobooks by keyword in title or publisher.
4. Replace missing page counts with the median for printed books.
5. Engineer features from the title, authors, page count and rating counts.
6. Encode authors and publishers by frequency and ordinal value; group languages into
   five buckets and one-hot encode.
7. Drop redundant and weakly correlated features, leaving 21.
8. Compare linear regression, random forest and gradient boosting on an 80/20 split.
9. Validate the chosen model with 5-fold cross-validation and save it as a bundle.


## Run Locally

```powershell
conda create -n bookrating python=3.12
conda activate bookrating
pip install -r requirements.txt
streamlit run App/app.py
```

The app opens at `localhost:8501`. Enter a book's metadata and it returns a predicted
rating out of 5.


## Results

The dataset contains 11,128 books, reduced to 11,101 after cleaning, with 21 engineered
features.

| Model | R² | MAE |
|---|---|---|
| Linear Regression | 0.132 | 0.211 |
| **Random Forest (200 trees)** | **0.173** | **0.198** |
| Gradient Boosting | 0.131 | 0.202 |

- Selected model: **Random Forest**
- 5-fold cross-validated R²: **0.134 ± 0.024**
- Most important features: `log_ratings` (0.125), `num_pages` (0.124),
  `rev_per_rating` (0.118), `publisher_ord` (0.116)


## Limitations

- The dataset contains no genre, description or content features, which are likely the
  strongest real predictors of a book's rating.
- The two most important features derive from ratings the book has already received, so
  the model is weakest on unrated books — its intended use case.
- 5.3% of rows share a title and author with another edition. Because the split is
  random, the same work can appear in both training and test sets, making the reported
  metrics slightly optimistic.
- Authors and publishers are ordinal-encoded, which imposes an arbitrary ordering, and
  co-authored strings are treated as categories distinct from the same authors' solo work.
- The training data ends in 2020.