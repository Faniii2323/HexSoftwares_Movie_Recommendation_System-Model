# HexSoftwares_Movie_Recommendation_System-Model
Task 2,  Project 03 Movie Recommendation System

# Movie Recommendation System using Machine Learning

A complete Movie Recommendation System built in Google Colab using multiple recommendation techniques, including popularity-based ranking, weighted rating, content-based filtering, collaborative filtering, matrix factorization, and a hybrid recommender.

This project was developed as part of the **Hex Softwares Machine Learning Internship**.

---

## Project Objective

The objective of this project is to recommend movies to users based on:

- movie popularity and ratings
- movie metadata such as genre, overview, keywords, cast, and director
- user rating history
- similar user and item behavior
- content similarity
- predicted user preferences

The final system demonstrates different recommendation approaches and compares how each one works.

---

## Dataset Used

The project uses the following dataset files:

| File | Purpose |
|---|---|
| `movies_metadata.csv` | Contains movie title, genres, overview, popularity, vote average, vote count, release date, and other metadata |
| `ratings_small.csv` | Contains user ratings with `userId`, `movieId`, `rating`, and `timestamp` |
| `links_small.csv` | Maps MovieLens movie IDs with TMDB and IMDb IDs |
| `keywords.csv` | Contains movie keywords and tags |
| `credits.csv` | Contains cast and crew information |

### Dataset Shapes

| Dataset | Shape |
|---|---:|
| `movies_metadata.csv` | 45,466 rows × 24 columns |
| `ratings_small.csv` | 100,004 rows × 4 columns |
| `links_small.csv` | 9,125 rows × 3 columns |
| `keywords.csv` | 46,419 rows × 2 columns |
| `credits.csv` | 45,476 rows × 3 columns |

After cleaning and merging, the final movie dataset contains:

```text
8,809 movies × 31 columns
```

---


## Project Workflow

The notebook follows this workflow:

1. Project title and introduction
2. Importing libraries
3. Loading dataset files
4. Dataset description
5. Data inspection
6. Data cleaning
7. Exploratory Data Analysis
8. Feature engineering
9. Popularity-based recommendation
10. Weighted rating recommendation
11. TF-IDF content-based recommendation
12. Metadata content-based recommendation
13. KNN collaborative filtering
14. SVD matrix factorization
15. Hybrid recommendation system
16. Model evaluation
17. Testing recommendation functions
18. Saving models and files
19. Conclusion

---

## Data Cleaning

The project performs several cleaning steps:

- removes duplicate rows
- handles missing values
- converts movie IDs into numeric format
- removes invalid IDs
- converts `tmdbId` values where possible
- parses JSON-like columns such as genres, keywords, cast, and crew
- extracts useful fields such as:
  - genres
  - keywords
  - top cast members
  - director
- creates clean text features for similarity-based recommendation

---

## Exploratory Data Analysis

The notebook includes EDA visualizations for:

- top 10 movies by popularity
- top 10 movies by vote average
- distribution of movie ratings
- distribution of vote average
- number of movies by year
- most common genres
- most active users
- most rated movies
- correlation analysis of numeric features

---

## Recommendation Techniques Implemented

### 1. Popularity-Based Recommendation

Recommends movies based on popularity score.

### 2. Weighted Rating Recommendation

Uses an IMDb-style weighted rating formula:

```text
Weighted Rating = (v / (v + m) × R) + (m / (v + m) × C)
```

Where:

- `v` = vote count
- `m` = minimum votes required
- `R` = average rating of the movie
- `C` = mean vote average across all movies

### 3. TF-IDF Content-Based Recommendation

Uses movie overviews and TF-IDF vectorization to recommend similar movies.

Function:

```python
recommend_by_overview(movie_title, top_n=10)
```

### 4. Metadata-Based Recommendation

Uses genres, keywords, cast, and director to create a combined metadata feature called `soup`.

Function:

```python
recommend_by_metadata(movie_title, top_n=10)
```

### 5. KNN Collaborative Filtering

Uses the user-movie rating matrix and cosine similarity to recommend movies with similar rating behavior.

Function:

```python
recommend_by_knn(movie_title, top_n=10)
```

### 6. SVD Matrix Factorization

Uses the Surprise library to learn hidden user-movie rating patterns and predict ratings for unseen movies.

Function:

```python
recommend_by_svd(user_id, top_n=10)
```

### 7. Hybrid Recommendation System

Combines metadata-based content similarity with SVD predicted rating.

Function:

```python
hybrid_recommendation(user_id, movie_title, top_n=10)
```

---

## Model Evaluation

For matrix factorization using SVD, the notebook evaluates the model using RMSE and MAE.

| Model | RMSE | MAE |
|---|---:|---:|
| SVD Matrix Factorization | 0.9024 | 0.6954 |

Other recommendation approaches are evaluated through qualitative recommendation results because content-based and ranking-based systems do not use normal classification accuracy.

---

## Sample Recommendation Functions

The notebook tests the system using sample inputs such as:

```python
recommend_by_overview("Toy Story", 10)
recommend_by_metadata("Toy Story", 10)
recommend_by_knn("Toy Story", 10)
recommend_by_svd(user_id=1, top_n=10)
hybrid_recommendation(user_id=1, movie_title="Toy Story", top_n=10)
```

The functions also include safe handling for incorrect movie titles.

---

## Saved Files

The notebook saves important models and processed files in the `saved_recommender_files` folder:

- `tfidf_vectorizer.pkl`
- `count_vectorizer.pkl`
- `tfidf_matrix.npz`
- `count_matrix.npz`
- `movie_user_sparse_matrix.npz`
- `knn_collaborative_model.pkl`
- `svd_model.pkl`
- `cleaned_movies.csv`
- `performance_comparison.csv`

---

## Final Result

This project successfully builds a complete Movie Recommendation System using different machine learning-based recommendation techniques.

The hybrid recommendation system is the most advanced part of the project because it combines:

- content-based movie similarity
- user preference prediction
- collaborative filtering behavior

This makes the project suitable for real-world recommendation platforms such as Netflix-style movie recommendation systems.

---
##Author

Fani2323
Hex Software Internship Project
Computer Science 

---
