"""Deterministic cleaning and pre-release feature construction."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NUMERIC = ['log_budget', 'runtime', 'release_year', 'month_sin', 'month_cos']
CATEGORICAL = ['original_language', 'primary_company', 'director']
GENRES = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary',
          'Drama', 'Family', 'Fantasy', 'Foreign', 'History', 'Horror', 'Music',
          'Mystery', 'Romance', 'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western']
GENRE_COLS = ['genre_' + g for g in GENRES]
FEATURES = NUMERIC + CATEGORICAL + GENRE_COLS

def parse_list(value):
    if not isinstance(value, str) or not value.strip():
        return []
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError('Expected a JSON list')
    return parsed

def load_data():
    movies = pd.read_csv(ROOT / 'data/raw/tmdb_5000_movies.csv')
    credits = pd.read_csv(ROOT / 'data/raw/tmdb_5000_credits.csv')
    if movies.id.duplicated().any() or credits.movie_id.duplicated().any():
        raise ValueError('Movie IDs must be unique before merging')
    merged = movies.merge(credits[['movie_id', 'cast', 'crew']], left_on='id',
                          right_on='movie_id', how='left', validate='one_to_one', indicator=True)
    audit = {'raw_movies': len(movies), 'raw_credits': len(credits),
             'matched_credits': int((merged['_merge'] == 'both').sum()),
             'zero_budget': int((movies.budget == 0).sum()),
             'zero_revenue': int((movies.revenue == 0).sum())}
    merged['release_date'] = pd.to_datetime(merged.release_date, errors='coerce')
    positive = merged.budget.gt(0) & merged.revenue.gt(0)
    audit['positive_financial_records'] = int(positive.sum())
    # Budgets below $1,000 are implausible/suspicious records, not verified corrections.
    audit['excluded_budget_below_1000'] = int((positive & merged.budget.lt(1000)).sum())
    eligible = positive & merged.budget.ge(1000) & merged.release_date.notna()
    clean = merged.loc[eligible].copy()
    clean['runtime'] = clean.runtime.where(clean.runtime.gt(0), np.nan)
    clean['genres_list'] = clean.genres.map(lambda v: [g['name'] for g in parse_list(v)])
    clean['primary_company'] = clean.production_companies.map(
        lambda v: next((x['name'] for x in parse_list(v)), 'Unknown'))
    clean['director'] = clean.crew.map(
        lambda v: next((x['name'] for x in parse_list(v) if x.get('job') == 'Director'), 'Unknown'))
    clean['cast_names'] = clean.cast.map(lambda v: [x['name'] for x in parse_list(v)[:3]])
    clean['release_year'] = clean.release_date.dt.year
    clean['release_month'] = clean.release_date.dt.month
    clean['success'] = (clean.revenue >= 2 * clean.budget).astype(int)
    clean = clean.sort_values(['release_date', 'id']).reset_index(drop=True)
    audit.update({'modeling_records': len(clean), 'success_cases': int(clean.success.sum()),
                  'below_threshold_cases': int((clean.success == 0).sum()),
                  'first_release': str(clean.release_date.min().date()),
                  'last_release': str(clean.release_date.max().date())})
    return movies, clean, audit

def make_features(frame):
    """Same feature contract for training and interactive predictions."""
    out = pd.DataFrame(index=frame.index)
    out['log_budget'] = np.log1p(frame.budget.astype(float))
    out['runtime'] = frame.runtime.astype(float)
    out['release_year'] = frame.release_year.astype(int)
    month = frame.release_month.astype(int)
    out['month_sin'] = np.sin(2 * np.pi * month / 12)
    out['month_cos'] = np.cos(2 * np.pi * month / 12)
    for col in CATEGORICAL:
        out[col] = frame[col].fillna('Unknown').astype(str)
    for genre, col in zip(GENRES, GENRE_COLS):
        out[col] = frame.genres_list.map(lambda values: int(genre in values))
    return out[FEATURES]

def split_data(clean):
    """Whole calendar years avoid splitting same-date releases across partitions."""
    train = clean[clean.release_year <= 2010].copy()
    validation = clean[clean.release_year.between(2011, 2013)].copy()
    test = clean[clean.release_year >= 2014].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError('Empty temporal partition')
    return train, validation, test
