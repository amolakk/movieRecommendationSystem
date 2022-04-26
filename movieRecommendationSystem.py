#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

metadata = pd.read_csv('movies_metadata.csv', low_memory = False)

# 1. Simple Recommender

# Mean rating of all movies
C = metadata['vote_average'].mean()
# Only use movies that are in the top 25% 
m = metadata['vote_count'].quantile(0.75)

# Make a new data frame for the movies that made above cutoff
qualified_movies = metadata.copy().loc[metadata['vote_count'] >= m]

# Computes weighted rating of each movie
def weighted_rating(x, m = m, C = C):
    
    v = x['vote_count']
    R = x['vote_average']
    
    # Use IMDB fomrula for calulcating movie rating
    return (v/(v+m) * R) + (m/(m+v) * C)

# Create new feature to hold our calculated weighted ratings from the function
qualified_movies['weighted_rating'] = qualified_movies.apply(weighted_rating, axis = 1)

# Sort movies by our weighted rating
qualified_movies = qualified_movies.sort_values('weighted_rating', ascending = False)


# 2. Content-Based Recommender (Suggest similar movies)

from sklearn.feature_extraction.text import TfidfVectorizer

# Create TF-IDF Vectorizer Object and remove all stop words in english
tfidf = TfidfVectorizer(stop_words = 'english')

# Fill null overviews with empty strings
metadata['overview'] = metadata['overview'].fillna('')

# Build the TF_IDF matrix by fitting and transforming data
tfidf_matrix = tfidf.fit_transform(metadata['overview'])

tfidf.get_feature_names()[5000:5010]

