import numpy as np
import pandas as pd


algorithms = [
    'Popular.results',
    'Popular_Reranked.results',
    'ImplicitMF.results',
    'ImplicitMF_Reranked.results',
    'IIIm_120_15_0001.results',
    'IIIm_120_15_0001_Reranked.results',
    'UUIm_30_2_001.results',
    'UUIm_30_2_001_Reranked.results'
]

movie_info = pd.read_csv('../data/full-dataset/ml-implicit/movies.csv')
# Columns: movi_id, title, genres, year
movie_info.rename(columns={'movieId': 'movie_id'}, inplace=True)
genre_dict = {}
for row in movie_info.itertuples():
    genre_list = []
    for genre in row.genres.strip().split('|'):
        genre_item = genre.replace(' ', '').strip().lower()
        if genre_item != '(nogenreslisted)':
            genre_list.append(genre_item)
            if genre_item not in genre_dict:
                genre_dict[genre_item] = 0
    movie_info.at[row.Index, 'genres'] = genre_list


for algorithm in algorithms:
    predictions = pd.read_csv(f'./output/{algorithm}', sep=' ', header=None,
                              names=['user_id', 'iteration', 'movie_id', 'rank', 'score', 'identifier'])

    score = 0
    user_ids = predictions.user_id.unique()
    for user_id in user_ids:
        genre_counts = genre_dict.copy()
        recommendations = predictions[predictions.user_id == user_id].movie_id.values.tolist()[:10]
        dissimilarities = []
        for i in range(len(recommendations) - 1):
            for j in range(i + 1, len(recommendations)):
                a_set = set(movie_info[movie_info['movie_id'] == recommendations[i]].genres.values[0])
                b_set = set(movie_info[movie_info['movie_id'] == recommendations[j]].genres.values[0])
                dissimilarities.append(1 - len(a_set.intersection(b_set)) / len(a_set.union(b_set)))
        score += 2 / (len(recommendations) * (len(recommendations) - 1)) * sum(dissimilarities)

    score = score / 1011

    print(f'{algorithm}: {score}')
