import numpy as np
import pandas as pd
from math import log2


class AlphaBetaGammaNDCG:
    def __init__(self, alpha, beta, gamma, delta, training_data_path, qrels_path: str, movie_info_path: str):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.training_data = pd.read_csv(training_data_path)
        # Columns: user_id, movie_id, rating, timestamp
        self.training_data.rename(columns={'userId': 'user_id', 'movieId': 'movie_id'}, inplace=True)
        self.training_data = self.training_data[self.training_data['rating'] >= 4].reset_index(drop=True)
        self.training_data = self.training_data[['user_id', 'movie_id']]
        self.qrels = pd.read_csv(qrels_path, sep=' ', header=None, names=['user_id', 'iteration', 'movie_id',
                                                                          'relevance'])
        self.qrels = self.qrels[self.qrels['relevance'] > 0][['user_id', 'movie_id']]

        self.movie_info = pd.read_csv(movie_info_path)
        # Columns: movi_id, title, genres, year
        self.movie_info.rename(columns={'movieId': 'movie_id'}, inplace=True)

        # Produce a dict of genres with counts
        self.genre_dict = {}
        for row in self.movie_info.itertuples():
            genre_list = []
            for genre in row.genres.strip().split('|'):
                genre_item = genre.replace(' ', '').strip().lower()
                if genre_item != '(nogenreslisted)':
                    genre_list.append(genre_item)
                    if genre_item not in self.genre_dict:
                        self.genre_dict[genre_item] = 0
            self.movie_info.at[row.Index, 'genres'] = genre_list

    def get_user_preferences(self, user_id: int, inference=False):
        if inference:
            user_preferences = self.training_data[self.training_data['user_id'] == user_id]
        else:
            user_preferences = pd.concat([self.training_data[self.training_data['user_id'] == user_id],
                                          self.qrels[self.qrels['user_id'] == user_id]], axis=0)
        genre_count = self.genre_dict.copy()
        user_preferences = user_preferences.merge(self.movie_info, on='movie_id', how='left').reset_index(drop=True)
        liked_movies = list(user_preferences.movie_id.unique())
        for row in user_preferences.itertuples():
            for genre in row.genres:
                if genre != '(nogenreslisted)':
                    genre_count[genre] += 1

        # Applying softmax
        counts = np.array(list(genre_count.values()))
        non_zero_count = np.count_nonzero(counts == 0)
        counts = counts * (1 - non_zero_count * self.gamma) / counts.sum()
        counts[counts == 0] = self.gamma
        return liked_movies, dict(zip(genre_count.keys(), counts))

    def get_alpha_beta_gamma_dcg_at_k(self, k: int, recommended_movie_ids: list, liked_movies, user_genre_probs,
                                      inference=False):
        genre_counts_with_like = self.genre_dict.copy()
        genre_counts_without_like = self.genre_dict.copy()
        dcg = 0
        for i in range(1, k + 1):
            multiplication = 1
            movie_id = recommended_movie_ids[i - 1]
            movie_genres = self.movie_info[self.movie_info['movie_id'] == movie_id].genres.values[0]
            for genre in movie_genres:
                if movie_id in liked_movies:
                    user_movie_genre_relevance = self.alpha
                else:
                    user_movie_genre_relevance = self.beta
                multiplication *= (1 - user_movie_genre_relevance * user_genre_probs[genre] *
                                   ((1 - self.alpha) ** genre_counts_with_like[genre]) *
                                   ((1 - self.beta) ** genre_counts_without_like[genre]))

                if movie_id in liked_movies:
                    genre_counts_with_like[genre] += 1
                else:
                    genre_counts_without_like[genre] += 1

            gain = (1 - multiplication) / log2(1 + i)
            dcg += gain
        return dcg

    def get_ideal_ranking_at_k(self, k: int, liked_movies: list, user_genre_probs: dict):
        candidate_movies = liked_movies.copy()
        ideal_ranking = []
        for i in range(1, k + 1):
            scores = []
            for movie in candidate_movies:
                scores.append(self.get_alpha_beta_gamma_dcg_at_k(k=i, recommended_movie_ids=ideal_ranking + [movie],
                                                                 liked_movies=liked_movies,
                                                                 user_genre_probs=user_genre_probs,
                                                                 inference=False))
            max_score = np.max(scores)
            max_indices = np.where(scores == max_score)[0]
            selected_index = np.random.choice(max_indices)

            ideal_ranking.append(candidate_movies[selected_index])
            candidate_movies.pop(selected_index)

        assert len(set(ideal_ranking)) == k
        return ideal_ranking

    def rerank_top_k_recommendations(self, k: int, user_id: int, recommended_movie_ids: list):
        liked_movies, user_genre_probs = self.get_user_preferences(user_id=user_id, inference=True)
        candidate_movies = recommended_movie_ids.copy()[1:]
        ideal_ranking = [recommended_movie_ids[0]]
        movie_genres = self.movie_info[self.movie_info['movie_id'] == recommended_movie_ids[0]].genres.values[0]
        genre_cumm_probs = dict(zip(self.genre_dict.keys(), [1] * len(self.genre_dict)))
        for genre in movie_genres:
            genre_cumm_probs[genre] *= (1 - self.alpha)

        for i in range(2, k + 1):
            scores = []
            for j, movie_id in enumerate(candidate_movies):
                movie_genres = self.movie_info[self.movie_info['movie_id'] == movie_id].genres.values[0]
                aspect = 1
                for genre in movie_genres:
                    user_movie_genre_relevance = max(1 - self.alpha + j * self.delta, 1)
                    aspect_prob = 1 - user_movie_genre_relevance * user_genre_probs[genre] * genre_cumm_probs[genre]
                    aspect *= aspect_prob
                scores.append(1 - aspect)

            max_score = np.max(scores)
            selected_index = np.where(scores == max_score)[0][0]

            ideal_ranking.append(candidate_movies[selected_index])
            candidate_movies.pop(selected_index)

            movie_genres = self.movie_info[self.movie_info['movie_id'] == ideal_ranking[-1]].genres.values[0]
            for genre in movie_genres:
                user_movie_genre_relevance = max(1 - self.alpha + i * self.delta, self.delta)
                genre_cumm_probs[genre] *= (1 - user_movie_genre_relevance)

        if len(set(ideal_ranking)) != 10:
            print(123)

        return ideal_ranking

    def get_alpha_beta_gamma_ndcg_at_k(self, k: int, user_id: int, recommended_movie_ids: list):
        liked_movies, user_genre_probs = self.get_user_preferences(user_id=user_id, inference=False)
        actual_dcg = self.get_alpha_beta_gamma_dcg_at_k(k=k, recommended_movie_ids=recommended_movie_ids,
                                                        liked_movies=liked_movies, user_genre_probs=user_genre_probs,
                                                        inference=False)
        ideal_ranking = self.get_ideal_ranking_at_k(k=k, liked_movies=liked_movies, user_genre_probs=user_genre_probs)
        ideal_dcg = self.get_alpha_beta_gamma_dcg_at_k(k=k, recommended_movie_ids=ideal_ranking,
                                                       liked_movies=liked_movies, user_genre_probs=user_genre_probs,
                                                       inference=False)
        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0


