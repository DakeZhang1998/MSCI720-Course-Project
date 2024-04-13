import pandas as pd
from tqdm import tqdm
from alpha_beta_gamma_ndcg import AlphaBetaGammaNDCG


algorithms = [
    # 'Popular.results',
    'ImplicitMF.results',
    # 'IIIm_120_15_0001.results',
    # 'UUIm_30_2_001.results',
    # 'Popular_Reranked.results',
    'ImplicitMF_Reranked.results',
    'IIIm_120_15_0001_Reranked.results',
    'UUIm_30_2_001_Reranked.results'
]

alpha_beta_gamma_ndcg = AlphaBetaGammaNDCG(
    alpha=0.7, beta=0.01, gamma=0, delta=0,
    training_data_path='../data/train-dataset/ml-implicit/ratings.csv',
    qrels_path='../1/output/test-ratings.binary.qrels',
    movie_info_path='../data/full-dataset/ml-implicit/movies.csv'
)

for algorithm in algorithms:
    # predictions = pd.read_csv(f'../data/lenskit-recs/{algorithm}', sep=' ', header=None,
    #                           names=['user_id', 'iteration', 'movie_id', 'rank', 'score', 'identifier'])
    predictions = pd.read_csv(f'./output/{algorithm}', sep=' ', header=None,
                              names=['user_id', 'iteration', 'movie_id', 'rank', 'score', 'identifier'])
    # assert predictions.shape[0] == 1011000
    score = 0
    user_ids = predictions.user_id.unique()
    for user_id in tqdm(user_ids):
        recommendations = predictions[predictions.user_id == user_id].movie_id.values.tolist()[:10]
        score_pre_user = alpha_beta_gamma_ndcg.get_alpha_beta_gamma_ndcg_at_k(k=10, user_id=user_id,
                                                                              recommended_movie_ids=recommendations)
        score += score_pre_user

    score = score / 1011

    print(f'{algorithm}: {score}')
