import torch
import nmslib
import pandas as pd
import numpy as np

from pathlib import Path


def cos_dist(a, b):
    return 1 - torch.nn.functional.cosine_similarity(a, b)


def create_index(root_dir, model_name):
    embedding_path = root_dir + 'vectors/' + model_name + '.tsv.gz'
    article_embeddings = pd.read_csv(embedding_path,
                                     sep='\t').set_index('image')
    Path(root_dir + 'knn_index/').mkdir(parents=True, exist_ok=True)
    index_path = root_dir + 'knn_index/' + model_name
    try:
        index = nmslib.init(method='hnsw', space='cosinesimil')
        index.loadIndex(index_path)
    except RuntimeError:
        index = nmslib.init(method='hnsw', space='cosinesimil')
        index.addDataPointBatch(article_embeddings.values)
        index.createIndex({'post': 2}, print_progress=True)
        index.saveIndex(index_path, save_data=True)
    return index, article_embeddings


def get_nearest_neighbors(index, query_vec, k=5):
    index.setQueryTimeParams({'efSearch': k})
    ids, distances = index.knnQuery(query_vec, k + 1)
    return ids[1:], distances[1:]


def get_nearest_neighbors_batch(index, query_vecs, k=5):
    index.setQueryTimeParams({'efSearch': k})
    res = np.array(index.knnQueryBatch(query_vecs, k + 1))
    ids, distances = res[:, 0, 1:], res[:, 1, 1:]
    return ids.astype(int), distances


def get_rank(index, article_embeddings, query_name, rank_name):
    query_id = article_embeddings.index.get_loc(query_name)
    rank_id = article_embeddings.index.get_loc(rank_name)

    query_vec = article_embeddings.iloc[query_id].values
    ids, dists = get_nearest_neighbors(index, query_vec,
                                       len(article_embeddings))

    ranks = np.where(ids == rank_id)[0]
    return ranks[0] if len(ranks) > 0 else -1


def torch_arctanh(x, eps=1e-6):
    x *= (1. - eps)
    return torch.log((1 + x) / (1 - x)) * 0.5


def to_tanh_space(x_original):
    return torch_arctanh((x_original * 2) - 1)


def to_original_space(x_tanh):
    return (torch.tanh(x_tanh) + 1.0) / 2.0
