import pickle
from matplotlib import pyplot as plt
from sklearn.cluster import AffinityPropagation
from pandas import DataFrame
import numpy as np

import utils

# Read dataset
CUTOFF = 40
df_globo = utils.read_globo_csv("../data/globo/clicks/*.csv")
df_globo.reset_index(inplace=True, drop=True)
df_globo.rename(columns={column: column.split("click_")[-1] for column in df_globo.columns}, inplace=True)

np_article_ids = np.array(df_globo['article_id'])
np_article_ids.sort()
np_article_ids = np.unique(np_article_ids)

# Read features
features = pickle.load(open("../data/tsne_features.p", "rb"))
np_clicks = np.ndarray((np_article_ids.shape[0], 2), dtype=float)
for i, item in enumerate(np_article_ids):
    np_clicks[i] = features[item]

# get cluster centers and label it with unique names
sample=10000
tsne_clicks_sample = utils.get_sample(np_clicks, sample)
clustering = AffinityPropagation(damping=0.9).fit(tsne_clicks_sample)

cluster_centers = utils.get_cluster_centers(np_clicks, clustering)
df_center = DataFrame(cluster_centers, columns=['x', 'y'])
df_center['label'] = utils.get_random_labels(df_center.shape[0])
while df_center['label'].unique().shape[0] < df_center.shape[0]:
    df_center['label'] = utils.get_random_labels(df_center.shape[0])

df_labeled_articles = DataFrame(np_article_ids, columns=['article_id'])
df_labeled_articles[['x', 'y']] = DataFrame(np_clicks)

# get corresponding centroid labels
print("[ INFO ] Applying fictional topic labels")
df_labeled_articles['label'] = ''
df_labeled_articles['label'] = df_labeled_articles.apply(lambda row: utils.get_nearest_label(row, df_center), axis=1)

# get corresponding centroid coordinates
df_labeled_articles['x_centroid'] = df_labeled_articles['label'].map(df_center.set_index('label')['x'])
df_labeled_articles['y_centroid'] = df_labeled_articles['label'].map(df_center.set_index('label')['y'])

np_uids = pickle.load(open("../data/np_uids_140.p", "rb"))

# reduces size of dataframe with only sampled users
df_reduced = df_globo[df_globo['user_id'].isin(np_uids)]
df_reduced['x_centroid'] = df_reduced['article_id'].map(df_labeled_articles['x_centroid'])
df_reduced['y_centroid'] = df_reduced['article_id'].map(df_labeled_articles['y_centroid'])

df_reduced.sort_values(by='timestamp', inplace=True)
df_reduced.reset_index(inplace=True, drop=True)

# Generate simulated sessions
sample_sessions = utils.build_simulated_sessions(np_uids, df_reduced)
list(map(utils.euclidean_from_centroid, sample_sessions))

cut_sessions = utils.get_cut_sessions(sample_sessions, CUTOFF)
