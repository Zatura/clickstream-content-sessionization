import pickle
import utils
from pandas import DataFrame
from sklearn.cluster import AffinityPropagation
import numpy as np

# Read dataset
df_globo = utils.read_globo_csv("../data/globo/clicks/*.csv")
df_globo.reset_index(inplace=True, drop=True)
df_globo.rename(columns={column: column.split("click_")[-1] for column in df_globo.columns}, inplace=True)

with open("../data/df_labeled.p", "rb") as file:
     features = pickle.load(file)

# Get unique article_ids
np_article_ids = np.array(df_globo['article_id'])
np_article_ids.sort()
np_article_ids = np.unique(np_article_ids)

# Associate features with article_ids
tsne_results = pickle.load(open("../data/tsne_features.p", "rb"))
np_clicks = np.ndarray((np_article_ids.shape[0], 2), dtype=float)
for i, item in enumerate(np_article_ids):
    np_clicks[i] = tsne_results[item]

with open("../data/np_uids_200.p", "rb") as file:
    np_user_ids = pickle.load(file)

df_labeled_articles = DataFrame(np_article_ids, columns=['article_id'])
df_labeled_articles[['x', 'y']] = DataFrame(np_clicks)

# Calculate cluster centers
sample=10000
tsne_clicks_sample = utils.get_sample(np_clicks, sample)
clustering = AffinityPropagation(damping=0.9).fit(tsne_clicks_sample)

cluster_centers = utils.get_cluster_centers(np_clicks, clustering)
df_center = DataFrame(cluster_centers, columns=['x', 'y'])
df_center['label'] = utils.get_random_labels(df_center.shape[0])
while df_center['label'].unique().shape[0] < df_center.shape[0]:
    df_center['label'] = utils.get_random_labels(df_center.shape[0])

# get centroid labels
print("[ INFO ] Applying fictional topic labels")
df_labeled_articles['label'] = ''
df_labeled_articles['label'] = df_labeled_articles.apply(lambda row: utils.get_nearest_label(row, df_center), axis=1)

# get centroid coordinates
df_labeled_articles['x_centroid'] = df_labeled_articles['label'].map(df_center.set_index('label')['x'])
df_labeled_articles['y_centroid'] = df_labeled_articles['label'].map(df_center.set_index('label')['y'])

# with open("data/df_labeled.p", "wb") as file:
#     pickle.dump(df_labeled_articles, file)

with open("../data/df_labeled.p", "rb") as file:
     df_labeled = pickle.load(file)

sessions = utils.build_simulated_sessions(np_user_ids, df_globo)

for session in sessions:
    session
# sample_sessions = build_simulated_sessions(np_user_ids, df_clicks)
# print(type(sample_sessions))
# print(len(sample_sessions))
# print(sample_sessions[0])
