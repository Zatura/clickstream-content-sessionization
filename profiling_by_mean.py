import pickle
from helpers import *
from pandas import DataFrame
import numpy as np

# Read dataset
df_clicks = read_globo_csv()
df_clicks.reset_index(inplace=True, drop=True)

with open("data/df_labeled.p", "rb") as file:
     features = pickle.load(file)

np_article_ids = np.array(df_clicks['click_article_id'])
np_article_ids.sort()
np_article_ids = np.unique(np_article_ids)

# Read features
tsne_results = pickle.load(open("data/tsne_results.p", "rb"))
tsne_clicks = np.ndarray((np_article_ids.shape[0], 2), dtype=float)
for i, item in enumerate(np_article_ids):
    tsne_clicks[i] = tsne_results[item]

with open("data/np_uids_200.p", "rb") as file:
    np_user_ids = pickle.load(file)

df_labeled_articles = pd.DataFrame(np_article_ids, columns=['article_id'])
df_labeled_articles[['X', 'Y']] = pd.DataFrame(tsne_clicks)

cluster_centers = get_cluster_centers(tsne_clicks, damping=0.9, sample=10000)
df_center = pd.DataFrame(cluster_centers, columns=['X', 'Y'])
df_center['label'] = get_random_labels(df_center.shape[0])
while df_center['label'].unique().shape[0] < df_center.shape[0]:
    df_center['label'] = get_random_labels(df_center.shape[0])

# get corresponding centroid labels
print("[ INFO ] Applying ficticious topic labels")
df_labeled_articles['label'] = ''
df_labeled_articles['label'] = df_labeled_articles.apply(lambda row: get_nearest_label(row, df_center), axis=1)

# get corresponding centroid coordinates
df_labeled_articles['x_centroid'] = df_labeled_articles['label'].map(df_center.set_index('label')['X'])
df_labeled_articles['y_centroid'] = df_labeled_articles['label'].map(df_center.set_index('label')['Y'])

# with open("data/df_labeled.p", "wb") as file:
#     pickle.dump(df_labeled_articles, file)

with open("data/df_labeled.p", "rb") as file:
     df_labeled = pickle.load(file)

sessions = build_simulated_sessions(np_user_ids, df_clicks)

for session in sessions:
    session
# sample_sessions = build_simulated_sessions(np_user_ids, df_clicks)
# print(type(sample_sessions))
# print(len(sample_sessions))
# print(sample_sessions[0])
