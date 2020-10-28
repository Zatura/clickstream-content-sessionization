import pickle
from matplotlib import pyplot as plt
from helpers import *
from pandas import DataFrame

# Read dataset
CUTOFF = 40
df_clicks = read_globo_csv()
df_clicks.reset_index(inplace=True, drop=True)

np_article_ids = np.array(df_clicks['click_article_id'])
np_article_ids.sort()
np_article_ids = np.unique(np_article_ids)

# Read features
tsne_results = pickle.load(open("data/tsne_results.p", "rb"))
tsne_clicks = np.ndarray((np_article_ids.shape[0], 2), dtype=float)
for i, item in enumerate(np_article_ids):
    tsne_clicks[i] = tsne_results[item]

# get cluster centers and label it with unique names
cluster_centers = get_cluster_centers(tsne_clicks, damping=0.9, sample=10000)
df_center = pd.DataFrame(cluster_centers, columns=['X', 'Y'])
df_center['label'] = get_random_labels(df_center.shape[0])
while df_center['label'].unique().shape[0] < df_center.shape[0]:
    df_center['label'] = get_random_labels(df_center.shape[0])

df_labeled_articles = pd.DataFrame(np_article_ids, columns=['article_id'])
df_labeled_articles[['X', 'Y']] = pd.DataFrame(tsne_clicks)

# get corresponding centroid labels
print("[ INFO ] Applying ficticious topic labels")
df_labeled_articles['label'] = ''
df_labeled_articles['label'] = df_labeled_articles.apply(lambda row: get_nearest_label(row, df_center), axis=1)

# get corresponding centroid coordinates
df_labeled_articles['x_centroid'] = df_labeled_articles['label'].map(df_center.set_index('label')['X'])
df_labeled_articles['y_centroid'] = df_labeled_articles['label'].map(df_center.set_index('label')['Y'])

np_uids = pickle.load(open("data/np_uids_140.p", "rb"))

# reduces size of dataframe with only sampled users
df_reduced = df_clicks[df_clicks['user_id'].isin(np_uids)]
df_reduced['x_centroid'] = df_reduced['click_article_id'].map(df_labeled_articles['x_centroid'])
df_reduced['y_centroid'] = df_reduced['click_article_id'].map(df_labeled_articles['y_centroid'])

df_reduced.sort_values(by='click_timestamp', inplace=True)
df_reduced.reset_index(inplace=True, drop=True)

# Generate simulated sessions
sample_sessions = build_simulated_sessions(np_uids, df_reduced)
list(map(euclidean_from_centroid, sample_sessions))

cut_sessions = get_cut_sessions(sample_sessions, CUTOFF)
