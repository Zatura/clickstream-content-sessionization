import pickle
from matplotlib import pyplot as plt
from utils import *
from pandas import DataFrame

# Read dataset
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

np_uids = pickle.load(open("data/np_uids_100.p", "rb"))

# reduces size of dataframe with only sampled users
df_reduced = df_clicks[df_clicks['user_id'].isin(np_uids)]
df_reduced['x_centroid'] = df_reduced['click_article_id'].map(df_labeled_articles['x_centroid'])
df_reduced['y_centroid'] = df_reduced['click_article_id'].map(df_labeled_articles['y_centroid'])

df_reduced.sort_values(by='click_timestamp', inplace=True)
df_reduced.reset_index(inplace=True, drop=True)

# Generate simulated sessions
sample_sessions = build_simulated_sessions(np_uids, df_reduced)
list(map(euclidean_from_centroid, sample_sessions))

mean_session_cutoff = np.zeros(shape=len(sample_sessions))
for index, session in enumerate(sample_sessions):
    mean_session_cutoff[index] = session[session['user_id'] != session['user_id'].shift(1)]['distance'].iloc[-1]

cutoff_array = np.linspace(start=1, stop=60, num=15)
error_array = np.ones(shape=len(sample_sessions))

cut_sessions = get_cut_sessions(sample_sessions, 40)
# cuttof error calculation
error_list = []
for index1, cutoff in enumerate(cutoff_array):
    for index2, session in enumerate(sample_sessions):
        session.reset_index(inplace=True, drop=True)
        rows = session[session['distance'] > cutoff]
        cutoff_index = -1
        if len(rows) > 0:
            cutoff_index = rows.index[0]
        true_index = session[session['user_id'] != session['user_id'].shift(1)].index[1]
        last_index = len(session) - 1
        error_array[index2] = get_error(cutoff_index, true_index, last_index)
    error_list.append({"cutoff": cutoff, "mean": error_array.mean(), "std": error_array.std()})
    print({"cutoff": cutoff, "mean": error_array.mean(), "std": error_array.std()})

df_errors = pd.DataFrame(error_list)


######################################################
# OPCIONAL!!!!
# Este calculo abaixo demora(umas 3 horas), o resultado dele
# está no pickle df_clicks.pickle
######################################################
# df_clicks['label'] = ''
# count = 0
# df_clicks = df_clicks.apply(lambda row: labels_from_articles(row, df_labeled_articles), axis=1)
# df_clicks = df_clicks.sort_values(by=['click_timestamp'])
# df_clicks = df_clicks.sort_values(by=['user_id'], kind='mergesort')


######################################################
# Example of frequency chart from user id=22
######################################################
df_pickle = pickle.load(open("data/df_clicks.pickle", "rb"))
df_pickle[df_pickle['user_id'] == 22].sort_values(by=['label'], kind='mergesort')['label'].value_counts().plot.barh()
df_pickle.sort_values(by=['click_timestamp'])

area_original = 0.1
area_sample = 4
color_original = (0, 0, 0)
color_sample = (1, 0, 0)

# ploting all points
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], s=area_original, c=color_original, alpha=0.5)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=30, c=(1, 0, 0), alpha=0.8)
plot_globo_clicks(damping=0.9, sample=7000)

# scikti-learn.org plot example
# https://scikit-learn.org/stable/auto_examples/cluster/plot_affinity_propagation.html#sphx-glr-auto-examples-cluster-plot-affinity-propagation-py
