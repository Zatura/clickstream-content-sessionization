import glob
import itertools
from itertools import cycle
import random
from sklearn.cluster import AffinityPropagation
import pandas as pd
import numpy as np
from pandas import DataFrame, Series
from matplotlib import pyplot as plt


def read_globo_csv(path: str) -> DataFrame:
    print("[ INFO ] reading dataset")
    df_clicks = DataFrame()
    files = glob.glob(path)
    # concatenate all files' data into a single dataframe
    csv_dataframes = [pd.read_csv(file) for file in files]
    df_clicks = pd.concat([df_clicks, *csv_dataframes])
    print("[ INFO ] finish reading dataset")
    return df_clicks


def get_error(cutoff_index, true_index, last_index):
    if cutoff_index < true_index:
        error = (true_index - cutoff_index) / true_index
    else:
        error = (cutoff_index - true_index) / (last_index - true_index)
    error = min(error, 1)
    return error


def get_sample(array: np.array, size: int):
    df_array = DataFrame(array)
    df_array_sample = df_array.sample(size)
    np_array_sample = np.ndarray((df_array_sample.shape[0], 2), dtype=float)
    np_array_sample = df_array_sample.sort_index().values
    return np_array_sample


def plot_globo_clicks(tsne_clicks, damping=0.5, sample=2000):
    # clustering
    tsne_clicks_sample = get_sample(tsne_clicks, sample)
    print(tsne_clicks_sample)
    clustering = AffinityPropagation(damping=damping).fit(tsne_clicks_sample)
    print("labels: ", clustering.labels_.size)
    print("clusters: ", len(clustering.cluster_centers_indices_))

    # plotting
    plot_affinity_propagation(clustering, tsne_clicks_sample)
    return tsne_clicks_sample[clustering.cluster_centers_indices_[:]]


def plot_affinity_propagation(clustering, tsne_clicks_sample):
    n_clusters_ = len(clustering.cluster_centers_indices_)
    colors = cycle('bgrcmy')
    for k, color in zip(range(n_clusters_), colors):
        class_members = clustering.labels_ == k
        cluster_center = tsne_clicks_sample[clustering.cluster_centers_indices_[k]]
        plt.plot(tsne_clicks_sample[class_members, 0], tsne_clicks_sample[class_members, 1], color + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=color, markeredgecolor='k', markersize=7)
        for x in tsne_clicks_sample[class_members]:
            plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], color)

def plot_tsne_points(tsne_features, cluster_centers):
    AREA = 0.1
    BLACK_COLOR = (0, 0, 0)
    plt.scatter(tsne_features[:, 0], tsne_features[:, 1], s=AREA, c=BLACK_COLOR, alpha=0.5)
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=30, c=(1, 0, 0), alpha=0.8)

def get_cluster_centers(tsne_clicks_sample, clustering):
    print("[ INFO ] Finding cluster centers trough affinity propagation")
    print("labels: ", clustering.labels_.size)
    print("n_clusters: ", len(clustering.cluster_centers_indices_))
    return tsne_clicks_sample[clustering.cluster_centers_indices_[:]]


def get_random_labels(amount=10, deterministic=True):
    colors = ["red", "green", "blue", "cyan", "magenta", "yellow", "black", "white", "orange", "pink", "purple",
              "aqua", "amber", "lazuli", "vanilla", "gray", "tan", "tangerine", "sand", "golden", "ginger", "rose",
              "ruby", "fuchsia", "salmon", "violet", "sky", "indigo", "lapis", "teal", "mint", "brown", "graphite"]
    topics = ["topics", "stories", "news", "contents", "subjects", "themes", "discussions", "novels", "articles",
              "phenomenons", "gossips", "rumors", "events", "occurrences", "incidents", "episodes", "disasters",
              "plots", "accidents"]
    labels = [f"{color}-{topic}" for color, topic in itertools.product(colors, topics)]
    if deterministic: 
        random.seed(1)
    random.shuffle(labels)
    return labels[:amount]


def get_nearest_label(row: Series, df_center: DataFrame):
    delta_x = row['x'] - df_center['x']
    delta_y = row['y'] - df_center['y']
    distances = __euclidean_distance(delta_x, delta_y)
    index = distances[distances == distances.min()].index[0]
    return df_center['label'][index]


def __euclidean_distance(x, y): np.sqrt(pow(x, 2) + pow(y, 2))


def set_article_label(row: Series, df_labeled_articles: DataFrame):
    global count
    count += 1
    if count % 10000 == 0:
        print(count)
    return df_labeled_articles[df_labeled_articles['article_id'] == row['article_id']]['label']


def labels_from_articles(row: Series, df_labeled_articles: DataFrame):
    global count
    count += 1
    _id = row['article_id']
    row['label'] = df_labeled_articles[df_labeled_articles['article_id'] == _id]['label']
    if count % 1000 == 0:
        print(count, ": ", row['label'].values[0])
    return row['label']


def build_simulated_sessions(user_ids: np.array, df_clicks: DataFrame, step: int = 1):
    print('[ INFO ] building simulated sessions')
    sample_sessions = []
    for index in range(0, len(user_ids) - step, step):
        df1 = df_clicks[df_clicks['user_id'] == user_ids[index]]
        df2 = df_clicks[df_clicks['user_id'] == user_ids[index + 1]]
        df_concatenated = pd.concat([df1, df2])
        df_concatenated.reset_index(inplace=True, drop=True)
        sample_sessions.append(df_concatenated)
        if (index % 1000) == 0:
            print(index)
    return sample_sessions


def calculate_sessions_mean_distance(user_ids: np.array, sessions: DataFrame, step: int = 1):
    distance_array_mean = np.zeros(shape=len(user_ids))
    distance_array_std = np.zeros(shape=len(user_ids))
    for index in range(0, len(user_ids) - step, step):
        df1 = sessions[sessions['user_id'] == user_ids[index]]
        delta_x = df1['x_centroid'] - df1['x_centroid'].shift(1)
        delta_y = df1['y_centroid'] - df1['y_centroid'].shift(1)
        df1['distance'] = __euclidean_distance(delta_x, delta_y)
        df1['distance'].fillna(value=0, inplace=True)
        distance_array_mean[index] = df1['distance'].mean()
        distance_array_std[index] = df1['distance'].std()
        if (index % 1000) == 0:
            print(index)
    return distance_array_mean, distance_array_std


def euclidean_from_centroid(session: DataFrame):
    delta_x = session['x_centroid'] - session['x_centroid'].shift(1)
    delta_y = session['y_centroid'] - session['y_centroid'].shift(1)
    session['distance'] = __euclidean_distance(delta_x, delta_y)
    session['distance'].fillna(value=0, inplace=True)
    return session


def get_cut_sessions(sessions: list, cutoff_threshold: float) -> list:
    cut_sessions = []
    for session in sessions:
        session.reset_index(inplace=True, drop=True)
        cuts = session[session['distance'] > cutoff_threshold]
        cutoff_index = cuts.index[0] if len(cuts) > 0 else None
        cut_session = (session, cutoff_index)
        cut_sessions.append(cut_session)
    return cut_sessions
