import src.utils as utils
import numpy as np
import pickle
from pandas import Series, DataFrame

MIN_SIZE = 20

df_globo = utils.read_globo_csv()
df_globo.reset_index(inplace=True, drop=True)
np_unique_uids = df_globo['user_id'].unique()
np.random.seed(1)
# np_unique_uids = np.random.choice(np_unique_uids, len(np_unique_uids), replace=False)
df_user_ids = DataFrame()
df_user_ids['user_id'] = Series(np.random.choice(np_unique_uids, len(np_unique_uids), replace=False))
df_user_ids['filter'] = True
np_uids = np.array([])
p = 1
total_length = p*len(np_unique_uids)


count = 0
df_size = df_user_ids.shape[0]


def filter_sessions(row: Series, df_clicks: DataFrame):
    global count
    count += 1
    if count % 1000 == 0:
        print(count/df_size*100)
    if len(df_clicks[df_clicks['user_id'] == row['user_id']]) > MIN_SIZE:
        return False


df_user_ids['filter'] = df_user_ids.apply(lambda row: filter_sessions(row, df_globo), axis=1)

count = 0
for uid in np_unique_uids:
    if len(df_globo[df_globo['user_id'] == uid]) > MIN_SIZE:
        np_uids = np.append(np_uids, uid)
    if len(np_uids) == total_length:
        break
    if count % 1000 == 0:
        print(count/total_length)
    count += 1

with open("np_uids_{}.pickle".format(MIN_SIZE), "wb") as file:
    pickle.dump(np_uids, file)