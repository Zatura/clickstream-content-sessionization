import utils
import numpy as np
import pickle
from pandas import Series, DataFrame

MIN_SIZE = 200

df_globo = utils.read_globo_csv()
df_globo.reset_index(inplace=True, drop=True)
with open("data/np_uids_190.p", "rb") as file:
    np_unique_uids = pickle.load(file)
np.random.seed(1)
df_user_ids = DataFrame()
df_user_ids['user_id'] = Series(np.random.choice(np_unique_uids, len(np_unique_uids), replace=False))
df_user_ids['filter'] = True
p = 1
total_length = p*len(np_unique_uids)


count = 0
df_size = df_user_ids.shape[0]


def filter_sessions(row: Series, frame: DataFrame):
    global count
    count += 1
    if count % 1 == 0:
        print(count/df_size*100)
    if len(frame[frame['user_id'] == row['user_id']]) > MIN_SIZE:
        return False


df_user_ids['filter'] = df_user_ids.apply(lambda row: filter_sessions(row, df_globo), axis=1)
np_uids = np.array(df_user_ids[df_user_ids['filter'] == False]['user_id'])

with open("data/np_uids_{}.pickle".format(MIN_SIZE), "wb") as file:
    pickle.dump(np_uids, file)
