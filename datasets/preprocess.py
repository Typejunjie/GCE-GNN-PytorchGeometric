import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import numpy as np

pre_data = pd.read_csv('./yoochoose/raw/yoochoose1_64.dat', index_col=0)

def get_seq(pre_data):
    pre_data['item_id'] = LabelEncoder().fit_transform(pre_data.item_id)
    n_node = pre_data.item_id.max()
    seq_list = {}
    sess_id = ''
    item_id = ''
    count = {}
    for index, row in tqdm(pre_data.iterrows(), colour='green', desc='Extracting sequence', leave=False):
        sess_id = row['sess_id']
        item_id = int(row['item_id']) + 1
        if sess_id not in seq_list:
            seq_list[sess_id] = [item_id]
        else:
            seq_list[sess_id].append(item_id)
        
        if item_id not in count:
            count[item_id] = 1
        else:
            count[item_id] += 1

    return seq_list, count, n_node + 1

seq_list, count, n_node = get_seq(pre_data)

for key in list(seq_list):
    if len(seq_list[key]) == 1:
        del seq_list[key]

for key in list(seq_list):
    filseq = list(filter(lambda x: count[x] > 4, seq_list[key]))
    if len(filseq) < 2:
        del seq_list[key]

del count
del key

seq = []

for key in list(seq_list):
    seq.append(seq_list[key])
seq_list = seq

del seq, key

index_test = np.array(range(int(len(seq_list) * 0.1)))
test_seq = []
for index in index_test:
    test_seq.append(seq_list[index])

count = 0
for index in list(index_test):
    index -= count
    del seq_list[index]
    count += 1

del index, count

def get_label(seq):
    new_seq = []
    label = []
    for list in seq:
        label.append(list[-1])
        new_seq.append(list[: -1])

    return new_seq, label

train_seq, train_label = get_label(seq_list)
test_seq, test_label = get_label(test_seq)

seq, label = [], []

del seq_list
del pre_data

# extend dataset
def extend_dataset(seq, label):
    extend_seq = []
    extend_label = []
    for i, session in enumerate(seq):
        for index, item in enumerate(session):
            if index == 0:
                continue
            extend_seq.append(session[: index])
            extend_label.append(item)
        
        extend_seq.append(session)
        extend_label.append(label[i])

    return extend_seq, extend_label

train_seq, train_label = extend_dataset(train_seq, train_label)
test_seq, test_label = extend_dataset(test_seq, test_label)

print(f'n_node {n_node}')
print('length of train session', len(train_seq))
print('length of test session', len(test_seq))

pickle.dump((train_seq, train_label), open('./yoochoose/raw/train.txt', 'wb'))
pickle.dump((test_seq, test_label), open('./yoochoose/raw/test.txt', 'wb'))