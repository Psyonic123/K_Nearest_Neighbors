import numpy as np
import warnings
from collections import Counter
import pandas as pd
import random
import seaborn as sns
from sklearn.metrics import confusion_matrix

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total groups')
    distances = []
    # Grab euclidean distances
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features - np.array(predict)))
            distances.append([euclidean_distance, group])

    # Votes is the nearest euclidean distances up to k
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k
    return vote_result, confidence


# Import dataset
df = pd.read_csv('breast-cancer-wisconsin.data', na_values="?", header=0).fillna(-99999).drop('ID', axis=1)
full_data = df.values.tolist()
random.shuffle(full_data)
# Manual train/test split
test_size = 0.2
train_set = {2: [], 4: []}
test_set = {2: [], 4: []}
train_data = full_data[:-int(test_size * len(full_data))]  # Up to the last 20% of data
test_data = full_data[-int(test_size * len(full_data)):]  # The last 20% of data

for i in train_data:
    train_set[i[-1]].append(i[:-1])  # remove class column
for i in test_data:
    test_set[i[-1]].append(i[:-1])

# Call function
correct = 0
total = 0
for group in test_set:
    for data in test_set[group]:
        vote, confidence = k_nearest_neighbors(train_set, data, k=5)
        if group == vote:
            correct += 1
        else:
            print(confidence)
        total += 1
print('Accuracy: ', correct/total)
