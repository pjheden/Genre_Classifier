import numpy as np

# Load training and eval data
filepath = './'
tr_data = np.load(filepath + './dataValMFCC_3C.npz')
print("keys:")
for key,val in tr_data.items():
    print(key)

train_data = np.asarray(tr_data["mfcc"], dtype=np.float32)
train_labels = np.asarray(tr_data["genre"], dtype=np.int32)
assert not np.any(np.isnan(train_data))
assert not np.any(np.isnan(train_labels))
print("training data size:", train_data.shape)
print("train_data max: ", np.max(train_data), np.min(train_data))
print("train_labels max: ", np.max(train_labels), np.min(train_labels))



unique, counts = np.unique(train_labels, return_counts=True)
print(dict(zip(unique, counts)))
