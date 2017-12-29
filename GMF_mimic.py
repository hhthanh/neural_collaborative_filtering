import tensorflow as tf
from Dataset import Dataset as dataholder
from time import time
import numpy as np

def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [],[],[]
    num_users = train.shape[0]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while ((u, j) in train):
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels

t1 = time()
data = dataholder('Data/ml-1m')
train, testRatings, testNegatives = data.trainMatrix, data.testRatings, data.testNegatives
num_users, num_items = train.shape

print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
          %(time()-t1, num_users, num_items, train.nnz, len(testRatings)))


user, item, labels = get_train_instances(train, 4)
user_array, item_array, labels_array = np.array(user), np.array(item), np.array(labels)
dataset = tf.data.Dataset.from_tensor_slices((user_array, item_array, labels_array))
dataset = dataset.shuffle(100000)
dataset = dataset.batch(256)

iterator = dataset.make_one_shot_iterator()
sess = tf.Session()
for i in range(int(len(user_array)/256)):
	next_val = iterator.get_next()
	print(sess.run(next_val))

exit()