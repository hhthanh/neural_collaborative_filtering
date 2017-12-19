import tensorflow as tf
import numpy as np
from Dataset import Dataset
from evaluate import evaluate_model
import argparse
from time import time
import multiprocessing as mp
import sys
import math

def parse_args():
    parser = argparse.ArgumentParser(description="Run GMF.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size.')
    parser.add_argument('--regs', nargs='?', default='[0,0]',
                        help="Regularization for user and item embeddings.")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    return parser.parse_args()

def get_model(num_users, num_items, latent_dim, regs=[0, 0]):
    model = tf.Graph()
    with model.as_default():

		#GMF input placeholder
        user_input = tf.placeholder(shape=(1,), dtype='int32', name='GMF_input_user')
        item_input = tf.placeholder(shape=(1,), dtype='int32', name='GMF_input_item')

		#GMF embedding layers
        MF_embedding_user = tf.nn.embedding_lookup(tf.Variable(tf.random_normal(shape=(num_users, latent_dim), stddev=0.01)), ids=user_input, max_norm=regs[0])
        MF_embedding_item = tf.nn.embedding_lookup(tf.Variable(tf.random_normal(shape=(num_items, latent_dim), stddev=0.01)), ids=item_input, max_norm=regs[1])

		#GMF latent vectors of user and item
        user_latent = tf.reshape(MF_embedding_user, [-1,latent_dim])
        item_latent = tf.reshape(MF_embedding_item, [-1,latent_dim])
        print(user_latent.get_shape())
		#Element-wise product of user and item embedding
        product_vector = tf.multiply(user_latent,item_latent,name="GMF_product_vector")
        label = tf.placeholder(shape=(1,),dtype='int32',name='label')
        output = tf.identity(tf.layers.dense(product_vector, 1, activation=tf.sigmoid, kernel_initializer=tf.initializers.random_uniform), name="GMF_output")
        #prediction = tf.layers.Dense(units=1, activation=tf.sigmoid, kernel_initializer=tf.keras.initializers.lecun_uniform)(product_vector)
    return model

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

if __name__ == '__main__':
    args = parse_args()
    num_factors = args.num_factors
    regs = eval(args.regs)
    num_negatives = args.num_neg
    learner = args.learner
    learning_rate = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    verbose = args.verbose
    
    topK = 10
    evaluation_threads = 1 # mp.cpu_count()
    print("GMF arguments: %s" %(args))
    model_out_file = 'Pretrain/%s_GMF_%d_%d.h5' %(args.dataset, num_factors, time())

    # Loading data
    t1 = time()
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
          %(time()-t1, num_users, num_items, train.nnz, len(testRatings)))
    print("token-1")
    model = get_model(num_users, num_items, num_factors, regs)
    print("token0")
    user_input = model.get_operation_by_name("GMF_input_user")
    item_input = model.get_operation_by_name("GMF_input_item")
    output = model.get_operation_by_name("GMF_output")
    label =  model.get_operation_by_name("label")
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=output)
    print("token0")
    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    train_step = optimizer.minimize(loss)
    
    best_hr, best_ndcg, best_iter = 0, 0, -1

    init = tf.tf.global_variables_initializer()


    sess = tf.Session()
    print("token")
    sess.run(init)

    for epoch in range(epochs):
        t1 = time()
        # Generate training instances
        user_input_set, item_input_set, labels = get_train_instances(train, num_negatives)
        print(user_input_set)
        print(item_input_set)
        print(labels)
        # Training

        train_data={user_input: user_input_set, item_input: item_input_set, label: labels}
        sess.run([train_step, loss], feed_dict=train_data)

        t2 = time()
        print(t1+"-"+t2)
    for idx in range(len(testRatings)):
        u = rating[0]
        gtItem = rating[1]

        print(tf.Session.run(loss, {user_input:u, item_input:gtItem}))
    