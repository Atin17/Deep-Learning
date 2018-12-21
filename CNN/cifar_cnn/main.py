import tensorflow as tf
import pickle
import sys
import os
import time
import numpy as np
import cv2
import pandas as pd


class_name = ["aeroplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

def classify_name(predicts):
    max =predicts[0,0]
    temp =0
    for i in range(len(predicts[0])):
        if predicts[0,i]>max:
                max = predicts[0,i]
                temp = i
    print(class_name[temp])
    
def load_dataset(dirpath='C:\tmp\data\cifar-10-batches-py'):
    X, y = [], []
    i = 0
    for i in range(5):
        path = "data_batch_" + str(i + 1)
        with open(path, 'rb') as f:
            batch = pickle.load(f,encoding='latin1')
        X.append(batch['data'])
        y.append(batch['labels'])
        
    X = np.concatenate(X)/np.float32(255)
    y = np.concatenate(y).astype(np.int32)

    X = np.dstack((X[:, :1024], X[:, 1024:2048], X[:, 2048:]))
    X = X.reshape((X.shape[0], 32, 32, 3))
    
    Y_train = np.zeros((40000,10),dtype = np.float32)
    Y_valid = np.zeros((10000,10), dtype = np.float32)
    y_test = np.zeros((10000,10),dtype = np.int32)
    
    X_train = X[-40000:]
    y_train = y[-40000:]
    
    X_valid = X[:-40000]
    y_valid = y[:-40000]
    
    for i in range(40000):
        a = y_train[i]
        Y_train[i,a] = 1

    for i in range(10000):
        a = y_valid[i]
        Y_valid[i,a] = 1
    
    path = 'test_batch'
    with open(path, 'rb') as f:
        batch = pickle.load(f,encoding='latin1')
    X_test = batch['data'] /np.float32(255)
    X_test = np.dstack((X_test[:, :1024], X_test[:, 1024:2048], X_test[:, 2048:]))
    X_test = X_test.reshape((X_test.shape[0], 32, 32, 3))
    y_t = np.array(batch['labels'], dtype=np.int32)

    for i in range(10000):
        a = y_t[i]
        y_test[i,a] = 1

    offset = np.mean(X_train, 0)
    scale = np.std(X_train, 0).clip(min=1)
    X_train = (X_train - offset) / scale
    X_valid = (X_valid - offset) / scale
    X_test = (X_test - offset) / scale
    return X_train, Y_train, X_valid, Y_valid, X_test, y_test

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

# Convolution neural network model
def build_model(input_val,w,b):

    conv1 = tf.nn.conv2d(input_val,w['w1'],strides = [1,1,1,1], padding = 'SAME')
    conv1 = tf.nn.bias_add(conv1,b['b1'])
    conv1 = tf.nn.relu(conv1)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    conv2 = tf.nn.conv2d(pool1,w['w2'],strides = [1,1,1,1], padding = 'SAME')
    conv2 = tf.nn.bias_add(conv2,b['b2'])
    conv2 = tf.nn.relu(conv2)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    conv3 = tf.nn.conv2d(pool2,w['w3'],strides = [1,1,1,1], padding = 'SAME')
    conv3 = tf.nn.bias_add(conv3,b['b3'])
    conv3 = tf.nn.relu(conv3)  
    pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    shape = pool3.get_shape().as_list()
    dense = tf.reshape(pool3,[-1,shape[1]*shape[2]*shape[3]])
    dense1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(dense,w['w4']),b['b4']))
    
    out = tf.nn.bias_add(tf.matmul(dense1,w['w5']),b['b5'])

    softmax = tf.nn.softmax(out)
    
    return out,softmax

def main_function(num_epochs=10):
    
    x = tf.placeholder(tf.float32,[None,32,32,3])
    y = tf.placeholder(tf.int32,[None,10])
    
    weights = {
        'w1': tf.Variable(tf.random_normal([5,5,3,120],stddev = 0.1)),
        'w2': tf.Variable(tf.random_normal([5,5,120,60],stddev = 0.1)),
        'w3': tf.Variable(tf.random_normal([4,4,60,30],stddev = 0.1)),
        'w4': tf.Variable(tf.random_normal([4*4*30,30],stddev = 0.1)),
        'w5': tf.Variable(tf.random_normal([30,10],stddev = 0.1))
    }

    biases = {
        'b1': tf.Variable(tf.random_normal([120],stddev = 0.1)),
        'b2': tf.Variable(tf.random_normal([60],stddev = 0.1)),
        'b3': tf.Variable(tf.random_normal([30],stddev = 0.1)),
        'b4': tf.Variable(tf.random_normal([30],stddev = 0.1)),
        'b5': tf.Variable(tf.random_normal([10],stddev = 0.1))
    }

    predict,out_predict = build_model(x,weights,biases)
    
    error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = predict,labels = y))
    optm = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(error)
    corr = tf.equal(tf.argmax(predict,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(corr,tf.float32))
    saver = tf.train.Saver()

    init = tf.global_variables_initializer()
    if not os.path.exists('model.ckpt.meta'):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(init)
        print("loading dataset...")
        X_train,y_train,X_val, y_val,X_test,y_test = load_dataset()
        print("Starting training...")
        for epoch in range(num_epochs):
            train_err = 0
            train_acc = 0
            train_batches = 0
            start_time = time.time()
            for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
                inputs, targets = batch
                sess.run([optm],feed_dict = {x: inputs,y: targets})
                err,acc= sess.run([error,accuracy],feed_dict = {x: inputs,y: targets})
                train_err += err
                train_acc += acc
                train_batches += 1
                
            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
                inputs, targets = batch
                sess.run([optm],feed_dict = {x: inputs,y: targets})
                err, acc = sess.run([error,accuracy],feed_dict = {x: inputs,y: targets})
                val_err += err
                val_acc += acc
                val_batches += 1

            print("Epoch {} of {} took {:.3f}s".format(
                    epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation accuracy:\t\t{:.2f} %".format(
                    val_acc / val_batches * 100))
        
        test_err = 0
        test_acc = 0
        test_batches = 0
        for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
            inputs, targets = batch
            err, acc = sess.run([error,accuracy],feed_dict = {x: inputs,y: targets})
            test_err += err
            test_acc += acc
            test_batches += 1
        print("Final results:")
        print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
        print("  test accuracy:\t\t{:.2f} %".format(
                test_acc / test_batches * 100))

        save_path = saver.save(sess,"d:\Study\\tensorflow_API\\cifar_cnn\\model.ckpt")

    else:
        sess = tf.Session()
        sess.run(init)
        saver.restore(sess,"model.ckpt")
    
    img = cv2.imread('test.jpg')
    new_img = cv2.resize(img,dsize = (32,32),interpolation = cv2.INTER_CUBIC)
    new_img = np.asarray(new_img, dtype='float32') / 256.
    img_ = new_img.reshape((-1, 32, 32, 3))
    
    prediction = sess.run(out_predict,feed_dict={x: img_})

    classify_name(prediction)
    sess.close()

if __name__ == '__main__':

    main_function()
