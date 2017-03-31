#By @Kevin Xu
#kevin28520@gmail.com
#Youtube: https://www.youtube.com/channel/UCVCSn4qQXTDAtGWpWAe4Plw
#Chinese weibo: http://bit.ly/2nAmOcO

    
#The aim of this project is to use TensorFlow to process our own data.
#    - cifar10_input.py:  read in data and generate batches
#    - cifar10.py: build the model architecture, train, evaluate


# I used Ubuntu with Python 3.5, TensorFlow 1.0*, other OS should also be good.
# I didn't use data argumentation, I spent less than 30 mins with 10K steps.


# data: cifar10 binary version
# https://www.cs.toronto.edu/~kriz/cifar.html
# data size: ~184M

# How to run?
# 0. you need to change the data directory
# 1. run cifar10.py
# 2. call train() in the console to train the model
# 3. call evaluate() in the console to test on the test data

# Note: 
# it is suggested to restart your kenel to train the model multiple times 
# (in order to clear all the variables in the memory)
# Otherwise errors may occur: conv1/weights/biases already exist......


#%%

import os
import os.path
import math

import numpy as np
import tensorflow as tf

import cifar10_input

#%%

BATCH_SIZE = 128
learning_rate = 0.05
MAX_STEP = 10000 # with this setting, it took less than 30 mins on my laptop to train.


#%%

def inference(images):
    '''
    Args:
        images: 4D tensor [batch_size, img_width, img_height, img_channel]
    Notes:
        In each conv layer, the kernel size is:
        [kernel_size, kernel_size, number of input channels, number of output channels].
        number of input channels are from previuous layer, if previous layer is THE input
        layer, number of input channels should be image's channels.
        
            
    '''
    #conv1, [5, 5, 3, 96], The first two dimensions are the patch size,
    #the next is the number of input channels, 
    #the last is the number of output channels
    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weights', 
                                  shape = [3, 3, 3, 96],
                                  dtype = tf.float32, 
                                  initializer=tf.truncated_normal_initializer(stddev=0.05,dtype=tf.float32)) 
        biases = tf.get_variable('biases', 
                                 shape=[96],
                                 dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(images, weights, strides=[1,1,1,1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name= scope.name)
    
    
    #pool1 and norm1   
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1],strides=[1,2,2,1],
                               padding='SAME', name='pooling1')
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75, name='norm1')
    
    
    #conv2
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3,3,96, 64],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.05,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[64], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')
    
    
    #pool2 and norm2
    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75,name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1,3,3,1], strides=[1,1,1,1],
                               padding='SAME',name='pooling2')
    
    
    #local3
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, shape=[BATCH_SIZE, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights',
                                  shape=[dim,384],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.004,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[384],
                                 dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    
    
    #local4
    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights',
                                  shape=[384,192],
                                  dtype=tf.float32, 
                                  initializer=tf.truncated_normal_initializer(stddev=0.004,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[192],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')
     
        
    # softmax
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear',
                                  shape=[192, 10],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.004,dtype=tf.float32))
        biases = tf.get_variable('biases', 
                                 shape=[10],
                                 dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')
    
    return softmax_linear

#%%

def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        
        labels = tf.cast(labels, tf.int64)
        
        # to use this loss fuction, one-hot encoding is needed!
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits\
                        (logits=logits, labels=labels, name='xentropy_per_example')

                        
#        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
#                        (logits=logits, labels=labels, name='xentropy_per_example')
                        
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name+'/loss', loss)
        
    return loss


#%% Train the model on the training data
# you need to change the training data directory below



def train():
    
    my_global_step = tf.Variable(0, name='global_step', trainable=False)
    
    
    data_dir = '/home/kevin/tensorflow/CIFAR10/data/cifar-10-batches-bin/'
    log_dir = '/home/kevin/tensorflow/CIFAR10/logs234/'
    
    images, labels = cifar10_input.read_cifar10(data_dir=data_dir,
                                                is_train=True,
                                                batch_size= BATCH_SIZE,
                                                shuffle=True)
    logits = inference(images)
    
    loss = losses(logits, labels)
    
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step= my_global_step)
    
    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()
    
    
    
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
    
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                    break
            _, loss_value = sess.run([train_op, loss])
               
            if step % 50 == 0:                 
                print ('Step: %d, loss: %.4f' % (step, loss_value))
                
            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)                
    
            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
        
    coord.join(threads)
    sess.close()





#%% To test the model on the test data




def evaluate():
    with tf.Graph().as_default():
        
        log_dir = '/home/kevin/tensorflow/CIFAR10/logs10000/'
        test_dir = '/home/kevin/tensorflow/CIFAR10/data/cifar-10-batches-bin/'
        n_test = 10000
        
        
        # reading test data
        images, labels = cifar10_input.read_cifar10(data_dir=test_dir,
                                                    is_train=False,
                                                    batch_size= BATCH_SIZE,
                                                    shuffle=False)

        logits = inference(images)
        top_k_op = tf.nn.in_top_k(logits, labels, 1)
        saver = tf.train.Saver(tf.global_variables())
        
        with tf.Session() as sess:
            
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
                return
        
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess = sess, coord = coord)
            
            try:
                num_iter = int(math.ceil(n_test / BATCH_SIZE))
                true_count = 0
                total_sample_count = num_iter * BATCH_SIZE
                step = 0

                while step < num_iter and not coord.should_stop():
                    predictions = sess.run([top_k_op])
                    true_count += np.sum(predictions)
                    step += 1
                    precision = true_count / total_sample_count
                print('precision = %.3f' % precision)
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)
    
#%%








        