#By @Kevin Xu
#kevin28520@gmail.com
#Youtube: https://www.youtube.com/channel/UCVCSn4qQXTDAtGWpWAe4Plw
#
#The aim of this project is to use TensorFlow to process our own data.
#    - input_data.py:  read in data and generate batches
#    - model: build the model architecture
#    - training: train

# I used Ubuntu with Python 3.5, TensorFlow 1.0*, other OS should also be good.
# With current settings, 10000 traing steps needed 50 minutes on my laptop.


# data: cats vs. dogs from Kaggle
# Download link: https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data
# data size: ~540M

# How to run?
# 1. run the training.py once
# 2. call the run_training() in the console to train the model.

# Note: 
# it is suggested to restart your kenel to train the model multiple times 
#(in order to clear all the variables in the memory)
# Otherwise errors may occur: conv1/weights/biases already exist......


#%%

import tensorflow as tf
import numpy as np
import os

#%%

# you need to change this to your data directory
train_dir = '/home/kevin/tensorflow/cats_vs_dogs/data/train/'

def get_files(file_dir):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''
    cats = []
    label_cats = []
    dogs = []
    label_dogs = []
    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        if name[0]=='cat':
            cats.append(file_dir + file)
            label_cats.append(0)
        else:
            dogs.append(file_dir + file)
            label_dogs.append(1)
    print('There are %d cats\nThere are %d dogs' %(len(cats), len(dogs)))
    
    image_list = np.hstack((cats, dogs))
    label_list = np.hstack((label_cats, label_dogs))
    
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]
    
    
    return image_list, label_list


#%%

def get_batch(image, label, image_W, image_H, batch_size, capacity):
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''
    
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])
    
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    
    ######################################
    # data argumentation should go to here
    ######################################
    
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    
    # if you want to test the generated batches of images, you might want to comment the following line.
    image = tf.image.per_image_standardization(image)
    
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 64, 
                                                capacity = capacity)
    
    #you can also use shuffle_batch 
#    image_batch, label_batch = tf.train.shuffle_batch([image,label],
#                                                      batch_size=BATCH_SIZE,
#                                                      num_threads=64,
#                                                      capacity=CAPACITY,
#                                                      min_after_dequeue=CAPACITY-1)
    
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    
    return image_batch, label_batch


 
#%% TEST
# To test the generated batches of images
# When training the model, DO comment the following codes




#import matplotlib.pyplot as plt
#
#BATCH_SIZE = 2
#CAPACITY = 256
#IMG_W = 208
#IMG_H = 208
#
#train_dir = '/home/kevin/tensorflow/cats_vs_dogs/data/train/'
#
#image_list, label_list = get_files(train_dir)
#image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
#
#with tf.Session() as sess:
#    i = 0
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(coord=coord)
#    
#    try:
#        while not coord.should_stop() and i<1:
#            
#            img, label = sess.run([image_batch, label_batch])
#            
#            # just test one batch
#            for j in np.arange(BATCH_SIZE):
#                print('label: %d' %label[j])
#                plt.imshow(img[j,:,:,:])
#                plt.show()
#            i+=1
#            
#    except tf.errors.OutOfRangeError:
#        print('done!')
#    finally:
#        coord.request_stop()
#    coord.join(threads)


#%%


























#%% Method 2: use tfrecord to build input pipeline.

#def _int64_feature(value):
#  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
#
#
#def _bytes_feature(value):
#  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
#
#def convert_to_tfrecord(images, labels, save_dir, name):
#    filename = os.path.join(save_dir, name + '.tfrecords')
#    writer = tf.python_io.TFRecordWriter(filename)
#    n_samples = len(labels)
#    
#    for i in np.arange(0, n_samples):
#        img = Image.open(images[i])
#        img = np.array(img)
#        height = img.shape[0]
#        width = img.shape[1]
#        img_raw = img.tostring()
#        label = labels[i]
#        example = tf.train.Example(
#                features=tf.train.Features(feature={
#                        'height': _int64_feature(height),
#                        'width': _int64_feature(width),
#                        'image_raw': _bytes_feature(img_raw),
#                        'label':_int64_feature(label)
#                                                    }))
#
#        writer.write(example.SerializeToString())
#    writer.close()
#    return filename
#
#
#def read_and_decode(filename_queue):
#    
#    reader = tf.TFRecordReader()
#
#    _, serialized_example = reader.read(filename_queue)
#
#    features = tf.parse_single_example(serialized_example,
#                                       features={
#                                               'height': tf.FixedLenFeature([], tf.int64),
#                                               'width': tf.FixedLenFeature([], tf.int64),
#                                               'image_raw': tf.FixedLenFeature([], tf.string),
#                                               'label': tf.FixedLenFeature([], tf.int64)})
#    
#    image = tf.decode_raw(features['image_raw'], tf.uint8)    
#    height = tf.cast(features['height'], tf.int32)
#    width = tf.cast(features['width'], tf.int32)
#    label = tf.cast(features['label'], tf.int32)
#    
#    image_shape = tf.stack([height, width, 3])
#    
#    image = tf.reshape(image, image_shape)
#        
#    # Random transformations can be put here: right before you crop images to predefined size. 
#    resized_image = tf.image.resize_image_with_crop_or_pad(image = image,
#                                           target_height = img_height,
#                                           target_width = img_width)
#    
#    
#    image, label = tf.train.shuffle_batch( [resized_image, label],
#                                                 batch_size=2,
#                                                 capacity=30,
#                                                 num_threads=2,
#                                                 min_after_dequeue=10)
#    
#    return image, label

#%%

# TO Test tfrecord method.

#_, _, validation, validation_label = get_train_validation_files(train_dir, 0.1)
#save_dir = '/home/kevin/tensorflow/cats_vs_dogs/tfrecords/'
#tfrecords_filename = convert_to_tfrecord(validation, validation_label,save_dir,'validation')
#filename_queue = tf.train.string_input_producer([tfrecords_filename])
#
#image, label = read_and_decode(filename_queue)
#
#
#with tf.Session()  as sess:
#    
#    sess.run(tf.global_variables_initializer())
#    
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(coord=coord)
#    
#    for i in np.arange(1):
#    
#        img, label = sess.run([image, label])
#        print('label: %d' %label[0])
#        plt.imshow(img[0, :,:,:])
#        plt.show()
#        print('label: %d' %label[1])
#        plt.figure()
#        plt.imshow(img[1, :,:,:])
#        plt.show()
#    coord.request_stop()
#    coord.join(threads)

#%%










#%%

#def get_test_file(file_dir):
#    test_list = []
#    for file in os.listdir(file_dir):
#        test_list.append(file_dir + file)
#    return test_list
#
##%%
#def get_test_batch(images_list, batch_size, mini):
#    
#    images_list = tf.cast(images_list, tf.string)
#    
#    with tf.name_scope('input'):
#        
#        input_queue = tf.train.string_input_producer(images_list)
#        
#        reader = tf.WholeFileReader()
#        _, value = reader.read(input_queue)
#        img = tf.image.decode_jpeg(value, channels=3)
#                
#        distorted_img = tf.image.resize_image_with_crop_or_pad(img, img_width, img_height) 
#        image = tf.image.per_image_standardization(distorted_img) 
#        
#
#        # !!! NOTE: the first parameter must be like [image] NOT image.
#        img_batch = tf.train.batch([image], 
#                                   batch_size= batch_size, 
#                                   num_threads = 16, 
#                                   capacity= mini+3*batch_size)
#        return img_batch
    
#%%   
    
    
