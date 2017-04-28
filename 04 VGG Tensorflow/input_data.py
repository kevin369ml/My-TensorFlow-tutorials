
import tensorflow as tf
import numpy as np
import os

#%% Reading data

def read_cifar10(data_dir, is_train, batch_size, shuffle):
    """Read CIFAR10
    
    Args:
        data_dir: the directory of CIFAR10
        is_train: boolen
        batch_size:
        shuffle:       
    Returns:
        label: 1D tensor, tf.int32
        image: 4D tensor, [batch_size, height, width, 3], tf.float32
    
    """
    img_width = 32
    img_height = 32
    img_depth = 3
    label_bytes = 1
    image_bytes = img_width*img_height*img_depth
    
    
    with tf.name_scope('input'):
        
        if is_train:
            filenames = [os.path.join(data_dir, 'data_batch_%d.bin' %ii)
                                        for ii in np.arange(1, 6)]
        else:
            filenames = [os.path.join(data_dir, 'test_batch.bin')]
          
        filename_queue = tf.train.string_input_producer(filenames)
    
        reader = tf.FixedLengthRecordReader(label_bytes + image_bytes)
    
        key, value = reader.read(filename_queue)
           
        record_bytes = tf.decode_raw(value, tf.uint8)
        
        label = tf.slice(record_bytes, [0], [label_bytes])   
        label = tf.cast(label, tf.int32)
        
        image_raw = tf.slice(record_bytes, [label_bytes], [image_bytes])     
        image_raw = tf.reshape(image_raw, [img_depth, img_height, img_width])     
        image = tf.transpose(image_raw, (1,2,0)) # convert from D/H/W to H/W/D       
        image = tf.cast(image, tf.float32)

     
#        # data argumentation

#        image = tf.random_crop(image, [24, 24, 3])# randomly crop the image size to 24 x 24
#        image = tf.image.random_flip_left_right(image)
#        image = tf.image.random_brightness(image, max_delta=63)
#        image = tf.image.random_contrast(image,lower=0.2,upper=1.8)


        
        image = tf.image.per_image_standardization(image) #substract off the mean and divide by the variance 


        if shuffle:
            images, label_batch = tf.train.shuffle_batch(
                                    [image, label], 
                                    batch_size = batch_size,
                                    num_threads= 64,
                                    capacity = 20000,
                                    min_after_dequeue = 3000)
        else:
            images, label_batch = tf.train.batch(
                                    [image, label],
                                    batch_size = batch_size,
                                    num_threads = 64,
                                    capacity= 2000)
        ## ONE-HOT      
        n_classes = 10
        label_batch = tf.one_hot(label_batch, depth= n_classes)
        label_batch = tf.cast(label_batch, dtype=tf.int32)
        label_batch = tf.reshape(label_batch, [batch_size, n_classes])
        
        return images, label_batch
#%%






