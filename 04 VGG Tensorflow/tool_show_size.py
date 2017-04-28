
import tensorflow as tf
import matplotlib.pyplot as plt
import tools

#%%
cat = plt.imread('cat.jpg') #unit8
plt.imshow(cat)
cat = tf.cast(cat, tf.float32) #[360, 300, 3]
x = tf.reshape(cat, [1, 360, 300, 3]) #[1, 360, 300, 3]

#%%

# First conv
with tf.variable_scope('conv1'):
    w = tools.weight([3,3,3,16], is_uniform=True)
    x_w = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
    
    b = tools.bias([16])
    x_b = tf.nn.bias_add(x_w, b)
    
    x_relu = tf.nn.relu(x_b)
    
    x_pool = tools.pool('test1', x_relu, kernel=[1,2,2,1], stride=[1,2,2,1],is_max_pool=True)

# Second conv
with tf.variable_scope('conv2'):
    w2 = tools.weight([3,3,16,32], is_uniform=True)
    x_w2 = tf.nn.conv2d(x_pool, w2, strides=[1, 1, 1, 1], padding='SAME')
    
    b2 = tools.bias([32])
    x_b2 = tf.nn.bias_add(x_w2, b2)
    
    x_relu2 = tf.nn.relu(x_b2)
    
    x_pool2 = tools.pool('test2',x_relu2, kernel=[1,2,2,1],stride=[1,2,2,1], is_max_pool=False)
    
    x_BN = tools.batch_norm(x_pool2)

#%%
def shape(x):
    return str(x.get_shape())

## First conv
print('\n')
print('** First conv: **\n')
print('input size: ', shape(x))
print('w size:', shape(w))
print('x_w size: ', shape(x_w))
print('b size: ', shape(b))
print('x_b size: ', shape(x_b))
print('x_relu size: ', shape(x_relu))
print('x_pool size: ', shape(x_pool))
print('\n')

## Second conv
print('** Second conv: **\n')
print('input size: ', shape(x_pool))
print('w2 size:', shape(w2))
print('x_w2 size: ', shape(x_w2))
print('b2 size: ', shape(b2))
print('x_b2 size: ', shape(x_b2))
print('x_relu2 size: ', shape(x_relu2))
print('x_pool2 size: ', shape(x_pool2))
print('x_BN size: ', shape(x_BN))
print('\n')




















