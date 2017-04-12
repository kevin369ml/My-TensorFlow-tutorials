# We are going to solve two problems here
  1. How to do training and validataion at the same time with Tensorflow? (more details about this question: [here](http://stackoverflow.com/questions/41162955/tensorflow-queues-switching-between-train-and-validation-data))
  2. How to plot training and validation curvers on Tensorboard?
  
# The method (details are in the CODE)
  1. generate train and validation batch with two queues
  2. fetch the contents of each queue independently with sess.run()
  3. during training, use feed_dict to select which one to be pushed into the computational graph
  
# Screenshots:
  ![training](https://github.com/kevin28520/My-TensorFlow-tutorials/blob/master/01%20cats%20vs%20dogs/new_version/images/101.png?raw=true)
  
  
  ![result1](https://github.com/kevin28520/My-TensorFlow-tutorials/blob/master/01%20cats%20vs%20dogs/new_version/images/103.png?raw=true)
  
  
  ![result2](https://github.com/kevin28520/My-TensorFlow-tutorials/blob/master/01%20cats%20vs%20dogs/new_version/images/102.png?raw=true)
