# VGG with Tensorflow
  1. VGG paper review
  2. Tensorflow implementation
  3. Fine-tuning from parameters trained on ImageNet dataset
# VGG Paper
https://arxiv.org/abs/1409.1556

# Paper keypoints
  1. the effect of the convolutional network **depth** on its accuracy in the large-scale image recognition setting
  
  2. using an architecture with very small (3 X 3) convolution filters, with stride 1
  
 3. max-pooling is performed over a 2 × 2 pixel window, with stride 2
  
 4. conv + 3 fully-connected layers (number of FC neurons: 4096 > 4096 > n_classes)
  
  5. learning rate decay, parameter initializaiton from pre-trained models, etc.
# Screenshots:

