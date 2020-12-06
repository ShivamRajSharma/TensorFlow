# Image Classification on CIFAR-10

In this repository we implent a multi class classifier on CIFAR-10 dataset using a CNN network with residual connections. We apply Image augmentation techniques to imporve our performance on validation dataset with an learning rate decay with warmup.

## Results :
1) Accuracy Plot 
  <img src="https://github.com/ShivamRajSharma/TensorFlow/blob/master/Image%20Classification/Output/acc_plot.png" height="300" />

2) Confusion Matrix 
<p align="center">
  <img src="https://github.com/ShivamRajSharma/TensorFlow/blob/master/Image%20Classification/Output/confusion_matrix.png" height="300" />
</p>

As we can see the model has difficulty in differentiating between cats and dogs, cat and frog. Therefore adding more cats pictures in the  dataset of cats would solve the problem and we'll be able to achieve more accuracy


## Extra Info
<pre>
1) Trainin Stratergy       : The whole network was trained from scratch.
2) Optimizer               : Adam optimizer.
3) Learning Rate Scheduler : Custom expontential decay with warmup.
4) Loss                    : Categorical Cross-Entropy Loss.
5) Regularization          : Dropout, Image Augementation(ShiftScaleRotate, Flip, Transpose, Rotate, RandomBrightness) .
6) Performance Metric      : Accuracy.
7) Epochs Trained          : 40
8) Performance             : 80.86% Accuracy.
9) Training Time           : 30 minutes.
</pre>
