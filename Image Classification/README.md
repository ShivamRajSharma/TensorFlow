# Image Classification on CIFAR-10

In this repository we implent a multi class classifier on cifar-10 dataset using a CNN network with residual connections. We apply Image augmentation techniques to imporve our performance on validation dataset with an learning rate decay with warmup.

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
