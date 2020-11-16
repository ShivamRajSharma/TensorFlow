# Object Detection 
 
 Object detection is a computer vision technique that allows us to identify and locate objects in an image or video. In this repoistory various techniques that can be used for Object Detection.</br>
 
 ## 1) Turning a Image Classifier into an Object Detection.
 
 Instead of training a multiclass object detection, we instead convert a pre-trained image classifier into an object detection using a sliding window and pyramid image scaling approach. This approch works because a CNN only focuses on a certain parts of an image on which it is able to take it decisions which can be visualized using Grad-CAM.
 
 ## 2) Face Detection from Scratch
 
 We use a pretrained MobileNetV2 model as our base model and fine tune its last layers for the face detection. We use faces class of the [PASCAL-VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/) to train our network. The results of Face Detection are available inside the notebook. </br> ### Output
 
 ## 3) Multiclass Object detection from Scratch
 We use a pretrained EfficientNetB0 model as our base model and fine tune its last layers for the Multiclass Object Detection. We use subset of classes present in [PASCAL-VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/) to train our network. The results of Face Detection are available inside the notebook.
 
