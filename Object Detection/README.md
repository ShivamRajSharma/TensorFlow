# Object Detection 
 
 Object detection is a computer vision technique that allows us to identify and locate objects in an image or video. In this repoistory various techniques that can be used for Object Detection.</br>
 
 ## 1. Turning a Image Classifier into an Object Detection.
 
 Instead of training a multiclass object detection, we convert a pre-trained image classifier into an object detection using a sliding window and pyramid image scaling approach. This approch works because a CNN only focuses on a certain parts of an image on which it is able to take it decision to classify the objects. The region where CNN focuses can be visualized using [Grad-CAM](https://arxiv.org/abs/1610.02391). The results of Face Detection are available inside the notebook Classification_To_Localization.ipynb .</br>
 
 ### OUTPUT : 
 <p align="center">
  <img src="https://github.com/ShivamRajSharma/TensorFlow/blob/master/Object%20Detection/Output/Classification_to_Localization.png"/>
</p>
 
 
 ## 2. Face Detection from Scratch
 
 We use a pretrained MobileNetV2 model as our base model and fine tune its last layers for the face detection. We use faces class of the [PASCAL-VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/) to train our network. The results of Face Detection are available inside the notebook Face_Detection_From_Scratch.ipynb . </br> 
 
 ### OUTPUT : 
 <p align="center">
  <img src="https://github.com/ShivamRajSharma/TensorFlow/blob/master/Object%20Detection/Output/Face_Detection.png"/>
</p>
 
 ## 3. Multiclass Object detection from Scratch
 We use a pretrained EfficientNetB0 model as our base model and fine tune its last layers for the Multiclass Object Detection. We use subset of classes present in [PASCAL-VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/) to train our network. The results of Face Detection are available inside the notebook Multiclass_Object_Detection.ipynb . </br>
 
 ### OUTPUT : 
 <p align="center">
  <img src="https://github.com/ShivamRajSharma/TensorFlow/blob/master/Object%20Detection/Output/MultiClass_Object_Dectection.png" />
</p>
 
