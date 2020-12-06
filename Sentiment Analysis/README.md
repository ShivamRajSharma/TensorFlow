# Imdb Movie Reviews Sentiment Analysis 

<p align="center">
  <img src="https://mk0ecommercefas531pc.kinstacdn.com/wp-content/uploads/2019/12/sentiment-analysis.png" height="280" />
</p>

Sentiment Analysis is the process of determining whether a piece of writing is positive, negative or neutral. A sentiment analysis system for text analysis combines natural language processing (NLP) and machine learning techniques to assign weighted sentiment scores to the entities, topics, themes and categories within a sentence or phrase.

Sentiment analysis helps data analysts within large enterprises gauge public opinion, conduct nuanced market research, monitor brand and product reputation, and understand customer experiences. In addition, data analytics companies often integrate third-party sentiment analysis APIs into their own customer experience management, social media monitoring, or workforce analytics platform, in order to deliver useful insights to their own customers.

## Result
  <img src="https://github.com/ShivamRajSharma/TensorFlow/blob/master/Sentiment%20Analysis/Output/output.png" />


## Dataset Information 

The model was trained on IMDB movie reviews from kaggle. The dataset contains 50K movie reviews with 25k positive and 25k negetive movie reviews. </br>
Dataset : https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

## Usage 

1) Install the required libraries mentioned in requirement.txt.
2) Download the dataset from url provided above and place it inside ``` input/ ``` folder.
3) Run ```python3 train.py``` and let the model train for 2-3 iterations.
4) To infer on the trained model run ```python3 predict.py```.

## Model Architecure 
<p align="center">
  <img src="https://miro.medium.com/max/489/1*27JmK8VBdphpSCWNb4MhNA.png" height="280" />
</p>

A Unidirectional based simple LSTM network.


## Extra Info
<pre>
1) Trainin Stratergy       : The whole network was trained from scratch.
2) Optimizer               : Adam optimizer.
4) Loss                    : Binary Cross-Entropy Loss.
5) Regularization          : Dropout
6) Performance Metric      : Accuracy.
7) Epochs Trained          : 2
8) Performance             : 92% Accuracy.
9) Training Time           : 2 minutes.
</pre>

## Further Improvement 
1) Larger Dataset
2) Attention based network.
