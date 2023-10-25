# SHEL: Dataset generation for Single German playing cards and object detection using YOLO

## Information

Due to no available local GPU, the training of the model is done on Google Colab. Because of this, the [Jupyter Notebook containing the pipeline](../src/pipeline.ipynb) **cannot be run locally**. The 

## Used error metrics
For the evaluation we focus on precision and recall as metrics. In our opinion, both of these aspects are equally important. To keep the correct scores, we need to detect all cards (high recall), but also have to classify them correctly (high precision). Based on the accuracy measurement provided in [the paper](https://arxiv.org/pdf/2109.11861.pdf), we **aim for a precision over 0.99 and a recall over 0.90** . Because the way our image generation works when allowing overlapping pictures, it may happen that a card is completely overlapped by another card, which makes it impossible to detect, leading to lower recall. We try to account for that, but do not know, whether this value is adjusted appropriately. We may also train models on images with no overlapping cards to check the difference.

In theory we could also use some aspects of the game to ensure a correct score count. For example, it would be possible to safe the top probabilities of a detected card, which can then be used to correct a wrong classification after a game (e.g when a card was detected twice during a round, which is impossible). But this is not part of the project, but would make sense to implement, if the usecase was more critical.


## Results
Below are the results of our trained models. The large dataset will be loaded from Google Drive when executing the [Notebook](../src/pipeline.ipynb) when no new dataset is generated.

We achieved the following values for our chosen error metrics:
|Dataset size|Overlapping|Epochs|Precision|Recall|
|---|---|---|---|---|
|5,000|True|10|?|?|
|5,000|False|10|?|?|
|50,000|True|10|?|?|
|50,000|False|10|?|?|

We can see that..#todo
maybe images of curves?


The best trained model was also saved on our Google Drive, so it can be downloaded for further evaluation and also deployment for the application that will be implemented during Assignmnet 3.


## Time management

First, we saw after playing around with OpenCV for a while, that taking photos and then adjusting the brightness of them in the code is much simpler than shooting videos and filtering out the right frames. It took less time to shoot the photos as anticipated, because it was not that critical to take perfect pictures, due to the implemented preprocessing of them. The implementation of the dataset generation took a bit more time, because of the trial and error we got going on with the video-processing in the beginning and '1 pixel off'-errors when rounding in the end.

One of the biggest issues with the implementation of the pipeline was to keep it as clean as possible, due to the outsourcing to Google Colab. We managed to make a dataset available on our Google Drive, so that a new dataset generation each time can be avoided.

Evaluating and making simple tests with live video feeds was not that time intesive after a clean pipeline implementation.

Below is the anticipated time and the actual (approximate) time spent on a task:

|Topic|Short description|Anticipated time in hours|Actual time spent
|---|---|---|
|Take videos|Create suitable setup, shoot videos|5|2|
|Implement dataset generation|Strategy, find useful packages, implementing, testing, variable parameters, aim for minimal user input|25|30|
|Model Training|Getting familiar with model, create setup for training and evaluation, train model, fine-tune with model parameters and dataset-generation parameters|15|25|
|Test with live data|Test model with live data from a connected camera|1|






code commenting/documentation/usage/code-tests/code-style/autoformat code