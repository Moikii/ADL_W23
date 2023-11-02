# SHEL: Dataset generation for Single German playing cards and object detection using YOLO

## Information

Due to no available local GPU, the training of the model is done on Google Colab. Because of this, the [Jupyter Notebook containing the pipeline](../src/pipeline.ipynb) **cannot be run locally**. The 

## Used error metrics
For the evaluation we focus on precision and recall as metrics. In our opinion, both of these aspects are equally important. To keep the correct scores, we need to detect all cards (high recall), but also have to classify them correctly (high precision). Based on the accuracy measurement provided in [the paper](https://arxiv.org/pdf/2109.11861.pdf), we **aim for a precision over 0.99 and a recall over 0.90** . Because the way our image generation works when allowing overlapping pictures, it may happen that a card is completely overlapped by another card, which makes it impossible to detect, leading to lower recall. We try to account for that, but do not know, whether this value is adjusted appropriately. We may also train models on images with no overlapping cards to check the difference.

In theory we could also use some aspects of the game to ensure a correct score count. For example, it would be possible to safe the top probabilities of a detected card, which can then be used to correct a wrong classification after a game (e.g when a card was detected twice during a round, which is impossible). But this is not part of the project, but would make sense to implement, if the usecase was more critical.



## Usage of pipeline in Jupyter Notebook
The usage of the [pipeline](../src/pipeline.ipynb) is described in the notebook directly.


## Results
Below are the results of our trained models. We varied the dataset size, as well as the boolean variable, which decides if overlapping cards are allowed. The training of a model takes quite some time, which is why we stuck only to these two variables, and fixed the others to the following values:

- max_number_of_cards_per_image = 4
- min_size = 0.2
- max_size = 0.7
- seed = 42

We achieved the following values for our chosen error metrics:

|Dataset size|Overlapping|Epochs|Precision|Recall|
|---|---|---|---|---|
|6,000|True|10|0.97032|0.9245|
|6,000|False|10|0.90016|0.89837|
|60,000|True|10|0.99256|0.97052|
|60,000|False|10|0.89694|0.89706|

We can see that, against our expectations, the dataset with overlapping cards achieves higher values for precision and recall. Training on more data lead to better results for the overlapping case, while the plateau for non-overlapping cards is already reached with the 6000 image dataset. This can also be seen in the plots below.

Additionally, we trained on a 60000 image dataset for 20 epochs, with varied card sizes (min_size = 0,1 max_size = 0,8). This lead to a precision of 0.99491 and recall of 0.9459. This is the best result regarding precision, but not for recall. We still ended up choosing this as the best model, because the lower recall can be explained with the larger range of card-sizes, because they lead to more completely covered cards than with the other parameters, but leaves us with a broader spectrum of card sizes, which also has influence on live predictions.

Training time of the models was around 1 hour (2 hours for 20 epochs), using the V100 GPU, that is provided when using Google Colab Pro.

![Precision on recall comparison for different models](../data/report_pictures/precision_recall_comparison.png)


As we can see in the image above, a plateau is reached by the non-overlapping datasets at around 0.9 for both, precision and recall. Datasets that allow overlapping cards perform better over time. This may be the case, because it learns to classify based on single features of a specifiy card and not the whole card as one. Initially we thought it would be the other way around, that non-overlapping cards were easier to detect and classify, but as we can see, the model learns more about the cards when they are not shown completely all the time.


The dataset that led to the best model, as well as the trained model itself are publicly available in our Google Drive and can be loaded easily into Colab for further evaluatio. Also the deployment of the application (Assignment 3) will download the model from there.


## Testing the code
The tests we implemented to check if pre- and postprocessing are working correctly are quite simple. We implemented the following tests:

- Dataset generation
    - dataset folder structure fits YOLO input
    - files in dataset are RGB images and of the right size
- Application launching
    - Downloaded model-file is useable YOLO model
    - Points in game are correctly calculated (Sum up to 157)

If  [code_tests.py](../src/code_tests.py) is executed, we should get back an 'OK' if everything works correctly.

## Time management

First, we saw after playing around with OpenCV for a while, that taking photos and then adjusting the brightness of them in the code is much simpler than shooting videos and filtering out the right frames. It took less time to shoot the photos as anticipated, because it was not that critical to take perfect pictures, due to the implemented preprocessing of them. The implementation of the dataset generation took a bit more time, because of the trial and error we got going on with the video-processing in the beginning and '1 pixel off'-errors when rounding.


One of the biggest issues with the implementation of the pipeline was to keep it as clean as possible, due to the outsourcing to Google Colab. We managed to make a dataset available on our Google Drive, so that a new dataset generation each time can be avoided. The storage on drive is limited, therefore we only keep the dataset that worked best available. If needed, the others used in the comparison above can be regenerated by using the documented Parameters.

Evaluating and testing with live video feeds was not that time intesive after a clean pipeline implementation.

Below is the anticipated time and the actual (approximate) time spent on a task:

|Topic|Short description|Anticipated time in hours|Actual time spent|
|---|---|---|---|
|Take videos|Create suitable setup, shoot videos|5|2|
|Implement dataset generation|Strategy, find useful packages, implementing, testing, variable parameters, aim for minimal user input|25|30|
|Model Training|Getting familiar with model, create setup for training and evaluation, train model, fine-tune with model parameters and dataset-generation parameters|15|30|
|Test with live data|Test model with live data from a connected camera|1|1|
