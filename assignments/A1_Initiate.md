# ADL_W23: Dataset generation for Single German playing cards and object detection using YOLO

## Introduction

Playing card detection is a specific topic in the area of computer vision and there are already working models out there, that can correctly identify playing cards in real time. This can be used to track a game for spectators, like it is mentioned in this paper about dataset generation for the game [Duplicate bridge](https://arxiv.org/pdf/2109.11861.pdf), to help players train [card counting](https://www.youtube.com/watch?v=Nf3zBJ2cDAs), or to keep track of the scores for each player. The latter application is also interesting for a game played in Vorarlberg, Austria called "*Jass*". In the end of each round, every player has to calculate their score by adding up the different values of the cards they won during the round. Unlike the cards used in the examples above, Jassa uses *Single German playing cards*, that are not as commonly used in other places. Because of this we were not able to find any dataset containing this type of cards, which will be adressed in this project.

## Project

The main part of this project will be to generate a suitable dataset for neural network training for Single German playing cards with a similar method proposed in the paper (reference). This dataset will then be used to train YOLO (reference), that is able to detect multiple objects in an image very fast. In the end we should be able to place multiple playing cards under a camera and let the program detect the position using bounding boxes and also the suit and value of the card.

Wether detection in different rotations of a card and partially overlapping cards is possible, will be interesting to see. This is because unlike poker cards, Single German playing cards are not symmetrical and even more important, they do not have their suit and value written in the corners. This makes classification of overlapping cards much more difficult and is therefore not a main goal of the project.

To wrap up the whole project, this playing card detection model will be integrated into an application, that is able to track a whole game and calculate the scores for each player.


### Dataset

To create a dataset, use a similar method as in [this paper](https://arxiv.org/pdf/2109.11861.pdf) We take pictures of the cards under different light conditions, annotate them once by hand and then generate images automatically. Features that will be considered in these images will be the following:

- Background
- Number of cards
- Position
- Light settings
- Size
- Rotation

If all of these make sense to use and lead to an improvement will be seen during the evaluation of the trained models.

Indentical to the paper, we use the 'Describing Textures Dataset' [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/) as backgrounds for our generated images.

The dataset generation will be implemented in a way that makes it easy to generate new datasets of a similar type (e.g. another type of playing cards). The way we intend it to work in the end is the following:

- Shoot short videos of each playing card with different light settings, with the camera straight above the card
- Giving the path to the folder containing the videos will suffice to process the videos, ending in cropped images of each card with differing lightning
- Giving the path to the folder containing the cropped images and parameters like number of images and the properties of the cards on the images (see above), generates automatically a dataset in the right format for YOLO to use.

### Model

The model we will be training is called YOLO (reference), that is currently one of the best models in detecting multiple objects in a single image in a very short amount of time. This makes it fitting for our task to detect multiple cards at once.





# Work-breakdown and schedule

Research for assignment 1

|Topic|Short description|Anticipated time in hours|
|---|---|---|
|Project idea|Come up with interesting ideas and select one of them|1|
|Research|Find Papers, come up with promising strategy, |5|
|Take photos|Create suitable and reproducable setup, documentat steps, shoot videos|10|
|Implement dataset generation|Strategy, find useful packages, implementing, testing, variable parameters, aim for minimal user input|20|
|Model Training|Getting familiar with model, create setup for training and evaluation, train model, fine-tune with model parameters and dataset-generation parameters|20|
|Test with live data|Test model with live data from a connected camera|5|
|Application|Implement model into application, detect cards, keep score of players|14|

# References

The currently used papers and websites for reference and inspiration are partially linked in this README.md file, but can also be found gathered together in this [file](./references/links.md).


