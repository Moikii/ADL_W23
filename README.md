# ADL_W23: Dataset generation for Single German playing cards and object detection using YOLO

## Introduction

Playing card detection is a specific topic in the area of computer vision with many working models out there, that can correctly identify playing cards in real time. This can be used to track a game for spectators, like it is mentioned in this paper about dataset generatin for the game [Duplicate bridge](https://arxiv.org/pdf/2109.11861.pdf), to help players train [card counting](https://www.youtube.com/watch?v=Nf3zBJ2cDAs), or to keep track of the scores for each player. The latter application is also interesting for a game played in Vorarlberg, Austria called "*Jassa*". In the end of each round, every player has to calculate their score by adding up the different values of the cards they won during the round. Unlike the cards used in the examples above, Jassa uses *Single German playing cards*, that are not as commonly used in other places. Because of this we were not able to find any dataset containing this type of cards, which will be adressed in this project.

## Project

The main part of this project will be to generate a suitable dataset for neural network training for Single German playing cards with a similar method proposed in the paper (reference). This dataset will then be used to train YOLO (reference), that is able to detect multiple objects in an image very fast. In the end we should be able to place multiple playing cards under a camera and the program detects the position using bounding boxes and also the suit and value of the card.

Wether detection in different rotations of a card and partially overlapping cards is possible, will be interesting to see. This is because unlike poker cards, Single German playing cards are not symmetrical and even more important, do not have their suit and value written in the corners. This makes classification of overlapping cards much more difficult and is therefore not a main goal of this project.

If enough time is available, an additional goal would be to integrate the model in an application, that is able to track a whole game and calculate the scores for each player.


### Dataset

To create a dataset, use a similar method as in [this paper](https://arxiv.org/pdf/2109.11861.pdf) We take pictures of the cards under different light conditions, annotate them once by hand and then generate images automatically. Features that will be considered in these images will be:

- Background
- Position
- Light settings
- Number of cards
- Size
- Orientation
- Angle?


Indentical to the paper, we use the 'Describing Textures Dataset' [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/) as backgrounds for our generated images.

### Model

The model we will be training is called YOLO (reference), that is currently one of the best models in detecting multiple objects in a single image in a very short amount of time. This makes it fitting for our task to detect multiple cards at once.


# Work-breakdown and schedule





# References

The currently used papers and websites for reference and inspiration are already partially linked in this README.md file, but can also be found gathered together in this [file](./References/links.md)


