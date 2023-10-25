# SHEL
In this project called SHEL (for Schelle, Herz, Eichel, Laub) for the course 'Applied deep learning' at the TU Wien, we aim to build an application, that is capable of keeping score of players in the game 'Jass'. For this we build our own dataset, because the cards used for this game are not standard poker cards and we found no dataset online. A pre-trained YOLO model will be fine-tuned on this dataset afterwards. Implementing the model in an application that can detect cards in realtime wraps up the project.

**Note**:\
If you want to train and evaluate models, **open the [pipeline notebook](./src/pipeline.ipynb) in Google Colab**, because the implementation was done for Colab, due to no available local GPU.

## Submissions for the Course

### Assignment 1
The first assignmet was to find a fitting project for this course and to gather information to start and plan the project. The submission of the first assignment can be found in [A1_Initiate.md](./assignments/A1_Initiate.md).


### Assignment 2
In this part a dataset generation and model training pipeline will be implemented. We set ourselves a performance goal and try to reach it be fine-tuning dataset and model parameters. A final evaluation of different models gives us insight in the impact of various parameters on the model performance. The submission with the details can be found in [A2_Hacking.md](./assignments/A2_Hacking.md).


### Assignment 3
The final part of the assignment is to deploy the model in an application. Because of no previous experience and the complexity of the application using a live video feed, we stick to a local application that can be run as a python script in the terminal. The [final report](./assignments/SHEL_final_report) for this project contains more details to the implementation.

The usage of the application is described in the section below.

## Usage of Application

#todo
