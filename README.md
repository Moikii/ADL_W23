# SHEL
In this project called SHEL (for Schelle, Herz, Eichel, Laub) for the course 'Applied deep learning' at the TU Wien, we aim to build an application, that is capable of keeping score of players in the game 'Jass'. For this we build our own dataset, because the cards used for this game are not standard poker cards and we found no dataset online. A pre-trained YOLO model will be fine-tuned on this dataset afterwards. Implementing the model in an application that can detect cards in realtime wraps up the project.

**Note**:\
If you want to train and evaluate models, **open the [pipeline notebook](./src/pipeline.ipynb) in Google Colab**, because the implementation was done for Colab, due to no available local GPU.

## Submissions for the Course

### Assignment 1
The first assignmet was to find a fitting project for this course and to gather information to start and plan the project. The submission of the first assignment can be found in [A1_Initiate.md](./assignments/A1_Initiate.md).


### Assignment 2
In this part a dataset generation and model training pipeline will be implemented. We set ourselves a performance goal and try to reach it be fine-tuning dataset and model parameters. A final evaluation of different models gives us insight in the impact of various parameters on the model performance. The submission with the details can be found in [A2_Hacking.md](./assignments/A2_Hacking.md). **The training was done with Google Colab, therefore the [Jupyter Notebook](src/pipeline.ipynb) contains code that can only be run there and not locally!**


### Assignment 3
The final part of the assignment is to deploy the model in an application. We try to build a [Streamlit Application](https://streamlit.io/). The [final report](./assignments/SHEL_final_report) wraps up the project, containing a documentation and final toughts.

The installation and usage of the application is described in the section below.

## Usage of Application

### Requirements:
To install and launch the application, docker must be installed on your machine. Also the app 'IP Webcam' should be installed on your phone, to connect the phone camera with the desktop application.

### Launch application

We provide two options to run the application on your local machine:

- **Build a new docker container:** To run the application, open a terminal in the cloned repository and execute the following commands. This might take a while, because the *ultralytics* package has a lot of dependencies that need to be installed.

```
docker build -t shel .
docker run -p 8501:8501 shel
```


- **Download an already build docker-image:** Run the application faster, the docker-image can be downloaded, and additionally launch the application with the commands below.

```
#todo
```

Using Streamlit to deploy the app did not work with our implementation, because we use a IP-Camera, that is only connected to the local network. Rewriting the code to make it work online would take up too much time at this point. Additionally we do not have a external webcam at hand to capture the videos from a nice angle, which is the reason we went with the phone camera as IP-Webcam in the first place. The video lags a bit, but dows not influence the predicitons of the model itself, just the user experience.



    


#todo
code commenting/documentation/usage/autoformat style code
docker commands
video
report