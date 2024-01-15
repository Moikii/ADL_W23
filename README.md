# SHEL
In this project called SHEL (for Schelle, Herz, Eichel, Laub) for the course 'Applied deep learning' at the TU Wien, we aim to build an application, that is capable of keeping score of players in the game 'Jass'. For this we build our own dataset, because the cards used for this game are not standard poker cards and we found no dataset online. A pre-trained YOLO model will be fine-tuned on this dataset afterwards. Implementing the model in an application that can detect cards in realtime wraps up the project.

## Intall and use application

### Requirements:
- Usage of Python version 3.10 is required.
- To install and launch the application, [Docker](https://www.docker.com/) must be installed on your machine.
- Also the app [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam) should be installed on your phone, to connect the phone camera with the desktop application.

### Launch application

We provide two options to run the application on your local machine:

- **Download an already build docker-image:** To run the application faster and not having to build it, the docker-image can be downloaded and launched from the Github Container Repository with the command below.

```
docker run -p 8501:8501 ghcr.io/moikii/shel:latest
```

- **Build a new docker container:** To run the application, open a terminal in the cloned repository and execute the following commands. This might take a while, because the *ultralytics* package has a lot of dependencies that need to be installed.

```
docker build -t shel .
docker run -p 8501:8501 shel
```


## Running Tests
To test the code using the implemented unittests, make sure you open a terminal in the virtual environment, with the installed depencencies from `requirements.txt`. Execute the following command with an active virtual environment, in a terminal opened in the root-directory of this project:

```
python src/code_tests.py
```

## Submissions for the Course

### Assignment 1
The first assignmet was to find a fitting project for this course and to gather information to start and plan the project. The submission of the first assignment can be found in [A1_Initiate.md](./assignments/A1_Initiate.md).


### Assignment 2
In this part a dataset generation and model training pipeline will be implemented. We set ourselves a performance goal and try to reach it be fine-tuning dataset and model parameters. A final evaluation of different models gives us insight in the impact of various parameters on the model performance. The submission with the details can be found in [A2_Hacking.md](./assignments/A2_Hacking.md).

**Note:**\
**The training was done with Google Colab, due to no locally available GPU. Therefore the [Jupyter Notebook](src/pipeline.ipynb) contains code that can only be run there and not locally!**


### Assignment 3
The final part of the assignment is to deploy the model in an application. We try to build a [Streamlit Application](https://streamlit.io/). The [final report](./assignments/A3_Report.pdf) wraps up the project, containing a documentation and final toughts.

How the app works and why we made certain design decisions, is described in [A3_Deliver](./assignments/A3_Deliver.md).

Also, a short demo video is available on [Youtube](https://www.youtube.com/watch?v=WZPn0D6OPzg).