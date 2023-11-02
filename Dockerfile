# app/Dockerfile

FROM python:3.10-slim

WORKDIR /app

# install libgl1, which is somehow missing, but required for streamlit to run
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -y
RUN apt install libgl1-mesa-glx -y
RUN apt-get install 'ffmpeg'\
   'libsm6'\
   'libxext6'  -y

ENV PYTHONDONTWRITEBYTECODE=1
COPY pip.conf /etc/pip.conf
COPY ./src /app/
COPY docker_requirements.txt /app/

# install ultralytics and torch without dependencies and then only the necessary ones
# result: docker image size went from 10.1GB to 2GB!
RUN pip3 install ultralytics --no-deps
RUN pip3 install torch==2.1.0+cpu torchvision==0.16.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install -r docker_requirements.txt

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
ENTRYPOINT ["streamlit", "run", "/app/application.py", "--server.port=8501", "--server.address=0.0.0.0"]

