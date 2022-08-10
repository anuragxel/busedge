# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

FROM ubuntu:18.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y \
    git \
    python3.6 \
    python3-pip \
    gpg-agent \
    gnupg \
    python3-cairocffi \
    protobuf-compiler \
    python3-lxml \
    python3-tk \
    wget \
    lsb-release

RUN pip3 install --upgrade pip 

#-----------------#
# Your Dockerfile #
#-----------------#

# Install ROS related packages
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN apt-get update
RUN apt-get install -y ros-melodic-ros-base ros-melodic-rosbag python3-numpy python3-yaml ros-melodic-cv-bridge


# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser

ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/pip/3.6/get-pip.py && \
	python3 get-pip.py --user && \
	rm get-pip.py

RUN pip3 install --user pycryptodomex gnupg rospkg

# Install gabriel related packages
RUN pip3 install --user gabriel-client\
    gabriel_protocol\
    imageio\
    opencv-python \
    protobuf \
    py-cpuinfo \
    PyQt5 \
    'pyzmq==18.1' \
    'websockets==8.0' \
    zmq

# install Torch for Detectron2
# See https://pytorch.org/ for other options
# RUN pip3 install --user tensorboard cmake   # cmake from apt-get is too old
RUN pip3 install --user torch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html

# install detectron2
RUN pip3 install --user detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html
# RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
# RUN pip install --user -e detectron2_repo

ENV AM_I_IN_A_DOCKER_CONTAINER Yes



RUN mkdir -p /home/appuser/client_node
WORKDIR /home/appuser/client_node
COPY . /home/appuser/client_node

# ENTRYPOINT [ "sleep", "10000" ]
ENTRYPOINT ["bash", "./ros_entrypoint.sh", "python3", "./can_filter_node.py" ]
#ENTRYPOINT ["bash", "./ros_entrypoint.sh"]
#CMD [ "python3", "./filter_node.py" ]
