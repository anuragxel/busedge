# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

FROM nvidia/cuda:10.1-cudnn7-devel

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

ENV DEBIAN_FRONTEND noninteractive

#change the python version 
RUN apt-get update && apt-get install -y \
    python3.7 python3.7-dev python3.7-distutils \
    python3-opencv ca-certificates git wget sudo ninja-build

# Make python3 available for python3.7
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2
RUN update-alternatives --config python3
# Make python available for python3.7
RUN ln -sv /usr/bin/python3.7 /usr/bin/python

# create a non-root user
RUN wget https://bootstrap.pypa.io/pip/get-pip.py && \
python3.7 get-pip.py --user && \
rm get-pip.py

# Install apt dependencies
RUN apt-get update && apt-get install -y \
    git \
    gpg-agent \
    python3-cairocffi \
    protobuf-compiler \
    python3-pil \
    python3-lxml \
    python3-tk \
    wget \
    lsb-release \
    vim \
    python3-pip

# Important! Otherwise, it uses existing numpy from host-modules which throws error
RUN pip3 install --user numpy==1.20.3


# Install ROS related packages
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN apt-get update
RUN apt-get install -y ros-melodic-ros-base ros-melodic-rosbag python3-numpy python3-yaml ros-melodic-cv-bridge
RUN pip3 install pycryptodomex gnupg rospkg

# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip3 install --user torch torchvision cython
RUN pip3 install --user 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
RUN pip3 install --user pandas

RUN pip3 install --user 'git+https://github.com/facebookresearch/fvcore'
# install detectron2
RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
# This will build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ENV TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
RUN pip3 install --user -e detectron2_repo

# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"

# Set a var to indicate running in docker container
ENV AM_I_IN_A_DOCKER_CONTAINER Yes

# Install gabriel related packages
RUN pip3 install --user --upgrade pip
RUN pip3 install --user gabriel-client\
    gabriel_protocol\
    imageio\
    opencv-python \
    protobuf \
    py-cpuinfo \
    PyQt5 \
    'pyzmq==18.1' \
    'websockets==8.0' \
    zmq \
    piexif \
    exif

# create a non-root user without password
# ARG USER_ID=1000
# RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
# RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
# USER appuser


# Will need to set the ROS environment variable by: bash /opt/ros/melodic/setup.bash before running the codes
# Or set an entrypoint like: https://github.com/osrf/docker_images/blob/master/ros/noetic/ubuntu/focal/ros-core/ros_entrypoint.sh


RUN mkdir -p /home/appuser/client_node
WORKDIR /home/appuser/client_node
COPY . /home/appuser/client_node


ENTRYPOINT ["bash", "./ros_entrypoint.sh"]

CMD [ "python3", "./crosswalk_change_filter_node.py" ]
