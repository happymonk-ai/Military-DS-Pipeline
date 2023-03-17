FROM nvcr.io/nvidia/l4t-base:r32.2.1
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl libfreetype6-dev libhdf5-dev libpng-dev libzmq3-dev pkg-config python3-dev python3-numpy python3-scipy python3-sklearn python3-matplotlib python3-pandas rsync unzip zlib1g-dev zip libjpeg8-dev hdf5-tools libhdf5-serial-dev python3-pip python3-setuptools
RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa && apt update
RUN apt-get update && apt-get -y --no-install-recommends install \
    sudo \
    vim \
    wget \
    build-essential \
    python3.8 \
    python3.8-dev \
    python3.8-venv \
    python-dev \
    python3-dev
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 0
RUN apt-get purge python3-pip -y && apt-get update && apt-get -y install python3-pip -y && apt-get install --reinstall python3-distutils -y && pip3 install --upgrade pip
RUN apt-get update && apt install --reinstall python3-apt -y
RUN pip3 install --upgrade pip
RUN apt-get update && apt -y --no-install-recommends install \
    git \
    cmake \
    autoconf \
    automake \
    libtool \
    gstreamer-1.0 \
    gstreamer1.0-dev \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-doc \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    gstreamer1.0-alsa \
    gstreamer1.0-gl \
    gstreamer1.0-gtk3 \
    gstreamer1.0-qt5 \
    gstreamer1.0-pulseaudio \
    python-gst-1.0 \
    libgirepository1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libcairo2-dev \
    gir1.2-gstreamer-1.0 \
    python-gi-dev \
    python3-gi \	
    python3-setuptools \
    ffmpeg \
    python3-gi-cairo 
RUN apt-get clean && \
        rm -rf /var/lib/apt/lists/*
RUN pip3 install -U pip -v
RUN pip3 --no-cache-dir install -U -v jupyter grpcio absl-py py-cpuinfo psutil portpicker six mock requests gast h5py astor termcolor protobuf keras keras-applications keras-preprocessing wrapt google-pasta
RUN pip3 install tensorflow
#RUN curl -L https://nvidia.box.com/shared/static/phqe92v26cbhqjohwtvxorrwnmrnfx1o.whl > /tmp/torch-1.3.0-cp36-cp36m-linux_aarch64.whl && pip3 --no-cache-dir -v install /tmp/torch-1.3.0-cp36-cp36m-linux_aarch64.whl && rm  /tmp/torch-1.3.0-cp36-cp36m-linux_aarch64.whl
RUN apt-get update && apt install build-essential autoconf libtool python-opengl python-pil python-pyrex python-pyside.qtopengl idle-python2.7 qt4-dev-tools qt4-designer libqtgui4 libqtcore4 libqt4-xml libqt4-test libqt4-script libqt4-network libqt4-dbus python-qt4 python-qt4-gl libgle3 python-dev libssl-dev -y
RUN apt install libjpeg-dev zlib1g-dev -y
RUN apt-get update && apt install build-essential libdbus-glib-1-dev -y
RUN apt-get update && apt install redis libicu-dev python3-distutils-extra libudev-dev libsystemd-dev libxml2-dev libcups2-dev libxmlsec1-dev libavformat-dev libavdevice-dev -y
COPY . .
RUN python3.8 -m pip install --ignore-installed PyGObject
RUN apt-get update && \
    apt-get install -y locales && \
    sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen
ENV LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 LANGUAGE=en_US.UTF-8
RUN apt-get update && apt-get install -y python3-gi
RUN --mount=type=cache,target=/root/.cache/pip pip3 install -r requirements.txt
RUN python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
EXPOSE 8888 6006
CMD ["python3", "military_Pipeline.py"]
