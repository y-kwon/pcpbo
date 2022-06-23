FROM nvcr.io/nvidia/pytorch:21.09-py3

# for avoidance of 'tzdata' configuring
ENV DEBIAN_FRONTEND=noninteractive
# for GUI
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# Coppelia Sim
RUN apt-get update && apt-get install -y --no-install-recommends qt5*
COPY ./simulators/assets/CoppeliaSim_Edu_V4_3_0_Ubuntu20_04 /workspace/CoppeliaSim_Edu_V4_3_0_Ubuntu20_04
RUN mkdir /run/user/0
ENV XDG_RUNTIME_DIR=/run/user/0
ENV COPPELIASIM_ROOT=/workspace/CoppeliaSim_Edu_V4_3_0_Ubuntu20_04
ENV LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
ENV QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
RUN echo 'export COPPELIASIM_ROOT=/workspace/CoppeliaSim_Edu_V4_3_0_Ubuntu20_04' >> /root/.bashrc
RUN echo 'export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH' >> /root/.bashrc
RUN echo 'export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT' >> /root/.bashrc
RUN echo "alias coppeliasim='/workspace/CoppeliaSim_Edu_V4_3_0_Ubuntu20_04/coppeliaSim.sh'" >> /root/.bashrc
# PyRep
RUN apt-get update && apt-get install -y xvfb
RUN git clone https://github.com/stepjam/PyRep.git && cd PyRep &&\
    pip install -r requirements.txt && pip install .

# build isaac gym
RUN apt-get update && apt-get install -y --no-install-recommends  \
    libxcursor-dev libxrandr-dev libxinerama-dev libxi-dev mesa-common-dev zip unzip make gcc-8 g++-8 wget  \
    vulkan-utils mesa-vulkan-drivers doxygen graphviz fonts-roboto python3-sphinx pigz git libegl1 git-lfs

# Force gcc 8 to avoid CUDA 10 build issues on newer base OS
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8

# WAR for eglReleaseThread shutdown crash in libEGL_mesa.so.0 (ensure it's never detected/loaded)
# Can't remove package libegl-mesa0 directly (because of libegl1 which we need)
RUN rm /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0 /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0.0.0 /usr/share/glvnd/egl_vendor.d/50_mesa.json

COPY isaacgym/docker/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY isaacgym/docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

WORKDIR /opt/isaacgym
# copy gym repo to docker
COPY isaacgym .

# install gym modules
RUN cd python && pip install -q -e .
RUN cd python/rlgpu/rl-pytorch && pip install -q -e .
#RUN pip uninstall -q -y numpy numpy-quaternion && pip install -q numpy==1.21.2 numpy-quaternion==2021.8.30.10.33.1
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all

# for this repo
RUN conda install -c conda-forge quaternion
RUN pip install dotmap seaborn

# For uid, gid
RUN apt-get update -qq && apt-get -y install gosu
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /opt/project
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
