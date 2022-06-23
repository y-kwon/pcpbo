# Physically Consistent Preferential Bayesian Optimization
## Description
This repository is an implementation of our work, "**[Physically Consistent Preferential Bayesian Optimization
for Food Arrangement](https://y-kwon.github.io/pcpbo/)**". It contains the deep-fried shrimp arrangement task used in 
our simulation experiments.

## Requirements
- Ubuntu 18.04 or 20.04
- Nvidia driver  > 470
- Docker

## Installation

1. Download and defrost [IsaacGym](https://developer.nvidia.com/isaac-gym) and place it in `./isaacgym`.
2. Download and defrost [CoppeliaSim](https://www.coppeliarobotics.com/downloads) and place it in `./simulators/assets`.
3. Setup with Docker

```shell
$ sudo docker build -t pcpbo .
```

## Usage

1. (Optional) Generate X virtual framebuffer for headless

- Install Xvfb

```shell
$ sudo apt-get update && sudo apt-get install xvfb
```

- Run Xvfb

```shell
$ Xvfb :2 -screen 0 1024x768x24 &
$ export DISPLAY=:2
```

2. Run docker

```shell
$ docker run --rm -it --gpus all -e DISPLAY=$DISPLAY\
             -e LOCAL_UID=$(id -u $USER) -e LOCAL_GID=$(id -g $USER)\ 
             -v /tmp/.X11-unix:/tmp/.X11-unix -v $PWD/:/opt/project\
              pcpbo
```

#### Estimate a preferred arrangement with Physically Consistent Preferential Bayesian Optimization (PCPBO)

```shell
(docker)$ python run_pcpbo.py --task deep-fried_shrimp --pref_weight 0.2 0.8
```

#### Generate an arrangement corresponding to a specific weight using Cross Entropy Method (CEM)

```shell
(docker)$ python run_cem.py --task deep-fried_shrimp --pref_weight 0.2 0.8
```

