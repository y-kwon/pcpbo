# Physically Consistent Preferential Bayesian Optimization
## Description
____
This repository is an implementation of our work, "**[Physically Consistent Preferential Bayesian Optimization
for Food Arrangement](https://y-kwon.github.io/pcpbo/)**". It contains the deep-fried shrimp arrangement task used in 
our simulation experiments.

The code for this repository will be released on June 23, 2022, at ~noon~ 18:00 (UTC).

## Requirements
___
- Ubuntu 18.04 or 20.04
- Nvidia driver  > 470
- Docker

## Installation
___
1. Download and defrost [IsaacGym](https://developer.nvidia.com/isaac-gym) and place it in `./isaacgym`.
2. Download and defrost [CoppeliaSim](https://www.coppeliarobotics.com/downloads) and place it in `./simulators/assets`.
3. Setup with Docker
```shell
# docker build -t pcpbo .
```

##  Usage