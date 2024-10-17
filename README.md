# MBPO_pendulumV1

This is a repo demonstrating the working of Model-based policy optimization (MBPO) paper by Janner et. al. on the Pendulum V1 environment in gymnasium.

Steps to run the repo:

Firstly, create a conda environment:

 conda create -n MBPO python=3.10
 
Activate the conda environment:

conda activate MBPO

pip install gymnasium[classic_control]
pip install torch
pip install pygame
pip install moviepy

pip install --upgrade gymnasium numpy torch     (to verify if they are the latest versions)


cd MBPO_pendulumV1
python3 demo.py


Training loop will begin shortly and the reward values for each episode can be seen. Video visualizations of the episode are logged in the 'logging' folder and the reward curve is also displayed.



