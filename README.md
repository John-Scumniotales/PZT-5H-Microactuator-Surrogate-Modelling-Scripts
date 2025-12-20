# Data-Driven Surrogate Modeling for Tileable PZT-5H Piezoelectric Microactuators

<img width="1200" height="900" alt="image" src="https://github.com/user-attachments/assets/d17eac18-e28a-4ccc-8b57-c6adbcc714ac" />

This repo contains the **training script + dataset** used for the paper **"Data-Driven Surrogate Modeling for Tileable PZT-5H Piezoelectric Microactuators"**.
A compact GRU encoder-decoder style sequence to sequence surrogate that predicts future displacement futures given short histories of an actuators electromechanical history. 
This enables fast, accurate, and parralel inference of microactuator behaviour, a stark contrast to its FEM counterpart.

## Quickstart

### 1) Create the conda enviroment ###
```bash
conda create -n <env_name> --file env_spec.txt
conda activate <env_name>
```
**CUDA:** training on GPU requires **CUDA 12.1** available on your system.
### 2) Model Training
```bash
python train.py
```
Artifacts are then written to the "/Model" directory.
 


