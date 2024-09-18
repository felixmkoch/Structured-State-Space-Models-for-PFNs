## Structured State Space Models for Prior-Fitted Data Networks

Source code to reproduce the results from the Master's Thesis _Structured State Space Models for Prior-Fitted Data Networks_.

This repository is built on the code of [TabPFN](https://github.com/automl/TabPFN). Many files and structures originate from their repository and are adapted for Structured State-Space Models.

Other Dependencies used:

- S4: https://github.com/state-spaces/s4
- Mamba (1 and 2): https://github.com/state-spaces/mamba
- Hydra: https://github.com/goombalab/hydra

## Getting Started

To launch experiments, the following requirements need to be fulfilled:

1. CUDA >= 11.8
2. Python 3.10
3. Dependencies from the requirements.txt
4. PyTorch >= 2.0.0 with CUDA support

Other versions from external dependencies may also be supported but were not tested properly.

A Dockerfile is provided with a setup.sh file in the root repository to build an Ubuntu-based Docker container with all dependencies. This container exceeds 5GB, so the whole setup takes time.

## Files for Training and Evaluation

For training a custom Prior-Data fitted Network, it is recommended to choose either a Transformer, Mamba, or Hydra as a Backbone. Each model can be trained with the train_custom_model.py script.
S4 and Mamba2 were also tested but did not work properly and therefore have a separate Python script to launch.

**Evaluation**
In the thesis, several evaluation steps were taken to better grasp the behavior of SSMs for PFNs on tabular data. Some have separate scripts to it:

- bptt_eval_script: Investigate the input sequence extrapolation performance (Chapter 5.3)
- evaluation_script: Evaluation for all the other result chapters.

## Results

The results_csvs folder contains output files in a CSV format that lists the output of evaluations conducted on different models. Data from tables can be found there.

Note: Most but not all results from the thesis are in this folder because some were observed via console output.

