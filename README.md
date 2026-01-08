## Structured State Space Models for Prior-Fitted Data Networks

Repository to reproduce the result of the paper **State-Space Models for Tabular Prior-Data Fitted Networks** available on [Arxiv](https://arxiv.org/pdf/2510.14573).

This repository is built on the code of [TabPFN](https://github.com/automl/TabPFN).

Other Dependencies used:

- Mamba (1 and 2): https://github.com/state-spaces/mamba
- Hydra: https://github.com/goombalab/hydra

## Getting Started

A Dockerfile is provided with a setup.sh file in the root repository to build an Ubuntu-based Docker container with all dependencies. This container exceeds 5GB, so the whole setup takes time.

To launch the experiments, we used the following combination of dependencies:

- python == 3.8.10
- CUDA == 11.8
- torch == 2.2.0

Other versions from external dependencies may also be supported but were not tested properly.

## Files for Training and Evaluation

For training a custom Prior-Data fitted Network, it is recommended to choose either a Transformer, Mamba, or Hydra as a Backbone. Each model can be trained with the train_custom_model.py script.

**Evaluation**  

We used the following three scripts for evaluation:

- evaluation_script.py
- permutation_eval_script.py
- whole_output_evaluation.py

For training, just use train_custom_model.py

## License

Most parts of this work are licensed under the MIT license if not stated otherwise.  

Files in the tabpfn/ folder except those regarding mamba or hydra are under the license of [TabPFN](https://github.com/automl/TabPFN)

## Citation
```
@article{ssmsforpfns2025,
  title={State-Space Models for Tabular Prior-Data Fitted Networks},
  author={Koch, Felix and Wever, Marcel and Raisch, Fabian and Tischler, Benjamin},
  journal={1st ICML Workshop on Foundation Models for Structured Data},
  year={2025}
}
```
