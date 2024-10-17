# DeliGrasp Policy Learning and Evaluation

This repository contains code for training and evaluating policies on [grasp force feedback datasets](https://justaddforce.github.io) dataset. This codebase and following README is adapted from [DROID Policy Learning](https://github.com/droid-dataset/droid_policy_learning), which itself is built as a fork of [`robomimic`](https://robomimic.github.io/), a popular repository for imitation learning algorithm development 

<!-- [**[Homepage]**](https://droid-dataset.github.io) &ensp; [**[Documentation]**](https://droid-dataset.github.io/droid) &ensp; [**[Paper]**](https://arxiv.org/abs/2403.12945) &ensp; [**[Dataset Visualizer]**](https://droid-dataset.github.io/dataset.html). -->
[**[Homepage]**](https://justaddforce.github.io)

-------
## Installation
Create a python3 conda environment (tested with Python 3.10) and run the following:

1. Create python 3.10 conda environment: `conda create --name droid_policy_learning_env python=3.10`
2. Activate the conda environment: `conda activate droid_policy_learning_env`
3. Install [octo](https://github.com/octo-models/octo/tree/main), pinned at commit `85b83fc19657ab407a7f56558a5384ae56fe453b` (used for data loading)
4. In `octo`, `git reset --hard 85b83fc` before pip installing.
5. Back here, run `pip install -e .` in `droid_policy_learning`.

With this you are all set up for training policies.

-------
## Training
To train policies, update `DATA_PATH`, `EXP_LOG_PATH`, and `EXP_NAMES` in `robomimic/scripts/config_gen/deligrasp_runs_language_conditioned_rlds.py` and then run:

`python robomimic/scripts/config_gen/deligrasp_runs_language_conditioned_rlds.py --wandb_proj_name <WANDB_PROJ_NAME>`

This will generate a python command that can be run to launch training. You can also update other training parameters within `robomimic/scripts/config_gen/droid_runs_language_conditioned_rlds.py`. Please see the `robomimic` documentation for more information on how `robomimic` configs are defined. The three
most important parameters in this file are:

- `DATA_PATH`: This is the directory in which all RLDS datasets were prepared.
- `EXP_LOG_PATH`: This is the path at which experimental data (eg. policy checkpoints) will be stored.
- `EXP_NAMES`: This defines the name of each experiment (as will be logged in `wandb`), the RLDS datasets corresponding to that experiment, and the desired sample weights between those datasets. See `robomimic/scripts/config_gen/droid_runs_language_conditioned_rlds.py` for a template on how this should be formatted.

During training, we use a [_shuffle buffer_](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle) to ensure that training samples are properly randomized. It is important to use a large enough shuffle buffer size.
The default `shuffle_buffer_size` is set to `500000`, but you may need to reduce this based on your RAM availability. For best results, we recommend using `shuffle_buffer_size >= 100000` if possible. All polices were trained on a single NVIDIA A100 GPU.

To specify your information for Weights and Biases logging, make sure to update the `WANDB_ENTITY` and `WANDB_API_KEY` values in `robomimic/macros.py`.

We also provide a stand-alone example to load data from DROID [here](examples/droid_dataloader.py).

-------
## Code Structure

|                           | File                                                    | Description                                                                   |
|---------------------------|---------------------------------------------------------|-------------------------------------------------------------------------------|
| Hyperparameters           | [droid_runs_language_conditioned_rlds.py](robomimic/scripts/config_gen/droid_runs_language_conditioned_rlds.py)     | Generates a config based on defined hyperparameters  |
| Training Loop             | [train.py](robomimic/scripts/train.py)                  | Main training script.                                                         |
| Datasets                  | [dataset.py](https://github.com/octo-models/octo/blob/main/octo/data/dataset.py)                      | Functions for creating datasets and computing dataset statistics,             |
| RLDS Data Processing      | [rlds_utils.py](robomimic/utils/rlds_utils.py)    | Processing to convert RLDS dataset into dataset compatible for DROID training                      |
| General Algorithm Class   | [algo.py](robomimic/algo/algo.py)             | Defines a high level template for all algorithms (eg. diffusion policy) to extend           |
| Diffusion Policy          | [diffusion_policy.py](robomimic/algo/diffusion_policy.py)    | Implementation of diffusion policy |
| Observation Processing    | [obs_nets.py](robomimic/models/obs_nets.py)    | General observation pre-processing/encoding |
| Visualization             | [vis_utils.py](robomimic/utils/vis_utils.py) | Utilities for generating trajectory visualizations                      |

-------

## Evaluating Trained Policies
WIP. Evaluation is all local on real hardware. 