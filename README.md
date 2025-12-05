<h1 align="center"><span
    style="font-family: 'Courier New', Courier, monospace; font-size: 115%;"><span style="font-size: 130%;">H</span>iMoE-VLA:</span>:<br><span
    style="font-size:2.22rem;">Hierarchical Mixture-of-Experts for Generalist Vision–Language–Action Policies
    </span></h1>
<p align="center"><a href=""><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
<a href=''><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>
<a href=''><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>
<a href=''><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Data-yellow'></a>
</p>
<p align="center"><img src="assets/overview.png" width="100%"></p>
HiMoE-VLA is a new vision–language–action (VLA) framework built to effectively handle the pronounced heterogeneity in modern large-scale robotic datasets. Existing VLA models struggle with the diversity of embodiments, action spaces, sensor setups, and control frequencies found in robotic demonstrations. HiMoE-VLA introduces a Hierarchical Mixture-of-Experts (HiMoE) action module that progressively abstracts away these differences across layers, enabling unified learning of shared robot behaviors. Across both simulated and real-world platforms, HiMoE-VLA consistently outperforms prior VLA baselines and exhibits stronger generalization to new robots and action spaces.

<!-- ## 📣 News
🔥🔥 We released HiMoE-VLA,. -->

## ✅ To-Do List
- [x] Release the base model checkpoint and evaluation code
- [ ] Release the dataset
- [ ] Release the fine-tuned model checkpoints
- [ ] Release the fine-tuning code
- [ ] Release the multi-dataset sampler and pre-training code
## 🔑 Installation
When cloning this repository, remember to initialize the submodules:
```
git clone --recurse-submodules ****

# If you’ve already cloned the project, you can fetch the submodules with:
git submodule update --init --recursive
```
First, install uv using the following command:
```
wget -qO- https://astral.sh/uv/install.sh | sh
```
Once uv is installed, create the environment and install all dependencies:
```
GIT_LFS_SKIP_SMUDGE=1 uv sync
```
After the environment has been created, replace the relevant packages with our modified versions:
```
cp -r third_party/lerobot .venv/lib/python3.11/site-packages/
cp -r third_party/datasets .venv/lib/python3.11/site-packages/
cp third_party/modeling_gemma.py .venv/lib/python3.11/site-packages/transformers/models/gemma
```
## 🤖 Model Checkpoints
We provide the following pretrained models:

| Model | Description | Download |
| --- | --- | --- |
| Base model | Pretrained on OXE and Aloha | [Download]() |
| Calvin D | Finetuned on Calvin D Joint Angle | [Download]() |
| Libero 10 | Finetuned on Libero 10 | [Download]() |
| Libero Goal | Finetuned on Libero Goal | [Download]() |
| Libero Object | Finetuned on Libero Object | [Download]() |
| Libero Spatial | Finetuned on Libero Spatial | [Download]() |


## 🏋️‍♂️ Training
### Preparing data
We use the LeRoBot dataset, so you should convert your own data into the LeRobot format. We provide example scripts for reference, such as [`examples/calvin/convert_calvin_data_to_lerobot_joint.py`](examples/calvin/convert_calvin_data_to_lerobot_joint.py). You can modify it to convert your own data and run the script with:
```
uv run examples/calvin/convert_calvin_data_to_lerobot_joint.py --data_dir /path/to/your/calvin_d/data
```

### Defining your own training config
Here, we use the `calvin_d_joint` as an example. You need to update the following components:

- [`CalvinInputs` and `CalvinOutputs`](src/moevla/policies/calvin_policy.py): Define the data mapping from the CALVIN environment to the model and vice versa. Will be used for both, training and inference.
- [`LeRobotCalvinJointDataConfig`](src/moevla/training/config.py): Defines how to process raw CALVIN data from LeRobot dataset for training.
- [`DatasetConfig`](src/moevla/training/config.py): Defines dataset_name and data_mask
- [`TrainConfig`](src/moevla/training/config.py): Defines training hyperparameters, dataset_mixture, and the pretrain model.
- [`DATASET_MIXTURES`](src/moevla/training/mixtures.py): Defines the training datasets and their corresponding weights.

**Note:** The data dir is os.path.join(`assets_base_dir`, `repo_id`)

After completing the steps above, you need to compute normalization statistics for your own data. Run the script below with the name of your training config:
```
uv run scripts/compute_norm_stats.py --config-name calvin_d_joint
```
**Note:** The dataset_mixture of calvin_d_joint must contain only one dataset.

Now, you can run training using the command below:
```
accelerate launch scripts/train.py --deepspeed=scr/openpi/training/zero2.json --config=calvin_d_joint --exp-name=calvin_d_joint
```

## ⚖️ Evaluation
To effeciently manage environment, we use server and client to run evaluation. First, you can launch a model server by the command below:
```
uv run scripts/serve_policy.py --env CALVIN_D_FINETUNE --port 9000
```
You can then launch a client for quering the server. See the [CALVIN README](examples/calvin/README.md) for more details.

For Real-World Deployment, you can run with the commands below:
```
from moevla.policies import policy_config as _policy_config
from moevla.training import config as _config

# specific these parameter
train_config = ""
dataset_config = ""
checkpoint_dir = ""

policy = _policy_config.create_trained_policy(
    _config.get_training_config(train_config),    
    _config.get_dataset_config(dataset_config), 
    checkpoint_dir, 
    default_prompt=None
)

# Run inference on a dummy example.
example = {
    "observation/exterior_image_1_left": ...,
    "observation/wrist_image_left": ...,
    ...
    "prompt": "fold clothes"
}

action_chunk = policy.infer(example)["actions"]

```
Even the commands above can run infer, we still recommand using server and client for deployment.

## 🤝 Acknowledgements
We are deeply grateful for the development of [openpi](https://github.com/Physical-Intelligence/openpi/tree/main) and [LeRobot](https://github.com/huggingface/lerobot), from which our project draws extensively. We extend our sincere thanks to all contributors to these libraries for their hard work and dedication.
## 📜 Citation
If you find our work useful in your research, please consider citing our paper:
```

```