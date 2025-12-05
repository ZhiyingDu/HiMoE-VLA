# CALVIN Benchmark

Terminal window 1:

```
# Install calvin environment according the official document
cd third_party/calvin
conda create -n calvin_venv python=3.8
conda activate calvin_venv
bash install.sh

# install necessary package
pip install -e packages/openpi-client
pip install tyro
pip install numpy==1.23.5

# if you meet ModuleNotFoundError: No module named 'calvin_env'. Please add init file in the "calvin_env"

cp examples/calvin/robot.py third_party/calvin/calvin_env/calvin_env/robot
cp examples/calvin/play_table_env.py third_party/calvin/calvin_env/calvin_env/envs

export PYTHONPATH=$PYTHONPATH:$PWD/third_party/calvin

# Run the simulation
python examples/calvin/main.py --port 9000

# Note: you must specific the args.calvin_dataset_path(line103)
```

Terminal window 2:

```bash
# Run the server
uv run script/server_policy.py --env CALVIN_D_FINETUNE --port 9000
```