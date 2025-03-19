# AuSeM
Automating Sensemaking Measurements

## Run our models on student data

1. Create the required python environment (we recommend mamba for package management) with `mamba env create -f environment.yaml`

2. Activate the environment `mamba activate ausem`

3. Run a model with `python src/train_loop.py --problem_config config/problem_config.yaml --hyper_config config/best_bow.yaml` 