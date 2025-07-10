# Getting started

1. Install the environment
```bash
conda env create -f conda.yaml
uv pip install -r requirements.txt
```

# Preprocessing

From the root directory of the repository run:
```bash
python scripts/preprocess_dataset.py --source_data_dir data/raw/set_24 --output_data_dir data/processed/set_24
python scripts/compute_resolution.py data/processed/set_24/resolutions.json
```


# Training

