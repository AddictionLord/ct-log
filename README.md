# Getting started

1. Install the environment
```bash
conda env create -f conda.yaml
uv pip install -r requirements.txt
```

# Dataset Preprocessing

From the root directory of the repository run:
```bash
python scripts/preprocess_dataset.py --source_data_dir data/raw/set_24 --output_data_dir data/processed/set_24
python scripts/compute_resolution.py data/processed/set_24/resolutions.json
```


# Training

```bash
python src/train_dino.py --config src/configs/train_dino.yaml
```

# Plans:

## Main:
1. ~~Setup metrics computation for training loop~~
2. ~~Accumulate the metrics from all the loops~~
3. Add the regression head for pith prediction
4. ~~Ensure metrics logging to MlFlow~~
5. Create a new dataset based on lates changes in the document (less classes)
7. Use weights for classes in loss computations
   1.  Research: Online vs Offline class loss weight computation
   2.  Implementation of the chosen approach
8. Train the model

## Optional:
1. Create an interface for the prediction model
2. Wrap the Dino model with heads to single class for reusability
3. Create an interface for loss functions, make it configurable
