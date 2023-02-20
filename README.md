# Self-attention model for MIMO 
This repository implements an enhanced OAMP estimator using a self-attention neural network model. 

## First steps
Install all dependencies using the "environment.yml" file.
```
conda env create -f environment.yml
```

## Training
### Training a self-attention model
Set the "data_path" in the config.json file to your desired data path, e.g., "path/to/data/5dB"
```
python GIT/self-attention-mimo/scripts/train.py --model_dir path/to/model
```


### Training multiple SNR and correlation values
Set the "data_path" in the config.json file to your desired data path including placeholders for the SNR range "XdB" , e.g., "path/to/data/XdB"
```
python GIT/self-attention-mimo/scripts/train_multi.py --model_dir /path/to/model/XdB/YC
```
## Evaluation
### Evaluating multiple SNR and correlation values
Set the "data_path" in the config.json file to your desired data path including placeholders for the SNR range "XdB" , e.g., "path/to/data/XdB"

```
python  python GIT/self-attention-mimo/scripts/evaluate.py --model_dir path/to/model/XdB/YC/oampsa
```

### Evaluating the generalization over multiple SNR and correlation values for a model trained on a specifc SNR range
Set the "data_path" in the config.json file to your desired data path including placeholders for the SNR range "XdB" , e.g., "path/to/data/XdB"

```
python GIT/self-attention-mimo/scripts/evaluate_generalization.py --model_dir path/to/model/5dB/YC/oampsa
```

## Citation
You may cite this project as:
```
put bib citation here!!
```

