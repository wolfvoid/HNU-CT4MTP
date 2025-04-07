# HNU-CT4MTP
HNU - Casual Transformer for multivariable time prediction (as the model part for National College Student Innovation Training Program 2023-2025 )

## Environment set

```bash
# if you use conda environment
conda create --name <env> python=3.12.9
conda activate <env>
pip install -r requirements.txt

# if you use pip environments
python -m venv <env>
# for Windows
<env>\Scripts\activate
# for macOS/Linux
source <env>/bin/activate
pip install -r requirements.txt
```



## How to recurrent

download dataset

```bash
export HF_ENDPOINT=https://hf-mirror.com	# for Chinese users
pip install -U huggingface_hub
huggingface-cli download --repo-type dataset wolfvoid/5MTP-datasets --local-dir ./5MTP-datasets
```

edit your target task in pipeline.py.

```bash
model_name = "gbrt"         # choose from ( casual || arima || gbrt )
dataset_name = "traffic"	# choose from (electricity || exchange_rate || PSM || traffic || weather)
seq_len, pred_len = 64, 64	# choose from 16,16 || 32,32 || 64,64 or others
```

now start to run

```bash
CUDA_VISIBLE_DEVICES=0 python pipeline.py
```

you will get answer in your terminal.

if you want your ckpt saved, set `save_model=True`

Any hyperparameter can be changed in `def run(model_name, dataset_name, seq_len, pred_len)`,. If you want to edit transformer frames, edit `runCasualTransformer.py` also `CasualTransformer.py`.



## How to get pretrained ckpt

we public our best ckpts on 5 datasets, you can get it following belows.

```bash
huggingface-cli download --repo-type model wolfvoid/CasualTransformer-ckpt-for-5-datasets --local-dir ./ckpts
```



## Other Dataset

Dataset concerning Spatiotemporal Prediction can be found as follows. Dataset are divided by feature or by places. You can use it on your own purpose.

[wolfvoid/SpatiotemporalPredictionHuNan](https://huggingface.co/datasets/wolfvoid/SpatiotemporalPredictionHuNan)

## Others

If you meet with path errors, try adjust your paths to fit your local path.

Some other models are also available on `model`. You can try for yourself.

Any other questions please raise issues.
