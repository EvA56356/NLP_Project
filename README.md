# Amazon Review Classifier
Group Member: Kenan Duan, Junpei Liao, Zhenkai Zhu
## Requirements
We run our experiments on a single RTX2080Ti GPU cloud server. And the environments required are as follows:
- Pytorch 1.8
- CUDA 11.1
- tqdm
- numpy
- sklearn

We use Amazon Reviews Datasets to train and test our models. Since the dataset is too large for us considering our limited funding, we randomly selected about 40,000+ examples from the original dataset in our experiments. The original dataset can be downloaded here: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/

## Usage
```
python -u run_dnn_cls.py --do_train --data_dir datasets --model_type cnn --word_type True --output_dir "outputs" --num_train_epochs 20 --learning_rate 2e-4 --per_gpu_train_batch_size 64 --per_gpu_eval_batch_size 128 --logging_steps 100

python -u run_ml_cls.py --do_train --data_dir datasets --model_type lr --output_dir "outputs" --n_jobs 4
```