# TabGEN-ICL

## Installation

Create a new conda environment with Python 3.10. e.g:
```shell
conda create -n tabgen python=3.10
```

Install the minimum requirements to run TabGEN-ICL
 ```shell
pip install -r requirements.txt
 ```

## Preparing Datasets
```shell
python download_and_process.py --dataname <dataset_name>
```

## Generating Synthetic Data and Evaluating with TabGEN-ICL
```shell
bash commands/run.sh
```
