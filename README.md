# RES-RAG

## Installation

Create a new conda environment with Python 3.10. e.g:
```shell
conda create -n res_rag python=3.10
```

Install the minimum requirements to run RES-RAG
 ```shell
pip install -r requirements.txt
 ```

## Preparing Datasets
```shell
python download_and_process.py --dataname <dataset_name>
```

## Generating Synthetic Data and Evaluating with RES-RAG
```shell
bash commands/run.sh
```