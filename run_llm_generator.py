import sys
import os
import traceback
import time
sys.path.append(os.path.abspath('..'))

from src_llm.utils import *
from src_llm.llm_gen import *
from src_llm.dataset_llm_templates import *
from src_llm.data_loader import *

import pandas as pd
from copy import deepcopy
import pickle
import argparse

parser = argparse.ArgumentParser(description='LLM for table generation')

parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
parser.add_argument('--model', type=str, default='gpt-4o-mini', help='Model name.')
parser.add_argument('--num_synthetic', type=int, default=2000, help='Number of synthetic samples to generate.')
parser.add_argument('--n_processes', type=int, default=1, help='Number of processes to run in parallel.') 
parser.add_argument('--chunk_size', type=int, default=100, help='size of chunk data sampled for each prompt (for each class).')
parser.add_argument('--exp_name', type=str, default='exp_default_name', help='Experiment name.')
parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for the LLM.')
parser.add_argument('--output_format', type=str, default='json', help='Output format.')

args = parser.parse_args()

ylabel_map= {
            "adult": "income",
            "beijing": "pm2.5",
            "default": "default payment next month",
            "magic": "class",
            "news": " shares",
            "shoppers": "Revenue",
            "california": "ocean_proximity",
            }

if __name__ == '__main__':
    print(args)
    
    openai_api_details = {
        "openai_api_key": "add openai key here"
    }

    model_short_name = args.model 

    if model_short_name == 'gpt-4o-mini':
        model = "gpt-4o-mini"
    elif model_short_name == 'gpt-4o':
        model = 'gpt-4o-2024-08-06'
    else:
        print('Model not supported, please choose from gpt-4o-mini or mixtral-8x22b')

    chunk_size = args.chunk_size 
    n_synthetic = args.num_synthetic 
    n_processes = args.n_processes 
    exp_name = args.exp_name
    output_format = args.output_format
    dataset = args.dataname
    temperature = args.temperature

    try:
        #sleep between runs from a rate limit perspective
        time.sleep(10)

        df_train = pd.read_csv(f'synthetic/{dataset}/real.csv')

        df_feat_train = df_train.drop(columns=[ylabel_map[dataset]])
        df_label_train = df_train[ylabel_map[dataset]]
        
        X_train, _, y_train, _ = sample_and_split(df_feat_train, df_label_train, ns='all')  

        X_train_orig = deepcopy(X_train)
        y_train_orig = deepcopy(y_train)

        results = {}
        
        # get prompt
        if output_format == 'markdown':
            generator_template, format_instructions, example_df = markdown_templates_RES_RAG(X_train_orig, y_train_orig, dataset=dataset, role='generator')
        elif output_format == 'json':
            generator_template = json_templates_RES_RAG(role='generator')
            format_instructions = '{"JSON":[{col1:value1,col2:value2, ...}, {col1:value1,col2:value2, ...}, ...]}'
        
        retries = 4  
        while retries > 0:
            try:
                df_orig = pd.concat([X_train_orig, y_train_orig], axis=1)

                json_response_schema = create_json_model(df_orig, dataname=dataset)

                example_df = df_orig.sample(frac=1).reset_index(drop=True)
                ic_samples = min(len(X_train), chunk_size)
                
                print(f'Running {dataset}, {model} --- {n_processes}')
                df_llm = llm_gen(dataset,
                                 generator_template, 
                                 openai_key=openai_api_details['openai_api_key'],
                                 n_samples=n_synthetic,
                                 example_df=example_df,
                                 output_format=output_format,
                                 response_schema=json_response_schema,
                                 format_instructions=format_instructions, 
                                 model=model, 
                                 temperature=temperature,
                                 n_processes=n_processes,
                                 ic_samples=ic_samples, 
                                 )
                print('generated shape', df_llm.shape)
                break  # if successful, break out of the loop
            except Exception as e:
                print(f"Error: {e}. Retrying with reduced n_processes...")
                print(traceback.format_exc())
                n_processes = int(n_processes/2)
                retries -= 1
                if n_processes < 1:
                    print("Error: Minimum n_processes reached. Exiting...")
                    break
        try:
            tmp_df = df_llm.astype(example_df.dtypes)
            df_llm = tmp_df
        except:
            pass
        
        ylabel =  ylabel_map[dataset]

        X_train_llm = df_llm.drop(columns=[ylabel])
        y_train_llm = df_llm[ylabel]

        results['llm'] = {"X": X_train_llm, 'y': y_train_llm, 'whole': df_llm}

        with open(f'save_dfs/row_df_llm_{dataset}_{model_short_name}_{output_format}_{exp_name}.pickle', 'wb') as f:
                pickle.dump(results, f)

    except Exception as e:
        print(traceback.format_exc())
        print(e)
            
    # Post-processing
    with open(f'save_dfs/row_df_llm_{dataset}_{model_short_name}_{output_format}_{exp_name}.pickle', 'rb') as f:
        results = pickle.load(f)
        
        for col in results['llm']['whole'].columns:
            results['llm']['whole'][col] = results['llm']['whole'][col].apply(lambda x: x.strip() if isinstance(x, str) else x)

            if df_train[col].dtype == 'object' and df_train[col].str.startswith(' ').any():
                results['llm']['whole'][col] = results['llm']['whole'][col].apply(lambda x: ' ' + x if isinstance(x, str) else x)
            
            results['llm']['whole'] = results['llm']['whole'][results['llm']['whole'][col] != ' null']
        
        for col in df_train.columns:
            if df_train[col].dtype == 'object': 
                allowed_values = df_train[col].unique()
                results['llm']['whole'] = results['llm']['whole'][results['llm']['whole'][col].isin(allowed_values)]
            if df_train[col].dtype == 'int64':
                results['llm']['whole'] = results['llm']['whole'][results['llm']['whole'][col].apply(lambda x: isinstance(x, int))]
            if df_train[col].dtype == 'float64':
                results['llm']['whole'] = results['llm']['whole'][results['llm']['whole'][col].apply(lambda x: isinstance(x, float))]

    results['llm']['whole'].to_csv(f'synthetic/{dataset}/{model_short_name}_{n_synthetic}_temp{temperature}_{exp_name}.csv', index=False)
    print('saved at', f'synthetic/{dataset}/{model_short_name}_{n_synthetic}_temp{temperature}_{exp_name}.csv')

