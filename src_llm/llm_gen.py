import re
import json
import pandas as pd
import numpy as np
from copy import deepcopy
import openai
import random
import time
import traceback
from openai import OpenAI
from langchain.prompts import ChatPromptTemplate
from src_llm.utils import get_indices, compute_distance

ylabel_map= {
            "adult": "income",
            "default": "default payment next month",
            "magic": "class",
            "shoppers": "Revenue",
            "california": "ocean_proximity",
            }

def llm_gen(
    dataname,
    generator_template,
    reflector_template,
    openai_key,
    n_samples,
    example_df,
    output_format,
    response_schema=None,
    format_instructions=None, 
    temperature=1.0,
    max_tokens=15000,
    model="gpt4_20230815",
    n_processes=1,
    ic_samples=20,
):
    
    client = OpenAI(api_key=openai_key)
    init = True
    data_count = 0

    for prompt_idx in range(3000):
        print("Prompt iteration = ", prompt_idx)
        df_list = []
        try:
            y_label = ylabel_map[dataname]
            df_label = example_df[y_label] 

            header = example_df.columns.tolist()
            sample_strategy = 'neo'

            if prompt_idx==0 or sample_strategy == 'uniform':
                num_classes = len(df_label.unique())
                print('num of uniform samples:', ic_samples*num_classes)
                sampled_indices_for_generator = example_df.sample(n=ic_samples*num_classes, replace=True).index
                example_df_generator = example_df.loc[sampled_indices_for_generator].reset_index(drop=True)

                if reflector_template is not None:  
                    sampled_indices_for_reflector = example_df.sample(n=ic_samples*num_classes, replace=True).index
                    example_df_reflector = example_df.loc[sampled_indices_for_reflector].reset_index(drop=True)
            
            elif sample_strategy == 'neo':
                #randomly pick a column from example_df 
                header = example_df.columns.tolist()
                random_col = random.choice(header)

                dtype = example_df[random_col].dtype
                
                sampled_indices_collection = get_indices(example_df, random_col, dtype, max_num=500)

                distances = compute_distance(example_df, sampled_indices_collection, df_llm, prompt_idx=prompt_idx) #df_tmp

                #print("distances = ", distances)

                best_distance_idx = np.argmin(distances)

                best_sampled_indices = sampled_indices_collection[best_distance_idx]

                print('num of neo samples:', len(best_sampled_indices))

                example_df_generator = example_df.loc[best_sampled_indices].reset_index(drop=True)

                if reflector_template is not None:
                    example_df_reflector = example_df.loc[best_sampled_indices].reset_index(drop=True)
            
            data_count += example_df_generator.shape[0]

            if output_format == 'markdown': 
                small_data_gen = str(example_df_generator.to_dict(orient="records"))
                if reflector_template is not None:
                    small_data_reflect = str(example_df_reflector.to_dict(orient="records"))
            else: 
                small_data_gen = example_df_generator.to_json(orient="records")
                if reflector_template is not None:
                    small_data_reflect = example_df_reflector.to_json(orient="records")
                
                # TODO: for debug
                # small_data_corr = small_data_gen
                # small_data_gen = str(header)

            prompt_gen = ChatPromptTemplate.from_template(template=generator_template)
            
            if output_format == 'json': 
                messages_gen = prompt_gen.format_messages(data=small_data_gen, format_instructions=format_instructions)
            # else: 
            #     messages_gen = prompt_gen.format_messages(data=small_data_gen, dataset_description=data_info, format_instructions=format_instructions)
                
            
            prompt_gen = "".join(messages_gen[0].content.split("\n")[1:])
            messages_gen = [
                {
                    "role": "system",
                    "content": "You are a synthetic data generator which can produce data that mirrors the given examples in both causal structure and feature-label distributions, while ensuring a high degree of diversity in the generated samples.",
                },
                {
                    "role": "user", 
                        "content": f"{prompt_gen}"
                }
            ]

            print("Generating data...")
            if output_format == 'json': 
                response = client.beta.chat.completions.parse(
                    model=model,
                    messages=messages_gen,
                    temperature=temperature,
                    n=n_processes,
                    frequency_penalty=0,
                    presence_penalty=0,
                    max_tokens=max_tokens,
                    stop=None,
                    response_format=response_schema,
                )
        
            # ======================================================================================
            for idx in range(n_processes):
                #print("process idx = ", idx)

                try:
                    data_gen = response.choices[idx].message.content 
                    #print("generated data:", data_gen)
                    
                    # print("Reflecting data...")
                    # prompt_reflector = ChatPromptTemplate.from_template(template=reflector_template)
                    # if output_format == 'json':
                    #     messages_reflector = prompt_reflector.format_messages(generated_data=data_gen, real_data=small_data_corr, format_instructions=format_instructions)
                    # else:
                    #     messages_reflector = prompt_reflector.format_messages(generated_data=data_gen, real_data=small_data_corr, format_instructions=format_instructions)
                    
                    # prompt_reflector = "".join(messages_reflector[0].content.split("\n")[1:])

                    # messages_reflector = [
                    #     {
                    #         "role": "system",
                    #         "content": "You are a synthetic data reflector.",
                    #     },
                    #     {"role": "user", "content": f"{prompt_reflector}"},
                    # ]
                    # if output_format == 'json':
                    #     response_reflector = client.chat.completions.create(
                    #         model=model,
                    #         messages=messages_reflector,
                    #         temperature=temperature, 
                    #         n=1,
                    #         frequency_penalty=0,
                    #         presence_penalty=0,
                    #         max_tokens=max_tokens,
                    #         stop=None,
                    #     )
                    # else:
                    #     response_reflector = client.chat.completions.create(
                    #         model=model,
                    #         messages=messages_reflector,
                    #         temperature=temperature,
                    #         n=1,
                    #         frequency_penalty=0,
                    #         presence_penalty=0,
                    #         max_tokens=max_tokens,
                    #         stop=None,
                    #     )

                    # feedback = response_reflector.choices[0].message.content 
                    # print('Reflector feedback:', feedback)
                     
                    # # ================================================================
                    # print("Refining data...")
                    # prompt_refiner = ChatPromptTemplate.from_template(template=refiner_template)
                    # if output_format == 'json':
                    #     messages_refiner = prompt_refiner.format_messages(generated_data=data_gen, feedback=feedback, format_instructions=format_instructions)
                    # else:
                    #     messages_refiner = prompt_refiner.format_messages(generated_data=data_gen, feedback=feedback, format_instructions=format_instructions)
                    
                    # prompt_refiner = "".join(messages_refiner[0].content.split("\n")[1:])

                    # messages_refiner = [
                    #     {
                    #         "role": "system",
                    #         "content": "You are a synthetic data refiner.",
                    #     },
                    #     {"role": "user", "content": f"{prompt_refiner}"},
                    # ]
                    # if output_format == 'json':
                    #     response_refiner = client.beta.chat.completions.parse(
                    #         model=model,
                    #         messages=messages_refiner,
                    #         temperature=temperature, 
                    #         n=1,
                    #         frequency_penalty=0,
                    #         presence_penalty=0,
                    #         max_tokens=max_tokens,
                    #         stop=None,
                    #         response_format=response_schema,
                    #     )
                    # else:
                    #     response_refiner = client.chat.completions.create(
                    #         model=model,
                    #         messages=messages_refiner,
                    #         temperature=temperature,
                    #         n=1,
                    #         frequency_penalty=0,
                    #         presence_penalty=0,
                    #         max_tokens=max_tokens,
                    #         stop=None,
                    #     )

                    # data_refined = response_refiner.choices[0].message.content 
                    # print('Refined data:', data_refined)

                    # TODO: for debug
                    data_refined = data_gen
                    
                    # ======================================================================================
                    
                    if output_format == 'markdown':
                        # Extract dict-like strings using regular expressions
                        dict_strings = re.findall(r"\{[^{}]*\}", data_refined)
                        # Convert dict-like strings to actual dictionaries
                        dicts = [json.loads(ds) for ds in dict_strings]
                    else:
                        dicts = json.loads(data_refined) # json to dicts
                        dicts = dicts["JSON"] # extract data from dicts

                except Exception as e:
                    print("Error in response processing:", e)
                    raise

                df_tmp = deepcopy(pd.DataFrame(dicts))
                df_tmp = df_tmp[
                    ~df_tmp.apply(
                        lambda row: any(
                            [
                                isinstance(cell, str)
                                and cell
                                in ["integer", "float", "numeric", "categorical"]
                                for cell in row
                            ]
                        ),
                        axis=1,
                    )
                ]
                df_list.append(df_tmp)

            df_tmp = df_list[0]
            for df_check in df_list[1:]:
                df_tmp = pd.concat([df_tmp, df_check], ignore_index=True)

            if init == True:
                df_llm = deepcopy(df_tmp)
                init = False
            else:
                df_llm = pd.concat([df_llm, df_tmp], ignore_index=True)

            n_gen = df_llm.shape[0]
            print('Prompt idx:', prompt_idx, 'Process idx:', idx, 'Progress:', f'[{n_gen}/{n_samples}]')

            if n_gen >= n_samples:
                print("Done...")
                print(n_gen, df_llm.shape)
                break

        except Exception as e:
            print(traceback.format_exc())
            print(e)
            time.sleep(10)
            continue
     
    print('Total number of samples used for prompting: ', data_count)
    return df_llm
