from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

import pandas as pd
from pydantic import BaseModel, create_model
from typing import List


ylabel_map= {
            "adult": "income", #binary classification
            "beijing": "pm2.5", #regression
            "default": "default payment next month", #binary classification
            "magic": "class", #binary classification
            "news": " shares", # integer classification
            "shoppers": "Revenue", #binary classification
            }


def create_json_model(df: pd.DataFrame, dataname=None) -> BaseModel:
    fields = {}
    
    for column in df.columns:
        if df[column].dtype == 'object':  
            fields[column] = (str, ...)
        elif df[column].dtype == 'int64':
            fields[column] = (int, ...)
        elif df[column].dtype == 'float64':
            fields[column] = (float, ...)
        elif df[column].dtype == 'bool':
            fields[column] = (bool, ...)
        else:
            raise TypeError(f"Unexpected dtype for column {column}: {df[column].dtype}")
    
    JSONModel = create_model(dataname, **fields)
    
    class JSONListModel(BaseModel):
        JSON: List[JSONModel]

    return JSONListModel

def json_templates_RES_RAG():
    
    generator_template = """
    You are a synthetic data generator tasked with creating new tabular data samples that closely mirror the distribution and characteristics of the original dataset.

    Instructions:
    1. Analyze the provided real samples carefully.
    2. Generate synthetic data that maintains the statistical properties of the real data.
    3. Ensure all attributes cover their full expected ranges, including less common or extreme values.
    4. Maintain the relationships and correlations between different attributes.
    5. Preserve the overall distribution of the real data while introducing realistic variations.

    Key points to consider:
    - Replicate the data types of each column (e.g., numerical, categorical).
    - Match the range and distribution of numerical attributes.
    - Maintain the frequency distribution of categorical attributes.
    - Reflect any patterns or trends present in the original data.
    - Introduce realistic variability to avoid exact duplication.

    Real samples:
    {data}

    Output Format:
    Present the generated data in a JSON format, structured as a list of objects, where each object represents a single data point with all attributes.
    """

    # Dummy template for testing
    # generator_template = """
    # Generate 50 samples of synthetic data.
    
    # Each sample should include the following attributes:
    # {data}

    # Make sure that the numbers make sense for each attribute. 

    # Output Format:
    # Present the generated data in a JSON format, structured as a list of objects, where each object represents a single data point with all attributes.
    # """

    return generator_template


def markdown_templates_RES_RAG(X_train_orig, y_train_orig):
    response_schemas = []

    df_orig = pd.concat([X_train_orig, y_train_orig], axis=1)

    for col in list(df_orig.columns):
        if col in ylabel_map:
            resp = ResponseSchema(
                name=ylabel_map[col],
                description=f"label column",
            )
        # if col == "income":
        #     resp = ResponseSchema(
        #         name="income",
        #         description=f"label if salary above 50K or not, {col}",
        #     )
        # elif col == "default payment next month":
        #     resp = ResponseSchema(
        #         name="default payment next month",
        #         description=f"binary label if default payment next month or not, {col}",
        #     )
        # elif col == "class":
        #     resp = ResponseSchema(
        #         name="class",
        #         description=f"binary label, {col}",
        #     )
        # elif col == "shares":
        #     resp = ResponseSchema(
        #         name="shares",
        #         description=f"integer label, {col}",
        #     )
        # elif col == "Revenue":
        #     resp = ResponseSchema(
        #         name="Revenue",
        #         description=f"binary label, {col}",
        #     )   
        # elif col == 'pm2.5':
        #     resp = ResponseSchema(
        #         name='pm2.5',
        #         description=f"continuous label, {col}",
        #     )
        else:
            resp = ResponseSchema(
                name=col,
                description=f"feature column",
            )
        response_schemas.append(resp)

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    
    generator_template = """
    You are a synthetic data generator tasked with creating new tabular data samples that closely mirror the distribution and characteristics of the original dataset.

    Instructions:
    1. Analyze the provided real samples carefully.
    2. Generate synthetic data that maintains the statistical properties of the real data.
    3. Ensure all attributes cover their full expected ranges, including less common or extreme values.
    4. Maintain the relationships and correlations between different attributes.
    5. Preserve the overall distribution of the real data while introducing realistic variations.

    Key points to consider:
    - Replicate the data types of each column (e.g., numerical, categorical).
    - Match the range and distribution of numerical attributes.
    - Maintain the frequency distribution of categorical attributes.
    - Reflect any patterns or trends present in the original data.
    - Introduce realistic variability to avoid exact duplication.

    Real samples:
    {data}

    {format_instructions}
    """

    df_orig = df_orig.sample(frac=1).reset_index(drop=True)

    return generator_template, format_instructions, df_orig
