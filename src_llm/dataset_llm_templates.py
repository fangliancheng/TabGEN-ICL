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
    # Dictionary to hold the fields for the Pydantic model
    fields = {}
    
    # Iterate over columns to determine their types
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
    
    # Dynamically create the Pydantic model
    JSONModel = create_model(dataname, **fields)
    
    class JSONListModel(BaseModel):
        JSON: List[JSONModel]

    return JSONListModel

def json_templates_reflector(role):
    if role == 'generator':
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
        # generator_template = """
        # Generate 50 samples of synthetic data.
        
        # Each sample should include the following attributes:
        # {data}

        # Make sure that the numbers make sense for each attribute. 

        # Output Format:
        # Present the generated data in a JSON format, structured as a list of objects, where each object represents a single data point with all attributes.
        # """

    if role == 'reflector':
        generator_template = """
        Review the generated dataset in comparison to the original data samples. 
        Your goal is to identify any discrepancies and provide recommendations to refine the generated data so that it more closely matches the original data's distribution.
        
        Task:
        1. Comparative Analysis:
        1.1 Analyze the statistical properties of both the original and generated datasets.
        1.2 Identify any differences in means, variances, distributions, correlations, and category frequencies.
        1.3 Note any inconsistencies or anomalies in the generated data.

        2. Feedback and Recommendations:
        2.1 Provide specific feedback on areas where the generated data deviates from the original data.
        2.2 Suggest actionable steps to adjust the data generation process (e.g., modifying distribution parameters, adjusting category proportions).
        2.3 Highlight any potential issues in the data generation process that could be causing the discrepancies.

        Generated data:
        {generated_data}

        Real data:
        {real_data}

        Output Format:
        Present your findings and recommendations in a clear, structured report.
        """
    if role == 'refiner':
        generator_template = """
        You will be provided with some generated data samples and a report with feedback on the generated data. 
        Your task is to follow the feedback in the report and refine the generated data samples to address the issues identified in the feedback.
            
        Generated data: 
        {generated_data}

        Feedback on the generated data:
        {feedback}

        Output Format:
        Present the refined data in a JSON format.
        """
    return generator_template

def json_templates(role, interpolation=False):
    if role == 'generator':
        generator_template = """
        You are a synthetic data generator tasked with producing new tabular data samples that closely mirror the distribution and characteristics of the original dataset.

        **Before you begin**, please:

        - **Review the provided statistical analysis report thoroughly** to understand key properties such as mean, median, mode, standard deviation, correlations, and distributions.
        - **Ensure diversity and balance** by reflecting the variety present in the original dataset, especially for categorical variables with uneven class distributions.
        - **Consider data augmentation techniques** to increase variability while maintaining the underlying data structure.

        **Task**:

        1. Create a dataset with the same columns and data types as the original.
        2. Generate data that aligns with the statistical properties provided in the analysis, ensuring statistical alignment.
        3. For numerical variables:
           - Generate values that follow the identified distributions and statistical parameters.
           - Introduce controlled variability while maintaining overall statistical integrity.
        4. For categorical variables:
           - Assign categories according to the observed frequencies and proportions.
           - Ensure all categories are appropriately represented.
        5. Maintain any identified correlations and relationships between variables.
        6. Reflect the diversity and balance of the original dataset in your generated data.

        **Statistical Analysis Report**:
        {dataset_description}

        **Real examples for reference**:
        {data}

        **Output Format**:
        Present the data in a JSON format.
        """
        return generator_template
    elif role == 'reflector':
        reflector_template = """
        You are tasked with analyzing the generated data samples and comparing them to the real data samples.

        **Your goal** is to provide detailed, actionable feedback highlighting discrepancies between the generated and real data, focusing on:

        - **Causal structure**: Are the relationships and dependencies between variables consistent?
        - **Feature distributions**: Do the distributions of each feature match between the real and generated data?
        - **Label distributions**: For labeled data, do the class proportions and distributions align?

        **Instructions**:

        1. **Perform statistical tests** (e.g., Kolmogorov-Smirnov test for continuous variables, Chi-squared test for categorical variables) to quantitatively assess similarities and differences.
        2. **Utilize visualization tools** such as histograms, box plots, and scatter plots to compare distributions and relationships visually.
        3. **Provide specific, actionable feedback** on any discrepancies, anomalies, or inconsistencies found.
        4. Highlight areas where the generated data deviates from the real data and suggest concrete improvements.

        **Generated data**:
        {generated_data}

        **Real data**:
        {real_data}

        **Output Format**:

        - Present your findings in a clear, structured report.
        - Include tables, charts, or graphs where appropriate.
        - Be specific in your suggestions for improvement.
        """
        return reflector_template
    elif role == 'refiner':
        refiner_template = """
        You are tasked with refining the generated data samples based on feedback from the reflector.

        **Your goals** are to:

        - **Incorporate the reflector's feedback** to adjust the generated data.
        - **Correct specific errors and discrepancies** identified.
        - **Enhance the overall quality** of the generated data to align more closely with the real data's statistical properties.

        **Instructions**:

        1. Review the reflector's feedback thoroughly.
        2. Adjust the generation parameters or data samples to address the specific issues raised.
        3. For numerical variables:
           - Modify distributions, means, and variances as needed to match the real data.
        4. For categorical variables:
           - Rebalance category frequencies to reflect the real data proportions.
        5. Reassess correlations and dependencies between variables to ensure consistency.
        6. Consider iterating through this process multiple times to progressively improve data quality.

        **Reflector's Feedback**:
        {feedback}

        **Generated data before refinement**:
        {generated_data}

        **Output Format**:

        Present the refined data in a JSON format.
        """
        return refiner_template
    else:
        generator_template = "Role not recognized. Please specify a valid role."




def markdown_templates(X_train_orig, y_train_orig, role, interpolation=False):
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
    
    if role == 'analyzer':
        generator_template = """                
        You are a data analyzer tasked with understanding and analyzing the provided real data. Your goal is to extract insights and patterns from the data that will inform the generation of synthetic data.
            
        Samples from the dataset provided for your analysis: {data}

        Tasks:
        1. Provide the range (minimum and maximum values) for each numerical feature.
        2. Describe the distribution (mean, median, mode, standard deviation) for each numerical feature.
        3. Calculate the proportions of each category for each catergorical feature.
        4. Identify any noticeable patterns and correlations between features.

        Begin your analysis with this sample data and provide a detailed summary.
        """

    elif role == "generator": 
        generator_template = """
        You are a synthetic data generator tasked with producing data that mirrors the given examples in both causal structure and feature-label distributions, while ensuring a high degree of diversity in the generated samples.

        Here is how you will proceed:
        1. I will provide you with real examples to learn from. I will also provide a detailed description of the dataset.
        2. Use your understanding of salary predictions based on demographic features to generate a realistic yet diverse set of samples, you can 
        also leverage the provided dataset description to guide your generation process.

        Your objectives are:
        - Reflect the causal relationships and distribution patterns found in the example data.
        - Generate new, unique samples that do not replicate the examples provided.
        - Ensure the generated data is statistically indistinguishable from the real data.

        Below are the real examples:
        {data}

        Below is a detailed description and summary of the dataset:
        {dataset_description}

        {format_instructions}

        Remember:
        - Your goal is to create realistic, diverse samples with correct labels based on the given features.
        - DO NOT copy the examples.
        """
        
    elif role == "corrector":
        generator_template = """
        Your task is to analyze the generated examples and compare them to the real examples, focusing on causal structure, feature distributions, and label distributions.
        If any discrepancies are found, correct or remove the samples to ensure the generated data is as realistic as possible. Just output the corrected samples, do not add explanations or comments.

        Generated data: {generated_data}

        Real data: {real_data}

        {format_instructions}
        """

    
    df_orig = df_orig.sample(frac=1).reset_index(drop=True)

    return generator_template, format_instructions, df_orig
