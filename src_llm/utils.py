import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance
from scipy import stats
from scipy.stats import wasserstein_distance_nd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from geomloss import SamplesLoss
from sklearn.utils import resample
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
)

import pdb

def wasserstein_distance(X: np.ndarray, X_syn: np.ndarray) -> float:
    """
    Compute Wasserstein distance between original data and synthetic data.

    Args:
        X: Original data (numpy array)
        X_syn: Synthetically generated data (numpy array)

    Returns:
        WD_value: Wasserstein distance (float)
    """
    # Reshape data if necessary
    X_ = X.reshape(len(X), -1)
    X_syn_ = X_syn.reshape(len(X_syn), -1)

    # Ensure both datasets have the same number of samples
    if len(X_) > len(X_syn_):
        X_syn_ = np.concatenate([X_syn_, np.zeros((len(X_) - len(X_syn_), X_.shape[1]))])

    # Normalize data
    scaler = MinMaxScaler().fit(X_)
    X_ = scaler.transform(X_)
    X_syn_ = scaler.transform(X_syn_)

    # Convert to tensors
    X_ten = torch.from_numpy(X_)
    Xsyn_ten = torch.from_numpy(X_syn_)

    # Compute Wasserstein distance using Sinkhorn algorithm
    OT_solver = SamplesLoss(loss="sinkhorn")
    WD_value = OT_solver(X_ten, Xsyn_ten).cpu().numpy().item()

    return WD_value


def jensen_shannon_distance(X_gt: np.ndarray, X_syn: np.ndarray, normalize: bool = True, n_bins: int = 10, laplace_smoothing: bool = True) -> float:
    """
    Evaluate the average Jensen-Shannon distance between two 2D numpy arrays.

    Args:
        X_gt (np.ndarray): Ground truth data (2D array).
        X_syn (np.ndarray): Synthetic data (2D array).
        normalize (bool): If True, normalize the value counts to probabilities.
        n_bins (int): Maximum number of bins for histogram.
        laplace_smoothing (bool): If True, apply Laplace smoothing to avoid zero probabilities.

    Returns:
        float: The average Jensen-Shannon distance across all columns.
    """
    if X_gt.shape[1] != X_syn.shape[1]:
        raise ValueError("The number of columns in X_gt and X_syn must be the same.")
    
    jsd_values = []

    # Iterate over each column
    for i in range(X_gt.shape[1]):
        # Get the column data
        col_gt = X_gt[:, i]
        col_syn = X_syn[:, i]
        
        # Determine the number of unique values in the ground truth column for binning
        local_bins = min(n_bins, len(np.unique(col_gt)))
        
        # Create histograms for both ground truth and synthetic data
        gt_hist, gt_bins = np.histogram(col_gt, bins=local_bins, density=normalize)
        syn_hist, _ = np.histogram(col_syn, bins=gt_bins, density=normalize)
        
        # Apply Laplace smoothing if needed
        if laplace_smoothing:
            gt_hist += 1
            syn_hist += 1
        
        # Normalize histograms again after smoothing
        gt_hist = gt_hist / np.sum(gt_hist)
        syn_hist = syn_hist / np.sum(syn_hist)

        # Compute Jensen-Shannon distance for the column
        jsd_value = distance.jensenshannon(gt_hist, syn_hist)
        if np.isnan(jsd_value):
            raise RuntimeError(f"NaNs in JSD calculation for column {i}")
        
        jsd_values.append(jsd_value)

    # Return the average JSD across all columns
    return np.mean(jsd_values)

def compute_distance(df, sampled_indices_collection, df_tmp, over_sampling=True, dist_type=None, prompt_idx=0):
    assert dist_type is None

    distances = []

    for sampled_idx in sampled_indices_collection:
        sampled_df = df.iloc[sampled_idx]
        combined_df = pd.concat([sampled_df, df_tmp], axis=0)
        
        encoded_gt = []
        encoded_syn = []
        
        for col in df.columns:
            if df[col].dtype == 'object':
                #print(col, 'cat')
                cat_col_gt = df[col].to_numpy().reshape(-1, 1)
                cat_col_syn = combined_df[col].to_numpy().reshape(-1, 1)

                enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                encoded_cat_col_gt = enc.fit_transform(cat_col_gt)
                encoded_cat_col_syn = enc.transform(cat_col_syn)

                encoded_gt.append(encoded_cat_col_gt)
                encoded_syn.append(encoded_cat_col_syn)
            else: 
                #print(col, 'num')
                encoded_gt.append(df[col].to_numpy().reshape(-1, 1))
                encoded_syn.append(combined_df[col].to_numpy().reshape(-1, 1))
        
        combined_encoded_df = np.concatenate(encoded_gt, axis=1) #gt
        combined_encoded = np.concatenate(encoded_syn, axis=1) #syn

        
        if over_sampling:
            combined_encoded = resample(combined_encoded, replace=True, n_samples=combined_encoded_df.shape[0], random_state=2024)
        
        if dist_type == 'jensenshannon' or prompt_idx % 2 == 0:
            dist = jensen_shannon_distance(combined_encoded, combined_encoded_df)
        
        elif dist_type == 'kstest' or prompt_idx % 2 == 1:
            dist = 0
            for col_idx in range(combined_encoded.shape[1]):
                cur_dist = stats.kstest(combined_encoded[:, col_idx], combined_encoded_df[:, col_idx]).statistic_location
                dist += cur_dist
            dist = dist / combined_encoded.shape[1]
        
        if dist_type == 'wasserstein' or prompt_idx % 10 == 0:
            print('wasserstein')
            dist = wasserstein_distance(combined_encoded, combined_encoded_df)
        
        distances.append(dist)

    return distances

def get_indices(df, col, dtype, max_num=500, num=100):
    subset_collection = []
    if dtype == 'int64' or dtype == 'float64':
        quantize_col = np.linspace(df[col].min(), df[col].max(), num=num)
        for l,r in zip(quantize_col[:-1], quantize_col[1:]):
            subset_idx = df[(df[col] >= l) & (df[col] < r)].index
            if len(subset_idx) > max_num:
                subset_idx = np.random.choice(subset_idx, size=max_num, replace=False)
            subset_collection.append(subset_idx)
        return subset_collection
    elif dtype == 'object':
        for cls in df[col].unique():
            subset_idx = df[df[col] == cls].index
            if len(subset_idx) > max_num:
                subset_idx = np.random.choice(subset_idx, size=max_num, replace=False)
            subset_collection.append(subset_idx)
        return subset_collection
    else:
        raise ValueError('Unsupported dtype: ' + dtype)


def split_data(df, target_column, n_samples_per_class, random_state=42):
    train_data = (
        df.groupby(target_column)
        .apply(lambda x: x.sample(n_samples_per_class, random_state=random_state))
        .reset_index(level=0, drop=True)
    )
    remaining_data = df.drop(train_data.index)

    return train_data, remaining_data


def sample_and_split(df_feat, df_label, ns=10, seed=100):
    # Args: ns: number of samples per class to sample.
    #       If set to int, sample equal number of samples per class.
    #       If set to 'all', use all samples.
    
    if isinstance(ns, int):
        # Check if n is greater than the minimum class size. If it is, raise an error.
        min_class_size = df_label.value_counts().min()
        if ns > min_class_size:
            raise ValueError(
                f"n is greater than your smallest class size of {min_class_size}."
            )

        # Sample n indices from each class
        sampled_indices = df_label.groupby(df_label).apply(
            lambda x: x.sample(ns, random_state=seed).index
        )

        # Concatenate indices from all classes into a single list
        sampled_indices = [idx for sublist in sampled_indices for idx in sublist]

        # Split df_feat into sampled indices and remaining
        sampled_df = df_feat.loc[sampled_indices]
        remaining_df = df_feat.drop(sampled_indices)

        # Split df_label into sampled indices and remaining
        sampled_labels = df_label.loc[sampled_indices]
        remaining_labels = df_label.drop(sampled_indices)
    
    elif ns == 'all':
        sampled_df = df_feat 
        sampled_labels = df_label
        remaining_df, remaining_labels = None, None

    else:
        raise ValueError("ns must be an integer or 'all'.")


    return sampled_df, remaining_df, sampled_labels, remaining_labels


def evaluate_model(X_train, y_train, X_test, y_test, clf):

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # compute y_score
    try:
        y_score = clf.predict_proba(X_test)[:, 1]
    except:
        y_score = y_pred

    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    try:
        auc = roc_auc_score(y_test, y_score)
    except:
        auc = 0

    return acc, rec, prec, f1, auc, clf


def compute_synthcity_metrics(results, X_train_orig, y_train_orig, X_ref):

    from typing import Any, Tuple, Type

    # synthcity absolute
    from synthcity.metrics.eval_statistical import (
        AlphaPrecision,
        ChiSquaredTest,
        InverseKLDivergence,
        JensenShannonDistance,
        KolmogorovSmirnovTest,
        MaximumMeanDiscrepancy,
        PRDCScore,
        SurvivalKMDistance,
        WassersteinDistance,
    )

    from synthcity.metrics.eval_sanity import NearestSyntheticNeighborDistance

    from synthcity.plugins import Plugin, Plugins

    from synthcity.plugins.core.dataloader import (
        DataLoader,
        GenericDataLoader,
        create_from_info,
    )

    def _eval_plugin(
        evaluator_t: Type, X: DataLoader, X_syn: DataLoader, **kwargs: Any
    ) -> Tuple:
        evaluator = evaluator_t(**kwargs)

        syn_score = evaluator.evaluate(X, X_syn)

        return syn_score

    metrics = [
        AlphaPrecision,
        InverseKLDivergence,
        JensenShannonDistance,
        # KolmogorovSmirnovTest,
        MaximumMeanDiscrepancy,
        PRDCScore,
        WassersteinDistance,
        NearestSyntheticNeighborDistance,
    ]

    # easy_train, ambig_train, hard_train, Curator_llm  = data_centric(X_train_orig = X_train_orig,
    #             y_train_orig= y_train_orig,
    #             X_check = results['llm']['X'],
    #             y_check = results['llm']['y'])

    data_check_dict = {}

    for model in list(results.keys()):
        # tmp_dict = results[model]['X']
        # tmp_dict['y'] = results[model]['y']
        data_check_dict[f"{model}"] = results[model]["df"]

    # data_check_dict["X_train_llm_easy"] = results['llm']['X'].iloc[easy_train,:]
    # data_check_dict["X_train_llm_ambig"] = results['llm']['X'].iloc[ambig_train,:]
    # data_check_dict["X_train_llm_hard"] = results['llm']['X'].iloc[hard_train,:]

    statistical_metrics = {}

    for metric in metrics:
        # print(f"Metric: {metric}")
        tmp_dict = {}
        metric_name = metric.__name__
        for method in data_check_dict.keys():

            try:
                data_check = data_check_dict[method]

                if metric == AlphaPrecision:

                    if X_ref.shape[0] > data_check.shape[0]:
                        trial_results = _eval_plugin(
                            metric,
                            GenericDataLoader(
                                X_ref.astype(float).sample(data_check.shape[0])
                            ),
                            GenericDataLoader(data_check.astype(float)),
                        )
                    elif X_ref.shape[0] < data_check.shape[0]:
                        trial_results = _eval_plugin(
                            metric,
                            GenericDataLoader(X_ref.astype(float)),
                            GenericDataLoader(
                                data_check.astype(float).sample(X_ref.shape[0])
                            ),
                        )
                    else:
                        trial_results = _eval_plugin(
                            metric,
                            GenericDataLoader(X_ref.astype(float)),
                            GenericDataLoader(data_check.astype(float)),
                        )
                    # print(f"{method}: {trial_results}")
                else:

                    trial_results = _eval_plugin(
                        metric,
                        GenericDataLoader(X_ref.astype(float)),
                        GenericDataLoader(data_check.astype(float)),
                    )
                    # print(f"{method}: {trial_results}")

                if len(trial_results.keys()) == 1:
                    tmp_dict[method] = trial_results[list(trial_results.keys())[0]]
                else:
                    tmp_dict[method] = trial_results

            except Exception as e:
                import traceback

                print(traceback.format_exc())
                print(method, e)
                continue

        statistical_metrics[metric_name] = tmp_dict

    return statistical_metrics


def process_gpt(dataset, n_synthetic, temp, gpt_model, ns, seed):
    import pickle
    
    filename = f"../save_dfs/pipeline_llm_{dataset}_{n_synthetic}_{gpt_model}_{ns}_{seed}.pickle"
    if gpt_model == "gpt4_nocol":
        filename = f"../save_dfs/pipeline_llm_{dataset}_{n_synthetic}_gpt4_{ns}_{seed}_nocol.pickle"
    with open(filename, "rb") as f:
        loaded = pickle.load(f)

    if gpt_model == "gpt4_nocol":
        print(loaded["llm"].keys())
        df = loaded["llm"]["df"]

    else:
        df = loaded["llm"]["X"]
        df["target"] = loaded["llm"]["y"]

    df = df.dropna()
    df = df[
        ~df.apply(
            lambda row: any(
                [
                    isinstance(cell, str)
                    and cell
                    in [
                        "integer",
                        "float",
                        "numeric",
                        "categorical",
                        "number",
                        "No",
                        "Yes",
                        "continuous",
                        "age in years",
                        "string",
                    ]
                    for cell in row
                ]
            ),
            axis=1,
        )
    ]

    example_df = loaded["Original"]["X"]
    example_df["target"] = loaded["Original"]["y"]

    if gpt_model == "gpt4_nocol":
        original_cols = example_df.columns
        example_df.columns = [
            "feat_" + str(i) for i in range(example_df.shape[1] - 1)
        ] + ["target"]

    try:
        df = df.astype(example_df.dtypes)
    except:
        # Assuming the dtypes from the example_df['Dtrain'].dataframe() is what you want
        target_dtypes = example_df.dtypes.to_dict()

        problematic_rows = set()

        for col, dtype in target_dtypes.items():
            for index, value in df[col].items():
                try:
                    _ = dtype.type(value)  # Try to convert the value
                except Exception:
                    problematic_rows.add(index)

        # Convert the problematic rows to a list and sort them
        problematic_rows = sorted(list(problematic_rows))

        # Drop the problematic rows
        df.drop(problematic_rows, inplace=True)

        # Identify rows where any cell is of type list
        rows_with_lists = df.applymap(lambda x: isinstance(x, list)).any(axis=1)

        # Drop those rows
        df = df[~rows_with_lists]

        df = df.astype(example_df.dtypes)

    if gpt_model == "gpt4_nocol":
        df.columns = original_cols
    return df



def process_llama(
    dataset, n_synthetic, temp, llama_model, ns, seed, path="./llama-gen/llama-data"
):
    import pickle

    filename = f"{path}/{llama_model}_{dataset}_{n_synthetic}_{ns}_{seed}.pickle"

    with open(filename, "rb") as f:
        loaded = pickle.load(f)

    df = loaded["llm"]["X"]
    df["target"] = loaded["llm"]["y"]

    # df = df.dropna()
    df = df[
        ~df.apply(
            lambda row: any(
                [
                    isinstance(cell, str)
                    and cell
                    in [
                        "integer",
                        "float",
                        "numeric",
                        "categorical",
                        "number",
                        "No",
                        "Yes",
                        "continuous",
                        "age in years",
                        "string",
                    ]
                    for cell in row
                ]
            ),
            axis=1,
        )
    ]

    example_df = loaded["Original"]["X"]
    example_df["target"] = loaded["Original"]["y"]

    if dataset == "compas" and llama_model == "llama13b":
        # Convert lists in the 'Sex' column to single values (strings)
        df = df[df["sex"].apply(lambda x: not isinstance(x, list))]
        # Define the mapping dictionary
        sex_mapping = {"Male": 1, "Female": 0}
        # Apply the mapping to the 'Sex' column
        df["sex"] = df["sex"].map(sex_mapping)

    if dataset == "adult" and llama_model == "llama13b":
        df = df[df["age"].apply(lambda x: not isinstance(x, list))]

        # Define a custom function to set the values based on conditions
        def set_target_value(target_value):
            try:
                target_value = float(target_value)
                if target_value > 1 and target_value < 50000:
                    return 0
                elif target_value >= 50000:
                    return 1
                else:
                    return target_value  # Keep the original value if it doesn't meet the conditions
            except (ValueError, TypeError):
                return None  # Return None for rows where the conversion to float fails

        # Apply the custom function to update the 'Target' column
        df["target"] = df["target"].apply(set_target_value)

        # Drop rows where 'Target' is None
        df = df.dropna(subset=["target"])

    try:
        df = df.astype(example_df.dtypes)
    except:
        # Assuming the dtypes from the example_df['Dtrain'].dataframe() is what you want
        target_dtypes = example_df.dtypes.to_dict()

        problematic_rows = set()

        for col, dtype in target_dtypes.items():
            for index, value in df[col].items():
                try:
                    _ = dtype.type(value)  # Try to convert the value
                except Exception:
                    problematic_rows.add(index)

        # Convert the problematic rows to a list and sort them
        problematic_rows = sorted(list(problematic_rows))

        # Drop the problematic rows
        df.drop(problematic_rows, inplace=True)

        # Identify rows where any cell is of type list
        rows_with_lists = df.applymap(lambda x: isinstance(x, list)).any(axis=1)

        # Drop those rows
        df = df[~rows_with_lists]

        df = df.astype(example_df.dtypes)

    return df


def process_together(dataset, n_synthetic, temp, gpt_model, ns, seed):
    import pickle

    filename = f"./together_dfs/pipeline_llm_{dataset}_{n_synthetic}_{gpt_model}_{ns}_{seed}.pickle"

    with open(filename, "rb") as f:
        loaded = pickle.load(f)

    # df = loaded['llm']['df']

    df = loaded["llm"]["X"]
    y_tmp = loaded["llm"]["y"]

    df["y"] = y_tmp

    example_df = loaded["Original"]["X"]
    example_df["target"] = loaded["Original"]["y"]

    df.loc[df["target"].isna(), "target"] = df.loc[df["target"].isna(), "y"]
    df = df[example_df.columns]

    # df = df.dropna()
    df = df[
        ~df.apply(
            lambda row: any(
                [
                    isinstance(cell, str)
                    and cell
                    in [
                        "integer",
                        "float",
                        "numeric",
                        "categorical",
                        "number",
                        "No",
                        "Yes",
                        "continuous",
                        "age in years",
                        "string",
                    ]
                    for cell in row
                ]
            ),
            axis=1,
        )
    ]

    example_df = loaded["Original"]["X"]
    example_df["target"] = loaded["Original"]["y"]

    if gpt_model == "gpt4_nocol":
        original_cols = example_df.columns
        print(example_df.shape)
        example_df.columns = [
            "feat_" + str(i) for i in range(example_df.shape[1] - 1)
        ] + ["target"]

    try:
        df = df.astype(example_df.dtypes)
    except:
        # Assuming the dtypes from the example_df['Dtrain'].dataframe() is what you want
        target_dtypes = example_df.dtypes.to_dict()

        problematic_rows = set()

        for col, dtype in target_dtypes.items():
            for index, value in df[col].items():
                try:
                    _ = dtype.type(value)  # Try to convert the value
                except Exception:
                    problematic_rows.add(index)

        # Convert the problematic rows to a list and sort them
        problematic_rows = sorted(list(problematic_rows))

        # Drop the problematic rows
        df.drop(problematic_rows, inplace=True)

        # Identify rows where any cell is of type list
        rows_with_lists = df.applymap(lambda x: isinstance(x, list)).any(axis=1)

        # Drop those rows
        df = df[~rows_with_lists]

        df = df.astype(example_df.dtypes)

    return df


def process_swahili(dataset, n_synthetic, temp, gpt_model, ns, seed):
    import pickle

    filename = f"./swahili_dfs/pipeline_llm_{dataset}_{n_synthetic}_{gpt_model}_{ns}_{seed}.pickle"

    with open(filename, "rb") as f:
        loaded = pickle.load(f)

    if gpt_model == "gpt4_nocol":
        print(loaded["llm"].keys())
        df = loaded["llm"]["df"]

    else:
        df = loaded["llm"]["X"]
        df["target"] = loaded["llm"]["y"]

    df = df.dropna()
    df = df[
        ~df.apply(
            lambda row: any(
                [
                    isinstance(cell, str)
                    and cell
                    in [
                        "integer",
                        "float",
                        "numeric",
                        "categorical",
                        "number",
                        "No",
                        "Yes",
                        "continuous",
                        "age in years",
                        "string",
                    ]
                    for cell in row
                ]
            ),
            axis=1,
        )
    ]

    example_df = loaded["Original"]["X"]
    example_df["target"] = loaded["Original"]["y"]

    df = df[example_df.columns]

    if gpt_model == "gpt4_nocol":
        original_cols = example_df.columns
        print(example_df.shape)
        example_df.columns = [
            "feat_" + str(i) for i in range(example_df.shape[1] - 1)
        ] + ["target"]

    try:
        df = df.astype(example_df.dtypes)
    except:
        # Assuming the dtypes from the example_df['Dtrain'].dataframe() is what you want
        target_dtypes = example_df.dtypes.to_dict()

        problematic_rows = set()

        for col, dtype in target_dtypes.items():
            for index, value in df[col].items():
                try:
                    _ = dtype.type(value)  # Try to convert the value
                except Exception:
                    problematic_rows.add(index)

        # Convert the problematic rows to a list and sort them
        problematic_rows = sorted(list(problematic_rows))

        # Drop the problematic rows
        df.drop(problematic_rows, inplace=True)

        # Identify rows where any cell is of type list
        rows_with_lists = df.applymap(lambda x: isinstance(x, list)).any(axis=1)

        # Drop those rows
        df = df[~rows_with_lists]

        df = df.astype(example_df.dtypes)

    return df
