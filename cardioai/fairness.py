import os
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
from functools import reduce
from fairlearn.metrics import (
    MetricFrame,
    count,
    false_negative_rate,
    false_positive_rate,
    true_positive_rate,
    selection_rate,
    demographic_parity_difference,
    equalized_odds_difference,
    demographic_parity_ratio,
    equalized_odds_ratio,
    equal_opportunity_difference,
    equal_opportunity_ratio
)
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score


def count_true(y_true, y_pred) -> int:
    r"""Calculate the number of data points in each group when working with `MetricFrame`.

    The ``y_true`` argument is used to make this calculation. For consistency with
    other metric functions, the ``y_pred`` argument is required, but ignored.

    Read more in the :ref:`User Guide <assessment>`.

    Parameters
    ----------
    y_true : array_like
        The list of true labels

    y_pred : array_like
        The predicted labels (ignored)

    Returns
    -------
    int
        The number of true data points in each group.
    """
    # check_consistent_length(y_true, y_pred)
    return np.sum(y_true)


metrics = {
    "balanced accuracy": balanced_accuracy_score,
    "accuracy": accuracy_score,
    "f1-score": f1_score,
    "false positive rate": false_positive_rate,
    "false negative rate": false_negative_rate,
    "true positive rate": true_positive_rate,
    "selection rate": selection_rate,
    "number of samples": count,
    "positive cases": count_true,
    }

fairness_metrics = {
    "Demographic parity difference": demographic_parity_difference,
    "Equalized odds difference": equalized_odds_difference,
    "Equal opportunity difference": equal_opportunity_difference,
    "Demographic parity ratio": demographic_parity_ratio,
    "Equalized odds ratio": equalized_odds_ratio,
    "Equal opportunity ratio": equal_opportunity_ratio
}

def create_data_with_predictions(predictions_filepath, dataset_dir):
    tab_directory = os.path.join(dataset_dir, "by_udi_decoded/")

    codes = {
    # "20209":"CMR",
    "53": "Date of attendance",
    "31":"Sex",
    "21000":"Ethnic background",
    "34":"Year of birth",
    "21002":"Weight",
    "12144":"Height",
    # "22427":"Body surface area",
    "21001":"BMI",
    "189":"Townsend index",
    # "41270":"Diagnoses ICD10",
    # "41280":"Date of diagnosis ICD10",
    # "41271":"Diagnoses ICD9",
    # "41280":"Date of diagnosis ICD9",
    # "53":"Date of attending"
}
    predictions = pd.read_csv(predictions_filepath, index_col=0)

    tables = []
    tables.append(predictions)
    # Check if dataset_dir is a file
    if os.path.isfile(dataset_dir):
        metadata = pd.read_csv(dataset_dir, index_col=0)
        metadata = metadata.drop(columns=['label'])
        tables.append(metadata)
        table_merged = reduce(lambda  left,right: pd.merge(left,right,on=['f.eid'],
                                                    how='left'), tables)
        # table_merged['BSA'] = 0.007184 * (table_merged['Height']**0.725) * (table_merged['Weight']**0.425)
        
        return table_merged
    else:
        for code in codes:
            table_code = pd.read_csv(tab_directory+"f.{}_decoded.tab".format(code), index_col=0, sep='\t')
            if 'f.eid' not in table_code.columns:
                table_code = pd.read_csv(tab_directory+"f.{}_decoded.tab".format(code), index_col=1, sep='\t')

            if 'f.20209.2.0' in table_code.columns:
                table_code = table_code.dropna(thresh=2)
                print(table_code.shape[0])
            
            if 'f.21000.2.0' in table_code.columns:
                table_code = table_code.drop(columns=['f.21000.2.0', 'f.21000.1.0'])

            table_code.columns = table_code.columns.str.replace('f.{}'.format(code), codes[code])

            tables.append(table_code)

        table_merged = reduce(lambda  left,right: pd.merge(left,right,on=['f.eid'],
                                                    how='left'), tables)
        
        date_columns = ['Date of attendance.0.0', 'Date of attendance.1.0', 'Date of attendance.2.0', 'Date of attendance.3.0']
        table_merged[date_columns] = table_merged[date_columns].apply(pd.to_datetime)
        table_merged['Age'] = table_merged["Date of attendance.2.0"].dt.year - table_merged['Year of birth.0.0']

        for code, feature in codes.items():
            # Chech if column exists
            if feature + '.2.0' in table_merged.columns and feature + '.0.0' in table_merged.columns:
                table_merged[feature] = table_merged[feature+'.2.0'].fillna(table_merged[feature+'.0.0'])
            elif feature + '.2.0' in table_merged.columns:
                table_merged[feature] = table_merged[feature+'.2.0']
            elif feature + '.0.0' in table_merged.columns:
                table_merged[feature] = table_merged[feature+'.0.0']
            
            table_merged = table_merged.dropna(subset = [feature])
        # BSA = 0.007184 x (Height(cm)^0.725) x (Weight(kg)^0.425)
        table_merged['BSA'] = 0.007184 * (table_merged['Height']**0.725) * (table_merged['Weight']**0.425)

    return table_merged


def group_data(data, protected_feature, n_groups):
    data[protected_feature+'_grouped'] = pd.qcut(data[protected_feature], n_groups,
                               labels = False)

    labels = data.groupby([protected_feature+'_grouped'])[protected_feature].median().to_list()
    labels = [round(x, 2) for x in labels]
    data[protected_feature+'_grouped'] = pd.qcut(data[protected_feature], n_groups,
                               labels = labels)
    return data


def group_by_bmi_rules(data, feature="BMI"):
    conditions = [
        data[feature].gt(30),
        data[feature].ge(25) & data[feature].lt(30),
        data[feature].lt(25),
        # data[feature].lt(18.5)
    ]
    choices = ["3 Obese", "2 Overweight", "1 Normal"]
    data["BMI_grouped"] = np.select(conditions, choices, default=choices[1])
    # sum label columns and protected feature
    # data["grouped_labels"] = data["protected_feature"] + data["label"]

    return data

def group_by_decades(data, feature="Age"):
    conditions = [
        data[feature].ge(80),
        data[feature].ge(70) & data[feature].lt(80),
        data[feature].ge(60) & data[feature].lt(70),
        data[feature].ge(50) & data[feature].lt(60),
        data[feature].ge(40) & data[feature].lt(50),
    ]
    choices = ["80s", "70s", "60s", "50s", "40s"]
    data["Age_grouped"] = np.select(conditions, choices, default=choices[1])

    return data

def get_metrics_for_feature(data, protected_feature, plot=False):
    y_true = data["label"].values
    y_pred = data["prediction"].values
    feature = data[protected_feature+"_grouped"].values

    # Analyze metrics using MetricFrame
    metric_frame = MetricFrame(
        metrics=metrics, y_true=y_true, y_pred=y_pred, sensitive_features=feature
    )

    # Add fairness metrics
    fairness_metrics_results = {}
    for name, metric in fairness_metrics.items():
        metric_value = metric(y_true, y_pred, sensitive_features=feature)
        fairness_metrics_results[name] = metric_value

    # Find group name with lowest metrics
    min_idxs = metric_frame.by_group.idxmin()
    # Find group name with highest metrics
    max_idxs = metric_frame.by_group.idxmax()
    # Add min and max group names to results
    fairness_metrics_results["DPD min vs max"] = f"{min_idxs['selection rate']} vs {max_idxs['selection rate']}"
    fairness_metrics_results["EOD min vs max"] = f"{min_idxs['false positive rate']} vs {max_idxs['false positive rate']}"
    fairness_metrics_results["EOR min vs max"] = f"{min_idxs['true positive rate']} vs {max_idxs['true positive rate']}"

    if plot:
        metric_frame.by_group.plot.bar(
            subplots=True,
            layout=[3, 3],
            legend=False,
            figsize=[12, 8],
            title="Fairness metrics by {}".format(protected_feature)
        )

    return metric_frame, fairness_metrics_results
    

def group_by_features(data, features):
    data_grouped = data
    for protected_feature in features:
        if protected_feature == "BMI":
            data_grouped = group_by_bmi_rules(data, protected_feature)
        elif protected_feature == "Sex":
            data_grouped['Sex_grouped'] = data[protected_feature]
        # elif protected_feature == "Age":
        #     data_grouped = group_by_decades(data, protected_feature)
        else: # Continuous features without rules
            data_grouped = group_data(data, protected_feature, n_groups=3)
    
    return data_grouped

def calculate_fairness(predictions_filepath, balanced=False):
    experiment_dir = str(Path(predictions_filepath).parent)
    config_path = os.path.join(experiment_dir, "config.yaml")
    with open(config_path) as file:
        config = yaml.safe_load(file)
    
    if config["dataset"]["UKBB"]["dataset_file"]:
        metadata_dir = config["dataset"]["UKBB"]["dataset_file"]
    else:
        dataset_path = config["dataset"]["UKBB"]["dataset_path"]
        metadata_dir = str(Path(dataset_path).parent)

    predictions_with_metadata = create_data_with_predictions(predictions_filepath, metadata_dir)

    protected_features = ["Sex", "Age", "BMI", "Townsend index"]

    grouped_data = group_by_features(predictions_with_metadata, protected_features)
    grouped_data["metagroup"] = grouped_data["label"].astype(str)
    # Create metagroup
    features_for_grouping = ["Sex", "Age", "BMI"]
    for protected_feature in features_for_grouping:
        grouped_data["metagroup"] = grouped_data["metagroup"] + grouped_data[protected_feature + "_grouped"].astype(str)
    data_grouped = grouped_data
    metrics_dict = {}

    for protected_feature in protected_features:
    #     if protected_feature == "BMI":
    #         data_grouped = group_by_bmi_rules(predictions_with_metadata, protected_feature)
    #     elif protected_feature == "Sex":
    #         data_grouped = predictions_with_metadata
    #         data_grouped['protected_feature'] = data_grouped[protected_feature]
    #     else: # Continuous features without rules
    #         data_grouped = group_data(predictions_with_metadata, protected_feature, n_groups=4)

        metric_frame, fairness_metrics = get_metrics_for_feature(grouped_data, protected_feature, plot=False)
        metrics_dict[protected_feature] = {"per_group_metrics": metric_frame, "fairness_metrics": fairness_metrics}

        if balanced:
            data_grouped = data_grouped.copy().groupby(['protected_feature', 'label'])
            rows_per_group = data_grouped.size().min()
            data_grouped = data_grouped.head(rows_per_group)
            metric_frame, fairness_metrics = get_metrics_for_feature(data_grouped, protected_feature, plot=False)
            metrics_dict[protected_feature + "_balanced"] = {"per_group_metrics": metric_frame, "fairness_metrics": fairness_metrics}

    return metrics_dict


if __name__ == "__main__":

    experiment_dir = "logs/"

    config_path = os.path.join(experiment_dir, "config.yaml")
    with open(config_path) as file:
        config = yaml.safe_load(file)
    predictions_path = os.path.join(experiment_dir, "test_predictions.csv")

    metrics_per_feature = calculate_fairness(predictions_path)

    for protected_feature, metrics in metrics_per_feature.items():
        metric_frame = metrics["per_group_metrics"]
        fairness_metrics = metrics["fairness_metrics"]
        print("\n\n\n" + protected_feature)
        to_print = metric_frame.by_group.to_string(float_format="{:.3f}".format)
        print(to_print)
        print("\n")
        for name, metric in fairness_metrics.items():
            print(name, metric)
