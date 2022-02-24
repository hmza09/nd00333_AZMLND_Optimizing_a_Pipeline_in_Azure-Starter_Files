# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary


## Scikit-learn Pipeline


Specified the parameter sampler:

```
ps = RandomParameterSampling( {
        "C": uniform(0.05, 0.1),
        "max_iter": choice(0, 80, 100, 120, 140)
    })
```
"C" is Regularization and "max_iter" is number of iteration and both are set as specified in train.py that "C" should be between 0.0 - 1.0 and "max_iter"
whole number.
_RandomParameterSampling_ is one of the choices available for the sampler and I chose it because it is the faster and supports early termination of 
low-performance runs. If budget is not an issue, we could use _GridParameterSampling_ to exhaustively search over the search space or 
_BayesianParameterSampling_ to explore the hyperparameter space.
Bandit is an early termination policy based on slack factor/slack amount and evaluation interval. The policy early terminates any runs where the 
primary metric is not within the specified slack factor/slack amount with respect to the best performing training run.
_slack_factor_: The amount of slack allowed with respect to the best performing training run. This factor specifies the slack as a ratio.

HyperDrive configuration includes information about hyperparameter space sampling, termination policy, primary metric, resume from configuration, estimator, and the compute target to execute the experiment runs on.
## AutoML
```
automl_config = AutoMLConfig(
    experiment_timeout_minutes=30,
    task="classification",
    primary_metric="accuracy",
    training_data=dataset,
    label_column_name='y',
    n_cross_validations=3)
  ```
task = 'Classification' specfies that the task is of classification with primary metric 'Accuracy'. The experiment_timeout_minutes is set already to '30 mins'.
n_cross_validations=3 parameter sets how many cross validations to perform, based on the same number of folds (number of subsets).

********************************************************************************************

 ITER   PIPELINE                                       DURATION            METRIC      BEST
    0   MaxAbsScaler LightGBM                          0:00:34             0.9139    0.9139
    1   MaxAbsScaler XGBoostClassifier                 0:00:36             0.9142    0.9142
    2   MaxAbsScaler ExtremeRandomTrees                0:00:34             0.7259    0.9142
    3   SparseNormalizer XGBoostClassifier             0:00:33             0.9135    0.9142
    4   MaxAbsScaler LightGBM                          0:00:30             0.9134    0.9142
    5   MaxAbsScaler LightGBM                          0:00:29             0.8880    0.9142
    6   StandardScalerWrapper XGBoostClassifier        0:00:30             0.9056    0.9142
    7   MaxAbsScaler LogisticRegression                0:00:32             0.9084    0.9142
    8   StandardScalerWrapper ExtremeRandomTrees       0:00:28             0.8895    0.9142
    9   StandardScalerWrapper XGBoostClassifier        0:00:30             0.9064    0.9142
   10   SparseNormalizer LightGBM                      0:00:28             0.9045    0.9142
   11   StandardScalerWrapper XGBoostClassifier        0:00:41             0.9127    0.9142
   12   MaxAbsScaler LogisticRegression                0:00:32             0.9087    0.9142
   13   MaxAbsScaler SGD                               0:00:29             0.8325    0.9142
   14   StandardScalerWrapper XGBoostClassifier        0:00:32             0.9144    0.9144
   15   SparseNormalizer RandomForest                  0:00:45             0.8161    0.9144
   16   StandardScalerWrapper LogisticRegression       0:00:32             0.9084    0.9144
   17   StandardScalerWrapper RandomForest             0:00:35             0.9008    0.9144
   18   StandardScalerWrapper XGBoostClassifier        0:00:34             0.9128    0.9144
   19   TruncatedSVDWrapper RandomForest               0:02:35             0.8153    0.9144
   20   TruncatedSVDWrapper RandomForest               0:03:19             0.8290    0.9144
   21   StandardScalerWrapper XGBoostClassifier        0:00:32             0.9125    0.9144
   22   MaxAbsScaler LightGBM                          0:00:30             0.8880    0.9144
   23   MaxAbsScaler LightGBM                          0:00:37             0.9088    0.9144
   24   MaxAbsScaler LightGBM                          0:00:30             0.9041    0.9144
   25   StandardScalerWrapper XGBoostClassifier        0:00:58             0.9128    0.9144
   26   StandardScalerWrapper XGBoostClassifier        0:02:25             0.9093    0.9144
   27   SparseNormalizer LightGBM                      0:00:29             0.9021    0.9144
   28   StandardScalerWrapper XGBoostClassifier        0:01:22             0.9110    0.9144
   29   VotingEnsemble                                 0:00:40             0.9176    0.9176
   30   StackEnsemble                                  0:00:50             0.9157    0.9176
Stopping criteria reached at iteration 31. Ending experiment.
********************************************************************************************

## Pipeline comparison
| HyperDrive Model | |
| :---: | :---: |
| id | HD_a28a0ad6-f34d-4e6a-894c-c2ea35f7c28f_10  |
| Accuracy | 0.91 |


| AutoML Model | |
| :---: | :---: |
| id | AutoML_e52b461c-fdd7-4ee9-9fb7-77cf2ed8818a_29 |
| Accuracy | 0.91757 |
| AUC_weighted | 0.94725 |
| Algortithm | VotingEnsemble |

## Future work
In the future it might be helpful to explore more feature engineering steps prior to training. Also, many of the AutoML runs use a scaler prior to model training and evaluation. Also Class imbalance is a very common issue in classification problems in machine learning. Imbalanced data negatively impact the model's accuracy because it is easy for the model to be very accurate just by predicting the majority class, while the accuracy for the minority class can fail miserably. Furthermore, exploring hyperdrive with a broader variety of classification models would also be informative.
