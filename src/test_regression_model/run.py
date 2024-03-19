#!/usr/bin/env python
"""
Test the random forest regressor
"""
import argparse
import logging
import wandb
import mlflow
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="test_regression_model")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    ######################
    # YOUR CODE HERE     #
    ######################
    logger.info("using model in W&B: %s", args.mlflow_model)
    model_local_path = run.use_artifact(
        args.mlflow_model, 
        type="model_export"
    ).download() 

    logger.info("the model will be evaluated against: %s", args.test_artifact)
    test_local_path = run.use_artifact(
        args.test_artifact, 
        type="split_data"
    ).file() 
    
    logger.info("reading model from local path %s", model_local_path)
    logger.info("reading model from local path is of type %s", type(model_local_path))
    pipe = mlflow.sklearn.load_model(model_local_path)
    df = pd.read_csv(test_local_path)

    # Extract the target from the features
    logger.info("Extracting target from dataframe")
    X_test = df.copy()
    y_test = X_test.pop("price")

    #used_columns = list(itertools.chain.from_iterable([x[2] for x in pipe['preprocessor'].transformers]))
    #pred_proba = pipe.predict_proba(X_test[used_columns])
    y_pred = pipe.predict(X_test)
    logger.info("Scoring")

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    run.summary["MAE"] = mae
    run.summary["MSE"] = mse
    run.log(
        {
            "MAE": mae,
            "MSE": mse,
        }
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test the new random forest regressor")


    parser.add_argument(
        "--mlflow_model", 
        type=str,
        help="Trained random forest model marked for production",
        required=True
    )

    parser.add_argument(
        "--test_artifact", 
        type=str,
        help="Test data to perform a test on the production model",
        required=True
    )


    args = parser.parse_args()

    go(args)
