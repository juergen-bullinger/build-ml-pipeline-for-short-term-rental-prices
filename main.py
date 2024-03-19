"""
Full pipeline to train a RF estimator to predict short term rental prices

Author: Jürgen Bullinger
Date 11.03.2024
"""

import json

import os
import tempfile
import logging
import mlflow
import wandb
import hydra
from omegaconf import DictConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # NOTE: We do not include this in the steps so it is not run by mistake.
    # You first need to promote a model export to "prod" before you can run this,
    # then you need to run this step explicitly
#    "test_regression_model"
]


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):
    """
    Do the processing - this function contains the full logic to execute
    the pipeline.

    Parameters
    ----------
    config : DictConfig
        Automatically passed by hydra.

    Returns
    -------
    None.

    """
    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # get the path at the root of the MLflow project
    root_path = hydra.utils.get_original_cwd()

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        components_repository = config['main']['components_repository']
        
        if "download" in active_steps:
            # Download file and load in W&B
            logger.info("starting download step...")
            _ = mlflow.run(
                f"{components_repository}/get_data",
                "main",
                version='main',
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )

        if "eda" in active_steps:
            ##################
            # Implement here #
            ##################
            logger.info("starting eda step...")
            _ = mlflow.run(
                os.path.join(root_path, "src", "eda"),
                "main",
                #version='main',
                parameters={},
            )
            
        if "basic_cleaning" in active_steps:
            ##################
            # Implement here #
            ##################
            logger.info("starting basic_cleaning step...")
            _ = mlflow.run(
                os.path.join(root_path, "src", "basic_cleaning"),
                "main",
                #version='main',
                parameters={
                    "input_artifact": "sample.csv",
                    "output_artifact": "clean_sample.csv",
                    "output_type": "clean_data",
                    "output_description": "split data",
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"],
                },
            )

        if "data_check" in active_steps:
            ##################
            # Implement here #
            ##################
            logger.info("starting data_check step...")
            _ = mlflow.run(
                os.path.join(root_path, "src", "data_check"),
                "main",
                #version='main',
                parameters={
                    "csv": "clean_sample.csv", # ???
                    "ref": "sample.csv", # ???
                    "kl_threshold": config["data_check"]["kl_threshold"],
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"]
                },
            )

        if "data_split" in active_steps:
            ##################
            # Implement here #
            ##################
            logger.info("starting data_split step...")
            #_ = mlflow.run(
            #    os.path.join(root_path, "src", "data_split"),
            #    "main",
            #    #version='main',
            #    parameters={
            #        "input_artifact": "sample.csv", # TODO: replace this
            #        "test_size": config["modeling"]["test_size"],
            #        "val_size": config["modeling"]["val_size"]
            #    },
            #)
            _ = mlflow.run(
                f"{components_repository}/train_val_test_split",
                "main",
                version='main',
                parameters={
                    "input_artifact": "sample.csv", # TODO: replace this
                    "test_size": config["modeling"]["test_size"],
                    "val_size": config["modeling"]["val_size"]
                },
            )

        if "train_random_forest" in active_steps:

            # NOTE: we need to serialize the random forest configuration into JSON
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)  # DO NOT TOUCH

            # NOTE: use the rf_config we just created as the rf_config 
            # parameter for the train_random_forest step

            ##################
            # Implement here #
            ##################
            logger.info("starting train_random_forest step...")
            rf_params = {
            }
            rf_params.update()
            _ = mlflow.run(config["modelling"],
                os.path.join(root_path, "src", "train_random_forest"),
                "main",
                #version='main',
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )

        if "test_regression_model" in active_steps:

            ##################
            # Implement here #
            ##################
            logger.info("starting test_regression_model step...")
            _ = mlflow.run(
                os.path.join(root_path, "src", "test_regression_model"),
                "main",
                #version='main',
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )


if __name__ == "__main__":
    go()
