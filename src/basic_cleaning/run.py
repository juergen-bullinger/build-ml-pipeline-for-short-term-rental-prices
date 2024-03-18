#!/usr/bin/env python
"""
Performs basic cleaning on the data and save the results in Weights & Biases
"""
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(project="nyc_airbnb", job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()
    input_artifact_name = args.input_artifact
    if ":" not in input_artifact_name:
        # if the version tag is not provided, use :latest
        input_artifact_name = f"{input_artifact_name}:latest"
    logger.info("using artifact %s", input_artifact_name)

    # load the source df from w&b
    run = wandb.init(project="nyc_airbnb", group="eda")
    local_path = wandb.use_artifact(input_artifact_name).file()
    df = pd.read_csv(local_path)
    
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()
    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])

    # writing the result to a new file    
    df.to_csv("clean_sample.csv", index=False)
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This steps cleans the data")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="input_artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="output_artifact",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="output_type",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="output_description",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="min value for the price",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="max value for the price",
        required=True
    )

    args = parser.parse_args()

    go(args)
