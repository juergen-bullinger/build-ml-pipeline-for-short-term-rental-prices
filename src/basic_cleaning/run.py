#!/usr/bin/env python
"""
Performs basic cleaning on the data and save the results in Weights & Biases
"""
import argparse
import logging
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    artifact_local_path = run.use_artifact(args.input_artifact).file()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This steps cleans the data")


    parser.add_argument(
        "--input_artifact", 
        type="string",
        help="input_artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type="string",
        help="output_artifact",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type="string",
        help="output_type",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type="string",
        help="output_description",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type="float",
        help="min value for the price",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type="float",
        help="max value for the price",
        required=True
    )

    args = parser.parse_args()

    go(args)
