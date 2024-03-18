#!/usr/bin/env python
"""
Split the data in train/test
"""
import argparse
import logging
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="data_split")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    artifact_local_path = run.use_artifact(args.input_artifact).file()
    # TODO: to be continued


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Split the data in train/test")

    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="the input artifact which needs to be split",
        required=True
    )

    parser.add_argument(
        "--test_size", 
        type=float,
        help="fraction of the input artifact to be used as test data",
        required=True
    )

    parser.add_argument(
        "--val_size", 
        type="flaot",
        help="fraction of the input artifact to be used as valid. data",
        required=True
    )

    args = parser.parse_args()

    go(args)
