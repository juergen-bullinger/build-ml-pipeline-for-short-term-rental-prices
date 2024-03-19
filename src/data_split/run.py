#!/usr/bin/env python
"""
Split the data in train/test
"""
import os
import argparse
import logging
import wandb
import pandas as pd
from sklearn.model_selection import train_test_split
import tempfile


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def _get_versioned_artifact(artifact):
    """
    append :latest if artifact does not contain a version reference

    Parameters
    ----------
    artifact : str
        artifact spec.

    Returns
    -------
    the versioned artifact reference.
    """
    if ":" not in artifact:
        return f"{artifact}:latest"
    else:
        return artifact



def go(args):

    run = wandb.init(job_type="data_split")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    input_artifact_name = _get_versioned_artifact(args.input_artifact)
    artifact_local_path = run.use_artifact(input_artifact_name).file()
    df = pd.read_csv(artifact_local_path)
    splits = {}
    splits["trainval"], splits["test"] = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df[args.stratify] if args.stratify_by != 'null' else None,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        for split, df in splits.items():

            # Make the artifact name from the provided root plus the name of the split
            artifact_name = f"{split}_data.csv"
    
            # Get the path on disk within the temp directory
            temp_path = os.path.join(tmp_dir, artifact_name)
    
            logger.info(f"Uploading the {split} dataset to {artifact_name}")
    
            # Save then upload to W&B
            df.to_csv(temp_path)
    
            artifact = wandb.Artifact(
                name=artifact_name,
                type="split_data",
                description=f"{split} split of dataset {args.input_artifact}",
            )
            artifact.add_file(temp_path)
    
            logger.info("Logging artifact")
            run.log_artifact(artifact)

        # This waits for the artifact to be uploaded to W&B. If you
        # do not add this, the temp directory might be removed before
        # W&B had a chance to upload the datasets, and the upload
        # might fail
        artifact.wait()


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
        type=float,
        help="fraction of the input artifact to be used as valid. data",
        required=True
    )

    parser.add_argument(
        "--random_seed", 
        type=int,
        help="seed for the random number generator",
        required=True
    )

    parser.add_argument(
        "--stratify_by", 
        type=str,
        help="name of the column by which the split is statified",
        required=True
    )

    args = parser.parse_args()

    go(args)
