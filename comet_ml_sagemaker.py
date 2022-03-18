# -*- coding: utf-8 -*-
# Copyright 2019 Boris Feld <boris@comet.ml>

"""Top-level package for Comet.ml Sagemaker."""

__author__ = """Boris Feld"""
__email__ = "boris@comet.ml"
__version__ = "0.1.0"

import calendar
import collections

import boto3
from comet_ml.api import APIExperiment
from sagemaker.analytics import TrainingJobAnalytics


def _get_boto_client():
    return boto3.client("sagemaker")


def _flatten(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(_flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def log_last_sagemaker_job(api_key=None, workspace=None, project_name=None):
    client = _get_boto_client()
    last_name = client.list_training_jobs()["TrainingJobSummaries"][0][
        "TrainingJobName"
    ]
    return log_sagemaker_job_by_name(
        last_name, api_key=api_key, workspace=workspace, project_name=project_name
    )


def log_sagemaker_job(estimator, api_key=None, workspace=None, project_name=None):
    # Retrieve the training job name from the estimator
    if not hasattr(estimator, "latest_training_job"):
        raise ValueError("log_sagemaker_job expect a sagemaker Estimator object")

    if estimator.latest_training_job is None:
        raise ValueError(
            "The given Estimator object doesn't seems to have trained a model, call log_sagemaker_job after calling the fit method"
        )

    return log_sagemaker_job_by_name(
        estimator.latest_training_job.job_name,
        api_key=api_key,
        workspace=workspace,
        project_name=project_name,
    )


def log_sagemaker_job_by_name(
    sagemaker_job_name, api_key=None, workspace=None, project_name=None
):
    # Metadata
    client = _get_boto_client()
    metadata = client.describe_training_job(TrainingJobName=sagemaker_job_name)

    if metadata["TrainingJobStatus"] != "Completed":
        raise ValueError(
            "Not importing %r as it's not completed, status %r"
            % (sagemaker_job_name, metadata["TrainingJobStatus"])
        )

    experiment = APIExperiment(
        api_key=api_key,
        workspace=workspace,
        project_name=project_name,
        experiment_name=sagemaker_job_name,
    )
    start_time = metadata["TrainingStartTime"]
    start_time_timestamp = calendar.timegm(start_time.utctimetuple())
    experiment.set_start_time(start_time_timestamp * 1000)
    end_time = metadata.get("TrainingEndTime")
    if end_time:
        experiment.set_end_time(calendar.timegm(end_time.utctimetuple()) * 1000)

    for param_name, param_value in metadata["HyperParameters"].items():
        experiment.log_parameter(param_name, param_value)

    other_list = [
        "BillableTimeInSeconds",
        "EnableInterContainerTrafficEncryption",
        "EnableManagedSpotTraining",
        "EnableNetworkIsolation",
        "RoleArn",
        "TrainingJobArn",
        "TrainingJobName",
        "TrainingJobStatus",
        "TrainingTimeInSeconds",
    ]
    for other_name in other_list:
        other_value = metadata.get(other_name)
        if other_value:
            experiment.log_other(other_name, other_value)

    experiment.log_other(
        "TrainingImage", metadata["AlgorithmSpecification"]["TrainingImage"]
    )
    experiment.log_other(
        "TrainingInputMode", metadata["AlgorithmSpecification"]["TrainingInputMode"]
    )

    for other_key, other_value in _flatten(
        metadata.get("ModelArtifacts", {}), "ModelArtifacts"
    ).items():
        experiment.log_other(other_key, other_value)

    for other_key, other_value in _flatten(
        metadata["OutputDataConfig"], "OutputDataConfig"
    ).items():
        experiment.log_other(other_key, other_value)

    for other_key, other_value in _flatten(
        metadata["ResourceConfig"], "ResourceConfig"
    ).items():
        experiment.log_other(other_key, other_value)

    for i, _input in enumerate(metadata["InputDataConfig"]):
        for other_key, other_value in _flatten(
            _input, "InputDataConfig.%d" % i
        ).items():
            experiment.log_other(other_key, other_value)

    response = client.list_tags(ResourceArn=metadata["TrainingJobArn"])
    for tag_name, tag_value in response["Tags"]:
        experiment.add_tags(["%s:%s" % (tag_name, tag_value)])
    # Metrics
    metrics_dataframe = TrainingJobAnalytics(
        training_job_name=sagemaker_job_name
    ).dataframe()

    for iloc, (timestamp, metric_name, value) in metrics_dataframe.iterrows():
        print("TS", start_time_timestamp + timestamp)
        experiment.log_metric(
            metric=metric_name, value=value, timestamp=start_time_timestamp + timestamp
        )

    return experiment