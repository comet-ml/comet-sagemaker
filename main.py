import argparse
import boto3

import sagemaker
from sagemaker import get_execution_role
from sagemaker.estimator import Estimator

DATASET_PREFIX = 'DEMO-comet-sagemaker-cifar10-example'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    parser.add_argument('--container_name')

    return parser.parse_args()


def main():
    args = get_args()

    sess = sagemaker.Session()
    role = get_execution_role()

    client = boto3.client('sts')
    account = client.get_caller_identity()['Account']

    my_session = boto3.session.Session()
    region = my_session.region_name

    container_name = args.container_name
    ecr_image = '{}.dkr.ecr.{}.amazonaws.com/{}:latest'.format(
        account, region, container_name)

    inputs = sess.upload_data(
        path=args.data, key_prefix=DATASET_PREFIX)

    hyperparameters = {
        'train-steps': 1000
    }
    instance_type = 'ml.m4.xlarge'
    estimator = Estimator(role=role,
                          hyperparameters=hyperparameters,
                          train_instance_count=1,
                          train_instance_type=instance_type,
                          image_name=ecr_image)
    estimator.fit(inputs)


if __name__ == '__main__':
    main()
