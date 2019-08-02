import boto3

import sagemaker
from sagemaker import get_execution_role
from sagemaker.estimator import Estimator

DATASET_PREFIX = 'DEMO-comet-sagemaker-cifar10-example'
WORK_DIRECTORY = '/tmp/cifar10'


def setup_dataset():
    sess = sagemaker.Session()

    return sess.upload_data(WORK_DIRECTORY, key_prefix=DATASET_PREFIX)


def main():
    role = get_execution_role()

    client = boto3.client('sts')
    account = client.get_caller_identity()['Account']

    my_session = boto3.session.Session()
    region = my_session.region_name

    algorithm_name = 'comet-sagemaker-cifar10'
    ecr_image = '{}.dkr.ecr.{}.amazonaws.com/{}:latest'.format(
        account, region, algorithm_name)

    # you mau use other bucket than the default, one more time
    # adapt the code to your needs
    data_location = 's3://{}/{}'.format(
        'arn:aws:s3:::sagemaker-us-east-2-094792403439',
        DATASET_PREFIX)

    hyperparameters = {
        'train-steps': 100
    }
    #instance_type = 'ml.m4.xlarge'
    instance_type = 'local'
    estimator = Estimator(role=role,
                          train_instance_count=1,
                          train_instance_type=instance_type,
                          image_name=ecr_image)
    estimator.fit(data_location)
