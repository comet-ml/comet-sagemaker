import os
import utils
import sagemaker

from sagemaker import get_execution_role
from sagemaker.tensorflow import TensorFlow


def main():
    sagemaker_session = sagemaker.Session()
    role = get_execution_role()

    utils.cifar10_download()
    inputs = sagemaker_session.upload_data(
        path='./cifar10_data', key_prefix='data/DEMO-cifar10')

    source_dir = os.path.join(os.getcwd(), 'source_dir')
    hyperparameters = {
        'throttle_secs': 30,
        'MOMENTUM': 0.9,
        'RESNET_SIZE': 32,
        'INITIAL_LEARNING_RATE': 0.05,
        'WEIGHT_DECAY': 2e-4,
        'BATCH_SIZE': 32,
        'NUM_DATA_BATCHES': 5
    }

    estimator = TensorFlow(
        entry_point='model.py',
        source_dir=source_dir,
        role=role,
        hyperparameters=hyperparameters,
        requirements_file='requirements.txt',
        training_steps=100,
        evaluation_steps=5,
        train_instance_count=1,
        framework_version='1.12',
        py_version='py3',
        train_instance_type='ml.c4.xlarge',
        base_job_name='comet-sagemaker')

    estimator.fit(inputs)


if __name__ == '__main__':
    main()
