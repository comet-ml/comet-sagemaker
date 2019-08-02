import os
import sys
import tarfile
from six.moves import urllib

DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def cifar10_download(data_dir='./cifar10_data'):
    """Download and extract the tarball from Alex's website."""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if os.path.exists(os.path.join(data_dir, 'cifar-10-batches-bin')):
        print('cifar dataset already downloaded')
        return

    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        sys.stdout.write('\r>> Downloading %s ' % (filename))

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

    tarfile.open(filepath, 'r:gz').extractall(data_dir)
