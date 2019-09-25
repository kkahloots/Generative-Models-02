import sys
sys.path.append('..')

import random

import numpy as np
import dask.array as da
from dask_ml.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
from skimage.color import grey2rgb

from utils.dataset import Dataset
from utils.configuration import default_config as config


'''  ------------------------------------------------------------------------------
                                    DATA METHODS
 ------------------------------------------------------------------------------ '''
scalar = None

def prepare_dataset(X):
    len_ = X.shape[0]
    shape_ = X.shape

    d = int(np.sqrt(X.flatten().reshape(X.shape[0], -1).shape[1]))

    if len(shape_) == 4:
        d = int(np.sqrt(X.flatten().reshape(X.shape[0], -1).shape[1] / 3))
        X = np.reshape(X, [-1, d, d, 3])

    elif d == shape_[1] and len(shape_) == 3:
        X = np.array(list(map(lambda x: grey2rgb(x), X)), dtype=np.float32)
        X = np.reshape(X, [-1, d, d, 3])

    else:
        r = d**2 - X.shape[1]
        train_padding = np.zeros((shape_[0], r))
        X = np.vstack([X, train_padding])

        X = np.reshape(X, [-1, d, d])
        X = np.array(list(map(lambda x: grey2rgb(x), X)), dtype=np.float32)

    print('Scaling dataset ... ')
    if scalar is not None:
        X = scaler.transform(X.flatten().reshape(-1, 1).astype(np.float32)).reshape(X.shape)
    else:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X.flatten().reshape(-1, 1).astype(np.float32)).reshape(X.shape)
    print('Creating dask array ... ')
    return da.from_array(X, chunks=100)#da.array(X)#

def process_data(X, y=None, test_size=0.2):
    if y is None:
        km = dask_ml.cluster.KMeans(n_clusters=10, init_max_iter=100)
        km.fit(X.flatten().reshape(-1, 1))
        y = km.labels_
    y_uniqs = np.unique(y[:,0])

    len_ = X.shape[0]
    X = prepare_dataset(X)

    shape_ = list(X.shape[1:])

    if test_size != 0:
        samples = list()
        samples_labels = list()
        print('Preparing samples ...')
        for _ in range(2):
            for y_uniq in y_uniqs:
                sample = list()
                label = list()
                for xa, ya in zip(chunks(X, 10),chunks(y[:,0], 10)):
                    try:
                        sample.append([xa[ya == y_uniq][random.randint(0, len(xa[ya == y_uniq]) - 1)]])
                        label.append(y_uniq)
                        if len(sample) >= len(y_uniqs):
                            break
                    except:
                        pass
                samples += sample
                samples_labels += label
        samples = da.vstack(samples)
        samples_labels = da.vstack(samples_labels)

    if test_size == 0:
        print('Training dataset shape x: ', X.shape)
        print('Training dataset shape y: ', y.shape)

        train_dataset = Dataset(X, y)
        return train_dataset

    else:
        X_train, X_test, y_train, y_test = train_test_split(X.flatten().reshape(len_, -1), y, test_size=test_size,
                                                        random_state=config.seeds)

        X_train = X_train.reshape([X_train.shape[0]] + shape_)
        X_test = X_test.reshape([X_test.shape[0]] + shape_)

        print('Training dataset shape: ', X_train.shape)
        print('Validation dataset shape: ', X_test.shape)

        train_dataset = Dataset(X_train, y_train)
        test_dataset = Dataset(X_test, y_test)

        train_dataset.samples = samples
        train_dataset.samples_labels = samples_labels

        print('Sample dataset shape: ', train_dataset.samples.shape)
        return train_dataset, test_dataset


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]




