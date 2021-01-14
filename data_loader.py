from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import torch
import numpy as np


class SensorDataset(Dataset):
    def __init__(self, manifest_path):
        with open(manifest_path, 'r') as m:
            ids = m.readlines()
        ids = [x.strip().split('\t') for x in ids]
        self.ids = ids
        self.size = len(ids)

    def __getitem__(self, idx):
        sample = self.ids[idx]
        ch_0, ch_1, ch_2, ch_3 = sample[1], sample[2], sample[3], sample[4]

        ch_0 = torch.FloatTensor(np.fromstring(ch_0, sep=' '))
        ch_1 = torch.FloatTensor(np.fromstring(ch_1, sep=' '))
        ch_2 = torch.FloatTensor(np.fromstring(ch_2, sep=' '))
        ch_3 = torch.FloatTensor(np.fromstring(ch_3, sep=' '))

        return ch_0, ch_1, ch_2, ch_3

    def __len__(self):
        return self.size

class SensorDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):

        """
        Creates a data loader for AudioDatasets.
        """
        super(SensorDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


def _collate_fn(batch):

    ch_0_len = len(batch[0][0])
    ch_1_len = len(batch[0][1])
    ch_2_len = len(batch[0][2])
    ch_3_len = len(batch[0][3])

    minibatch_size = len(batch)
    ch0s = torch.zeros(minibatch_size, ch_0_len)
    ch1s = torch.zeros(minibatch_size, ch_1_len)
    ch2s = torch.zeros(minibatch_size, ch_2_len)
    ch3s = torch.zeros(minibatch_size, ch_3_len)

    for i in range(minibatch_size):
        sample = batch[i]
        ch0 = sample[0]
        ch1 = sample[1]
        ch2 = sample[2]
        ch3 = sample[3]

        ch0s[i].copy_(ch0)
        ch1s[i].copy_(ch1)
        ch2s[i].copy_(ch2)
        ch3s[i].copy_(ch3)

    return ch0s, ch1s, ch2s, ch3s

class BucketingSampler(Sampler):
    def __init__(self, data_source, batch_size=-1):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        """
        if batch_size == -1:
            batch_size = len(data_source)
        super(BucketingSampler, self).__init__(data_source)
        self.data_source = data_source
        ids = list(range(0, len(data_source)))
        self.bins = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]

    def __iter__(self):
        for ids in self.bins:
            np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

    def shuffle(self, epoch):
        np.random.shuffle(self.bins)

