import os
import json
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.utils.data as data

import learning.treelstm.Constants as Constants
from learning.treelstm.tree import Tree
from learning.treelstm.vocab import Vocab


class QGDataset(data.Dataset):
    def __init__(self, path, num_classes):
        super(QGDataset, self).__init__()
        self.num_classes = num_classes

        self.lsentences = self.read_sentences(os.path.join(path, 'a.txt'))
        self.rsentences = self.read_sentences(os.path.join(path, 'b.txt'))


        self.labels = self.read_labels(os.path.join(path, 'sim.txt'))

        self.size = len(self.lsentences)

    def __len__(self):
        return self.size

    def __getitem__(self, index):


        lsent = deepcopy(self.lsentences[index])
        rsent = deepcopy(self.rsentences[index])
        label = deepcopy(self.labels[index])
        return ( lsent, rsent, label)

    def read_sentences(self, filename):
        with open(filename, 'r') as f:
            sentences = [self.read_sentence(line) for line in tqdm(f.readlines())]
        return sentences

    def read_sentence(self, line):
        #indices = self.vocab.convertToIdx(line.split(), Constants.UNK_WORD)


        #return torch.LongTensor(indices)
        return line



    def read_labels(self, filename):
        with open(filename, 'r') as f:
            labels = list(map(lambda x: float(x), f.readlines()))
            labels = torch.Tensor(labels)
        return labels
