from tqdm import tqdm

import torch
from torch.autograd import Variable as Var

from learning.treelstm.utils import map_label_to_target
from bertclass import BertClass

class Trainer(object):
    def __init__(self, args, model, criterion, optimizer):
        super(Trainer, self).__init__()
        self.args = args
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epoch = 0

    # helper function for training
    def train(self, dataset):
        self.model.train()
        self.optimizer.zero_grad()
        loss, k = 0.0, 0
        indices = torch.randperm(len(dataset))
        bertclass = BertClass()
        for idx in tqdm(range(len(dataset)), desc='Training epoch ' + str(self.epoch + 1) + ''):
            lsent, rsent, label = dataset[indices[idx]]
            #linput, rinput = Var(lsent), Var(rsent)


            #if self.args.cuda:
                #linput, rinput = linput.cuda(), rinput.cuda()
                #target = target.cuda()
            output = self.model(lsent, rsent, bertclass)

            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            target = Var(map_label_to_target(label, dataset.num_classes))

            err = self.criterion(output, target)
            loss += err.data
            err.backward()
            k += 1
            if k % self.args.batchsize == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        self.epoch += 1
        return loss / len(dataset)

    # helper function for testing
    def test(self, dataset):
        self.model.eval()
        loss = 0
        predictions = torch.zeros(len(dataset))
        #torch.set_default_tensor_type(torch.cuda.FloatTensor)
        indices = torch.arange(1, dataset.num_classes + 1, dtype=torch.float)
        bertclass = BertClass()
        for idx in tqdm(range(len(dataset)), desc='Testing epoch  ' + str(self.epoch) + ''):
            lsent,rsent, label = dataset[idx]
            #linput, rinput = Var(lsent, volatile=True), Var(rsent, volatile=True)
            #target = Var(map_label_to_target(label, dataset.num_classes), volatile=True)
            #if self.args.cuda:
                #linput, rinput = linput.cuda(), rinput.cuda()
                #target = target.cuda()



            output = self.model(lsent,rsent, bertclass)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            target = Var(map_label_to_target(label, dataset.num_classes))
            err = self.criterion(output, target)
            loss += err.data
            output = output.data.squeeze()

            indices = indices.cuda()
            predictions[idx] = torch.dot(indices, torch.exp(output))
        return loss / len(dataset), predictions
