from __future__ import division
from __future__ import print_function

import os
import random
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable as Var

import sys
# IMPORT CONSTANTS
from learning.treelstm.config import parse_args
from learning.treelstm.dataset import QGDataset
from learning.treelstm.model import DASimilarity, SimilarityTreeLSTM
from learning.treelstm.trainer import Trainer
from learning.treelstm.vocab import Vocab
import learning.treelstm.Constants as Constants

def testmain(one_dataset):
    global args
    args = parse_args()
    # global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    if args.sparse and args.wd != 0:
        logger.error('Sparsity and weight decay are incompatible, pick one!')
        exit()
    logger.debug(args)
    args.data = 'learning/treelstm/data/lc_quad/'
    args.save = 'learning/treelstm/checkpoints/'
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    dataset_vocab_file = "D:/downloads/QA/learning/treelstm/data/lc_quad/dataset.vocab"


    vocab = Vocab(filename=dataset_vocab_file,
                  data=[Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD])


    similarity = DASimilarity(args.mem_dim, args.hidden_dim, args.num_classes)
    # if args.sim == "cos":
    #     similarity = CosSimilarity(1)
    # else:
    #     similarity = DASimilarity(args.mem_dim, args.hidden_dim, args.num_classes, dropout=True)

    # initialize model, criterion/loss_function, optimizer
    model = SimilarityTreeLSTM(
        vocab.size(),
        args.input_dim,
        args.mem_dim,
        similarity,
        args.sparse)
    criterion = nn.KLDivLoss()  # nn.HingeEmbeddingLoss()

    if args.cuda:
        model.cuda(), criterion.cuda()
    else:
        torch.set_num_threads(4)
    logger.info("number of available cores: {}".format(torch.get_num_threads()))
    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)

    checkpoint_filename = "D:\\downloads\\QA\\learning\\treelstm\\learning\\treelstm\\checkpoints\\lc_quad,epoch=15,train_loss=0.2348909229040146.pt"


    checkpoint = torch.load(checkpoint_filename)
    model.load_state_dict(checkpoint['model'])
    args.epochs = 1

    # create trainer object for training and testing
    trainer = Trainer(args, model, criterion, optimizer)
    loss, dev_pred = trainer.test(one_dataset)
    return loss,dev_pred


if __name__ == "__main__":
    testmain()
