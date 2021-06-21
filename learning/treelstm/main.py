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
import Constants
# NEURAL NETWORK MODULES/LAYERS
from model import *
# DATA HANDLING CLASSES
from tree import Tree
from vocab import Vocab
# DATASET CLASS FOR SICK DATASET
from dataset import QGDataset
# METRICS CLASS FOR EVALUATION
from metrics import Metrics
# UTILITY FUNCTIONS
from utils import load_word_vectors, build_vocab
# CONFIG PARSER
from learning.treelstm.config import parse_args
# TRAIN AND TEST HELPER FUNCTIONS
from trainer import Trainer
import datetime


from learning.treelstm.model import DASimilarity, SimilarityTreeLSTM




def main():
    global args
    args = parse_args()
    # global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
    # file logger
    #fh = logging.FileHandler(os.path.join(args.save, args.expname) + '.log', mode='w')
   # fh.setLevel(logging.INFO)
    #fh.setFormatter(formatter)
  #  logger.addHandler(fh)
    # console logger
    #ch = logging.StreamHandler()
   # ch.setLevel(logging.DEBUG)
   # ch.setFormatter(formatter)
    # argument validation
    #args.cuda = args.cuda and torch.cuda.is_available()

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

    train_dir = os.path.join(args.data, 'train/')
    train_dir = "D:/downloads/QA/"+train_dir
    train_dir = "D:\\Academics\\thesis\\codeBERT\\BERT-QA\\learning\\treelstm\\data\\qald\\train"
    dev_dir = os.path.join(args.data, 'dev/')
    dev_dir = "D:/downloads/QA/"+dev_dir
    test_dir = os.path.join(args.data, 'test/')
    test_dir = "D:/downloads/QA/"+test_dir
    one_dir = os.path.join(args.data, 'one/')
    one_dir = "D:/downloads/QA/" + one_dir

    # write unique words from all token files

    #dataset_vocab_file = os.path.join(args.data, 'dataset.vocab')
    #dataset_vocab_file = "D:/downloads/QA/learning/treelstm/data/lc_quad/dataset.vocab"
    #if not os.path.isfile(dataset_vocab_file):
        #token_files_a = [os.path.join(split, 'a.toks') for split in [train_dir, dev_dir, test_dir]]
        #token_files_b = [os.path.join(split, 'b.toks') for split in [train_dir, dev_dir, test_dir]]
        #token_files = token_files_a + token_files_b
        #dataset_vocab_file = "D:/downloads/QA/learning/treelstm/data/lc_quad2/dataset.vocab"
        #build_vocab(token_files, dataset_vocab_file)

    # get vocab object from vocab file previously written
    vocab = Vocab(filename="D:/downloads/QA/learning/treelstm/data/lc_quad/dataset.vocab",
                  data=[Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD])
    logger.debug('==> Dataset vocabulary size : %d ' % vocab.size())

    # load dataset splits
    #train_file = os.path.join(args.data, 'dataset_train.pth')
    train_file = "D:\\Academics\\thesis\\codeBERT\\BERT-QA\\learning\\treelstm\\data\\qald\\dataset_train_qald.pth"
    if os.path.isfile(train_file):
        train_dataset = torch.load(train_file)
    else:
        train_dataset = QGDataset(train_dir, args.num_classes)
        torch.save(train_dataset, train_file)
    logger.debug('==> Size of train data   : %d ' % len(train_dataset))
    '''
    dev_file = os.path.join(args.data, 'dataset_dev.pth')
    dev_file = "D:/downloads/QA/"+dev_file
    if os.path.isfile(dev_file):
        dev_dataset = torch.load(dev_file)
    else:
        dev_dataset = QGDataset(dev_dir, args.num_classes)
        torch.save(dev_dataset, dev_file)
    logger.debug('==> Size of dev data     : %d ' % len(dev_dataset))
    test_file = os.path.join(args.data, 'dataset_test.pth')
    test_file = "D:/downloads/QA/"+test_file
    if os.path.isfile(test_file):
        test_dataset = torch.load(test_file)
    else:
        test_dataset = QGDataset(test_dir, args.num_classes)
        torch.save(test_dataset, test_file)
    logger.debug('==> Size of test data    : %d ' % len(test_dataset))

    #one_dataset = QGDataset("D:\\downloads\\QA\\output\\qald\\", vocab, args.num_classes)
    '''
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
    metrics = Metrics(args.num_classes)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.25)

    '''
    # for words common to dataset vocab and GLOVE, use GLOVE vectors
    # for other words in dataset vocab, use random normal vectors
    emb_file = os.path.join(args.data, 'dataset_embed.pth')
    emb_file = "D:/downloads/QA/"+emb_file
    if os.path.isfile(emb_file):
        emb = torch.load(emb_file)
    else:
       # EMBEDDING_DIM = 300
       # emb = torch.zeros(vocab.size(), EMBEDDING_DIM, dtype=torch.float)
       # fasttext_model = load_model("data/fasttext/wiki.en.bin")
       #  print('Use Fasttext Embedding')
       # for word in vocab.labelToIdx.keys():
       #      word_vector = fasttext_model.get_word_vector(word)
       #    if word_vector.all() != None and len(word_vector) == EMBEDDING_DIM:
       #        emb[vocab.getIndex(word)] = torch.Tensor(word_vector)
       #    else:
       #        emb[vocab.getIndex(word)] = torch.Tensor(EMBEDDING_DIM).uniform_(-1, 1)
        # # load glove embeddings and vocab
         args.glove = 'learning/treelstm/data/glove/'
         print('Use Glove Embedding')

         pathglove = "D:/downloads/QA/"+os.path.join(args.glove, 'glove.840B.300d')
         glove_vocab, glove_emb = load_word_vectors(pathglove)
         logger.debug('==> GLOVE vocabulary size: %d ' % glove_vocab.size())
         emb = torch.Tensor(vocab.size(), glove_emb.size(1)).normal_(-0.05, 0.05)
        # # zero out the embeddings for padding and other special words if they are absent in vocab
         for idx, item in enumerate([Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD]):
             emb[idx].zero_()
         for word in vocab.labelToIdx.keys():
             if glove_vocab.getIndex(word):
                 emb[vocab.getIndex(word)] = glove_emb[glove_vocab.getIndex(word)]
         torch.save(emb, emb_file)
    # plug these into embedding matrix inside model
    if args.cuda:
        emb = emb.cuda()
    model.emb.weight.data.copy_(emb)
    '''

    checkpoint_filename = '%s.pt' % os.path.join(args.save, args.expname)
    checkpoint_filename = "D:\\downloads\\QA\\learning\\treelstm\\learning\\treelstm\\checkpoints\\lc_quad,epoch=15,train_loss=0.2348909229040146.pt"

    if args.mode == "test":
        checkpoint = torch.load(checkpoint_filename)
        model.load_state_dict(checkpoint['model'])
        args.epochs = 1

    # create trainer object for training and testing
    trainer = Trainer(args, model, criterion, optimizer)
    # loss, dev_pred = trainer.test(one_dataset)
    # x = 0



    for epoch in range(args.epochs):
        if args.mode == "train":
            scheduler.step()

            train_loss = trainer.train(train_dataset)
            #train_loss, train_pred = trainer.test(train_dataset)
            #logger.info(
            #    '==> Epoch {}, Train \tLoss: {} {}'.format(epoch, train_loss,
            #                                               metrics.all(train_pred, train_dataset.labels)))
            checkpoint = {'model': trainer.model.state_dict(), 'optim': trainer.optimizer,
                          'args': args, 'epoch': epoch, 'scheduler': scheduler}
            checkpoint_filename = '%s.pt' % os.path.join(args.save,
                                                         args.expname + ',epoch={},train_loss={}'.format(epoch + 1,
                                                                                                       train_loss))
            #checkpoint_filename = "D:/Academics/thesis/QAsparql-master/"+checkpoint_filename
            torch.save(checkpoint, checkpoint_filename)

        #dev_loss, dev_pred = trainer.test(dev_dataset)
        #test_loss, test_pred = trainer.test(test_dataset)
        #logger.info(
        #    '==> Epoch {}, Dev \tLoss: {} {}'.format(epoch, dev_loss, metrics.all(dev_pred, dev_dataset.labels)))
        #logger.info(
        #    '==> Epoch {}, Test \tLoss: {} {}'.format(epoch, test_loss, metrics.all(test_pred, test_dataset.labels)))


if __name__ == "__main__":
    main()
