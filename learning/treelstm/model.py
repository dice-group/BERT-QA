import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var
import sys


path = os.getcwd()
print('path: ', path)
sys.path.insert(0, path)
import learning.treelstm.Constants as Constants
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
from transformers import BertTokenizer, BertModel
from torch import device as device_
# sys.path.insert(0,'/cluster/home/xlig/qg')



# module for childsumtreelstm
class ChildSumTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim):
        super(ChildSumTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)

    def node_forward(self, inputs, child_c, child_h):
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)
        #print(child_h_sum.shape)
        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        #print(iou.shape)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        #print(i.shape)
        #print(o.shape)
        #print(u.shape)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)
        #print(i.shape)
        #print(o.shape)
        #print(u.shape)

        f = F.sigmoid(
            self.fh(child_h) +
            self.fx(inputs).repeat(len(child_h), 1)
        )
        #print(f.shape)

        fc = torch.mul(f, child_c)
        #print(fc.shape)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        #print(c.shape)
        h = torch.mul(o, F.tanh(c))
        #print(h.shape)
        return c, h

    def forward(self, tree, inputs):
        _ = [self.forward(tree.children[idx], inputs) for idx in range(tree.num_children)]

        if tree.num_children == 0:
            child_c = Var(inputs[0].data.new(1, self.mem_dim).fill_(0.))
            child_h = Var(inputs[0].data.new(1, self.mem_dim).fill_(0.))
        else:
            child_c, child_h = zip(*map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)

        tree.state = self.node_forward(inputs[tree.idx], child_c, child_h)
        return tree.state


# module for distance-angle similarity
class DASimilarity(nn.Module):
    def __init__(self, mem_dim, hidden_dim, num_classes):
        super(DASimilarity, self).__init__()
        self.mem_dim = mem_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.wh = nn.Linear(2 * self.mem_dim, self.hidden_dim)
        self.wp = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, lvec, rvec):
        mult_dist = torch.mul(lvec, rvec)
        abs_dist = torch.abs(torch.add(lvec, -rvec))
        vec_dist = torch.cat((mult_dist, abs_dist), 1)

        vec_dist = F.dropout(vec_dist, p=0.2, training=self.training)
        out = F.sigmoid(self.wh(vec_dist))
        out = F.log_softmax(self.wp(out))
        return out


# module for cosine similarity
class CosSimilarity(nn.Module):
    def __init__(self, mem_dim):
        super(CosSimilarity, self).__init__()
        self.cos = nn.CosineSimilarity(dim=mem_dim)

    def forward(self, lvec, rvec):
        out = self.cos(lvec, rvec)
        out = torch.autograd.Variable(torch.FloatTensor([[1 - out.data[0], out.data[0]]]), requires_grad=True)
        if torch.cuda.is_available():
            out = out.cuda()
        return F.log_softmax(out)


class RNNLstm(nn.Module):

    def __init__(self, dimension=1024):
        super(RNNLstm, self).__init__()

        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=768,
                            hidden_size=dimension,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=0.5)

        self.fc = nn.Linear(2*dimension, 150)

    def forward(self, text_emb, text_len):


        print(text_emb.shape)
        print(text_len.type())
        packed_input = pack_padded_sequence(text_emb, text_len, batch_first=True)
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        self.lstm.flatten_parameters()
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        print(output.shape)

        out_forward = output[range(len(output)), text_len - 1, :self.dimension]
        print(out_forward.shape)
        out_reverse = output[:, 0, self.dimension:]
        print(out_reverse.shape)
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        print(out_reduced.shape)
        text_fea = self.drop(out_reduced)
        print(text_fea.shape)

        text_fea = self.fc(text_fea)
        print(text_fea.shape)
        text_fea = torch.squeeze(text_fea, 1)
        print(text_fea.shape)
        text_out = torch.sigmoid(text_fea)
        print(text_out.shape)

        return text_out


def getBertmodel():
    model = BertModel.from_pretrained("bert-base-cased")
    model.eval()
    model.cuda()
    return model

def readLines(filename):
    list = []
    count = 0;
    with open(filename, encoding="utf8") as f:

        while True:
            count += 1

            # Get next line from file
            line = f.readline()
            if(line !=""):
                list.append(line)
            # if line is empty
            # end of file is reached
            if not line:
                break
            # print("Line{}: {}".format(count, line.strip()))

        f.close()
    return list

def getBerttoken():
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    '''
    model = getBertmodel()
    bert_vocab = readLines("D:\\downloads\\QA\\listvocab.txt")

    for i in bert_vocab:
        print(i.rstrip())
        tokenizer.add_tokens([i.rstrip()])

    print(len(tokenizer))  # 28996

    model.resize_token_embeddings(len(tokenizer))
    # The new vector is added at the end of the embedding matrix
    print(tokenizer.get_vocab())
    '''
    return tokenizer


def getSize(linputs, tokenizer):
    marked_text = "[CLS] " + linputs + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)

    print(segments_ids)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    return tokens_tensor,segments_tensors


def getEmbedding(model,tokens_tensor, segments_tensors):

    device = model.device
    cp = device.type
    if(device.type == 'cpu'):
        tokens_tensor = torch.as_tensor(tokens_tensor, dtype=torch.int64, device='cpu')
        segments_tensors = torch.as_tensor(segments_tensors, dtype=torch.int64, device='cpu')

    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)


    token_vecs = encoded_layers[0]

    # Calculate the average of all 22 token vectors.
    sentence_embedding = torch.mean(token_vecs, dim=0)

    list = sentence_embedding.squeeze().tolist()

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    a = torch.Tensor(list)


    print(a.shape)
    sentence_embedding = a[np.newaxis, :]

    print(sentence_embedding.shape)

    return token_vecs

class SimilarityTreeLSTM(nn.Module):
    def __init__(self, vocab_size, in_dim, mem_dim, similarity, sparsity):
        super(SimilarityTreeLSTM, self).__init__()
        self.emb = nn.Embedding(vocab_size, in_dim, padding_idx=Constants.PAD, sparse=sparsity)
        self.childsumtreelstm = ChildSumTreeLSTM(in_dim, mem_dim)
        self.rnnlstm = RNNLstm()
        self.similarity = similarity

    def forward(self, linputs, rinputs, bertclass):
        #size = len(linputs)
        #size2 = len(rinputs)

        linputs_tokens_tensor, linputs_segments_tensors = getSize(linputs, bertclass.bert_token)
        rinputs_tokens_tensor, rinputs_segments_tensors = getSize(rinputs, bertclass.bert_token)

        linputs_tokens_tensor = linputs_tokens_tensor.cuda()
        rinputs_tokens_tensor = rinputs_tokens_tensor.cuda()
        linputs_segments_tensors = linputs_segments_tensors.cuda()
        rinputs_segments_tensors = rinputs_segments_tensors.cuda()



        #linputs = getEmbedding(self.model_BERT,linputs_tokens_tensor, linputs_segments_tensors)
        #rinputs = getEmbedding(self.model_BERT, rinputs_tokens_tensor, rinputs_segments_tensors)

        linputs,rinputs = bertclass.getSentenceEmbedding(bertclass,linputs_tokens_tensor,linputs_segments_tensors,
                                rinputs_tokens_tensor, rinputs_segments_tensors)

        #tokens_tensor = torch.as_tensor(tokens_tensor, dtype=torch.int64, device='cpu')
        linputs = linputs.cuda()
        rinputs = rinputs.cuda()
        size = linputs_tokens_tensor.shape[1]
        size2 = rinputs_tokens_tensor.shape[1]
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        arrsize = [size]
        arrsize2 = [size2]

        V = torch.tensor(arrsize)
        V2 = torch.tensor(arrsize2)



        print(linputs.shape)
        print(rinputs.shape)


        list = linputs.squeeze().tolist()
        list2 = rinputs.squeeze().tolist()
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        a = torch.Tensor(list)
        a2 = torch.Tensor(list2)

        print(a.shape)
        new_a = a[np.newaxis, :]
        new_a2 = a2[np.newaxis, :]
        print(new_a.shape)


        print(linputs.type())
        print(new_a.type())
        print(V.type())



        #lstate, lhidden = self.childsumtreelstm(ltree, linputs)
        #rstate, rhidden = self.childsumtreelstm(rtree, rinputs)

        V = torch.as_tensor(V, dtype=torch.int64, device='cpu')
        V2 = torch.as_tensor(V2, dtype=torch.int64, device='cpu')
        torch.set_default_tensor_type(torch.FloatTensor)


        outputLSTM = self.rnnlstm(new_a.contiguous(), V.contiguous())
        torch.set_default_tensor_type(torch.FloatTensor)
        outputLSTM2 = self.rnnlstm(new_a2, V2)
        #print(lstate.shape)
       # print(lhidden.shape)
        #print(rstate.shape)
        #print(rstate.shape)
        #print(outputLSTM.shape)

        #output = self.similarity(lstate, rstate)
       # output2 = self.similarity(outputLSTM, rstate)
        output3 = self.similarity(outputLSTM, outputLSTM2)
        return output3
