import torch
from transformers import BertTokenizer, BertModel

from learning.treelstm.model import getSize, getEmbedding


def getBertmodel():
    model = BertModel.from_pretrained("bert-base-cased")
    model.eval()
    return model

def getBerttoken(model):
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    #model = getBertmodel()
    bert_vocab = readLines("D:\\downloads\\QA\\listvocab.txt")

    for i in bert_vocab:
        print(i.rstrip())
        tokenizer.add_tokens([i.rstrip()])

    print(len(tokenizer))  # 28996

    model.resize_token_embeddings(len(tokenizer))
    # The new vector is added at the end of the embedding matrix
    print(tokenizer.get_vocab())

    return tokenizer

class BertClass():
    def __init__(self):
        super(BertClass, self).__init__()

        self.model_BERT = getBertmodel()
        self.bert_token = getBerttoken(self.model_BERT)

    def getSentenceEmbedding(bertclass,self,linputs_tokens_tensor,linputs_segments_tensors,
                            rinputs_tokens_tensor, rinputs_segments_tensors):

        #linputs_tokens_tensor, linputs_segments_tensors = getSize(linputs, self.bert_token)
       # rinputs_tokens_tensor, rinputs_segments_tensors = getSize(rinputs, self.bert_token)

        #linputs_tokens_tensor, rinputs_tokens_tensor = linputs_tokens_tensor.cuda(), rinputs_tokens_tensor.cuda()

        linputs = getEmbedding(self.model_BERT, linputs_tokens_tensor, linputs_segments_tensors)
        rinputs = getEmbedding(self.model_BERT, rinputs_tokens_tensor, rinputs_segments_tensors)

        return linputs,rinputs




def readLines(filename):
    list = []
    with open(filename, encoding="utf8") as f:

        while True:
            # Get next line from file
            line = f.readline()
            if(line !=""):
                list.append(line)
            if not line:
                break

        f.close()
    return list


