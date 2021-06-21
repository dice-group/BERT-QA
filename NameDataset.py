# References
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/pytorch_basics/main.py
# http://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import csv
import numpy as np
import re
import nltk
from copynews.lc_quad_linked import LC_Qaud_Linked
from copynews.lc_quad import LC_Qaud
from sklearn.datasets import load_files
nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords

from copynews.qald import Qald


def getSentebnce(X):
    documents = []

    from nltk.stem import WordNetLemmatizer

    stemmer = WordNetLemmatizer()

    document = re.sub(r'\W', ' ', str(X))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # Converting to Lowercase
    document = document.lower()

    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)

    documents.append(document)

    return document


def generateData(param):
    if(param == "train"):
        q_ds = Qald(Qald.qald_9_train)
        #q_ds = LC_Qaud(path="./data/LC-QUAD/data.json")
        name = "qald_question_type_train.csv"
    elif(param == "test"):
        q_ds = Qald(Qald.qald_9_test)
        #q_ds = LC_Qaud_Linked(path="./data/LC-QUAD/linked_test.json")
        name = "qald_question_type_test.csv"

    q_ds.load()
    q_ds.parse()

    qald = []
    q_y = []
    for qapair in q_ds.qapairs:

        sentence = getSentebnce(qapair.question.text);
        if "COUNT(" in qapair.sparql.query:

            qald.append(sentence + ":" +"count")
            q_y.append(2)
        elif "ASK" in qapair.sparql.query:
            qald.append(sentence+":"+"ask")
            q_y.append(1)

            x = ascii(qapair.sparql.query.replace('\n', ' ').replace('\t', ' '))
            print(x)
        else:
            qald.append(sentence+":"+"normal")
            q_y.append(0)

    q_y = np.array(q_y)
    qald = np.array(qald)
    print('LIST: ', sum(q_y == 0))
    print('ASK: ', sum(q_y == 1))
    print('COUNT: ', sum(q_y == 2))




    np.savetxt(name, qald, fmt="%s", delimiter=',',encoding='utf-8')


class NameDataset(Dataset):
    """ Diabetes dataset."""

    # Initialize your data, download, etc.
    def __init__(self, is_train_set=True):
        if(is_train_set):
            generateData("train")
        else:
            generateData("test")

        filename = 'D:/downloads/QA/qald_question_type_train.csv' if is_train_set else 'D:/downloads/QA/qald_question_type_test.csv'
        with open(filename, "rt") as f:
            reader = csv.reader(f, delimiter=':')
            rows = list(reader)

        self.names = [row[0] for row in rows]
        self.countries = [row[1] for row in rows]
        self.len = len(self.countries)

        self.country_list = list(sorted(set(self.countries)))

    def __getitem__(self, index):
        return self.names[index], self.countries[index]

    def __len__(self):
        return self.len

    def get_countries(self):
        return self.country_list

    def get_country(self, id):
        return self.country_list[id]

    def get_country_id(self, country):
        return self.country_list.index(country)

# Test the loader
if __name__ == "__main__":
    dataset = NameDataset(True)


    train_loader = DataLoader(dataset=dataset,
                              batch_size=10,
                              shuffle=True)

    print(len(train_loader.dataset))
    for epoch in range(2):
        for i, (question, type) in enumerate(train_loader):
            # Run your training process
            print(epoch, i, "question", question, "type", type)