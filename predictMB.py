import csv
import pickle
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import seaborn as sns


class Predict:
    def __init__(self):
        super(Predict, self).__init__()
        self._vectorizer = self.getvectorizer()
        self._tfidf = self.getfidf()
        self.model = self.clssify()

    def getresult(self, string):

        print(self.model.predict(self._vectorizer.transform([string])))
        return self.model.predict(self._vectorizer.transform([string]))

    def clssify(self):
        filenametrain = './lcqald_question_type_train.csv'

        filenametest = './lcqald_question_type_test.csv'
        X_train = self.getdoc(filenametrain)
        #X_test = self. getdoc(filenametest)

        with open("D:/downloads/QA/lcqald_question_type_test.csv", "rt") as f:
            reader = csv.reader(f, delimiter=':')
            rows = list(reader)

        y_test = [row[1] for row in rows]

        with open("D:/downloads/QA/lcqald_question_type_train.csv", "rt") as f:
            reader = csv.reader(f, delimiter=':')
            rows = list(reader)

        y_train = [row[1] for row in rows]

        # X_train, X_test, y_train, y_test = train_test_split(df['Consumer_complaint_narrative'], df['Product'], random_state = 0)

        X_train_counts = self._vectorizer.fit_transform(X_train)

        X_train_tfidf = self._tfidf.fit_transform(X_train_counts)

        clf = MultinomialNB().fit(X_train_tfidf, y_train)
        return clf

    def getdoc(self,filename):
        with open(filename, "rt") as f:
            reader = csv.reader(f, delimiter=':')
            rows = list(reader)

        X = [row[0] for row in rows]
        y = [row[1] for row in rows]

        documents = []

        from nltk.stem import WordNetLemmatizer

        stemmer = WordNetLemmatizer()

        for sen in range(0, len(X)):

            sentence = re.sub(r'\W', ' ', str(X[sen]))
            sentence = re.sub(r'\s+[a-zA-Z]\s+', ' ', sentence)
            sentence = re.sub(r'\^[a-zA-Z]\s+', ' ', sentence)
            sentence = re.sub(r'\s+', ' ', sentence, flags=re.I)
            sentence = re.sub(r'^b\s+', '', sentence)
            sentence = sentence.lower()
            sentence = sentence.split()
            sentence = [stemmer.lemmatize(word) for word in sentence]
            sentence = ' '.join(sentence)

            documents.append(sentence)

        return documents

    def getvectorizer(self):
        count_vect = CountVectorizer()
        return count_vect

    def getfidf(self):
        tfidf_transformer = TfidfTransformer()
        return tfidf_transformer




