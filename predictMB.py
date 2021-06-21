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

def getdoc(filename):
    with open(filename, "rt") as f:
        reader = csv.reader(f, delimiter=':')
        rows = list(reader)

    X = [row[0] for row in rows]
    y = [row[1] for row in rows]

    documents = []

    from nltk.stem import WordNetLemmatizer

    stemmer = WordNetLemmatizer()

    for sen in range(0, len(X)):
        # Remove all the special characters
        sentence = re.sub(r'\W', ' ', str(X[sen]))

        # remove all single characters
        sentence = re.sub(r'\s+[a-zA-Z]\s+', ' ', sentence)

        # Remove single characters from the start
        sentence = re.sub(r'\^[a-zA-Z]\s+', ' ', sentence)

        # Substituting multiple spaces with single space
        sentence = re.sub(r'\s+', ' ', sentence, flags=re.I)

        # Removing prefixed 'b'
        sentence = re.sub(r'^b\s+', '', sentence)

        # Converting to Lowercase
        sentence = sentence.lower()

        # Lemmatization
        sentence = sentence.split()

        sentence = [stemmer.lemmatize(word) for word in sentence]
        sentence = ' '.join(sentence)

        documents.append(sentence)

    return documents





if __name__ == "__main__":
    filenametrain = './lcqald_question_type_train.csv'

    filenametest = './lcqald_question_type_test.csv'
    X_train = getdoc(filenametrain)
    X_test = getdoc(filenametest)

    with open("D:/downloads/QA/lcqald_question_type_test.csv", "rt") as f:
        reader = csv.reader(f, delimiter=':')
        rows = list(reader)

    y_test = [row[1] for row in rows]

    with open("D:/downloads/QA/lcqald_question_type_train.csv", "rt") as f:
        reader = csv.reader(f, delimiter=':')
        rows = list(reader)

    y_train = [row[1] for row in rows]

    # X_train, X_test, y_train, y_test = train_test_split(df['Consumer_complaint_narrative'], df['Product'], random_state = 0)
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    clf = MultinomialNB().fit(X_train_tfidf, y_train)

    with open('mb_classifier', 'wb') as picklefile:
        pickle.dump(clf, picklefile)

    with open('mb_classifier', 'rb') as training_model:
        model = pickle.load(training_model)

    print(model.predict(count_vect.transform(['What is the time zone of Salt Lake City?'])))