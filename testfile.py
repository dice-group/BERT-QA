import csv

from common.graph.graph import Graph
from common.query.querybuilder import QueryBuilder
from copynews.lc_quad import LC_Qaud
from sklearn.model_selection import train_test_split
import os
import torch.optim as optim

from learning.treelstm.model import *
from learning.treelstm.vocab import Vocab
from learning.treelstm.trainer import Trainer
from learning.treelstm.dataset import QGDataset
import learning.treelstm.preprocess_lcquad as preprocess_lcquad
from common.container.uri import Uri
from common.container.linkeditem import LinkedItem
from copynews.lc_quad import LC_QaudParser
import common.utility.utility as utility
from learning.classifier.svmclassifier import SVMClassifier
import ujson
import learning.treelstm.Constants as Constants

import itertools
from copynews.lc_quad_linked import LC_Qaud_Linked
from copynews.qald import Qald
from common.container.sparql import SPARQL
from common.container.answerset import AnswerSet
from common.graph.graph import Graph
from common.utility.stats import Stats
from common.query.querybuilder import QueryBuilder
from linker.goldLinker import GoldLinker
from linker.earl import Earl
from learning.classifier.svmclassifier import SVMClassifier
import json
import argparse
import logging
import sys
import os
import itertools
from collections import Counter
import pickle

from qald_test import NumpyEncoder
from testmain import testmain
from sklearn.feature_extraction.text import CountVectorizer
from predictMB import Predict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

def rank(question, generated_queries ,parser):
    if len(generated_queries) == 0:
        return []
    if 2 > 1:
        # try:
        # Load the model
        checkpoint_filename = "D:/downloads/QA/learning/treelstm/learning/treelstm/checkpoints/lc_quad,epoch=15,train_loss=0.2348909229040146.pt"


        # Prepare the dataset
        json_data = [{"id": "test", "question": question,
                      "generated_queries": [{"query": " .".join(query["where"]), "correct": False} for query in
                                            generated_queries]}]
        output_dir = "./output/"
        preprocess_lcquad.splittest(json_data, parser)
        #preprocess_lcquad.save_split(output_dir, *preprocess_lcquad.split(json_data, parser))
        #preprocess_lcquad.parse(output_dir)
    test_dataset = QGDataset(output_dir, 2)
    test_loss, test_pred = testmain(test_dataset)
    # test_loss, test_pred = trainer.test(test_dataset)
    return test_pred


def generate_query(question, entities, relations, question_type, kb, parser, h1_threshold=9999999):
    ask_query = False
    sort_query = False
    count_query = False

    if question_type == 2:
        count_query = True
    elif question_type == 1:
        ask_query = True

    double_relation = False

    graph = Graph(kb)
    query_builder = QueryBuilder()
    graph.find_minimal_subgraph(entities, relations, double_relation=double_relation, ask_query=ask_query,
                                sort_query=sort_query, h1_threshold=h1_threshold)

    valid_walks = query_builder.to_where_statement(graph, parser.parse_queryresult, ask_query=ask_query,
                                                   count_query=count_query, sort_query=sort_query)

    if question_type == 0 and len(relations) == 1:
        double_relation = True
        graph = Graph(kb)
        query_builder = QueryBuilder()
        graph.find_minimal_subgraph(entities, relations, double_relation=double_relation, ask_query=ask_query,
                                    sort_query=sort_query, h1_threshold=h1_threshold)
        valid_walks_new = query_builder.to_where_statement(graph, parser.parse_queryresult,
                                                           ask_query=ask_query,
                                                           count_query=count_query, sort_query=sort_query)
        valid_walks.extend(valid_walks_new)

    try:

        scores = rank(question, valid_walks, parser)

    except Exception as e:
        print(e)
        scores = [1 for _ in valid_walks]
    for idx, item in enumerate(valid_walks):
        if idx >= len(scores):
            item["confidence"] = 0.3
        else:
            item["confidence"] = float(scores[idx] - 1)
            item["confidence"] = float(scores[idx] - 1)

    return valid_walks, question_type


def sort_query(linker, kb, parser, qapair, model, predictclass, force_gold=True):
    ask_query = False
    count_query = False
    question_type = predictclass.getresult(qapair.question.text)
    question = qapair.question.text
    if question_type == 'count':
        question_type = 2
    elif question_type == 'boolean':
        question_type = 1
    elif question_type == 'normal':
        question_type = 0

    entities, ontologies = linker.do(qapair, force_gold=force_gold)
    precision = None
    recall = None

    if qapair.answerset is None or len(qapair.answerset) == 0:
        return "-Not_Applicable", [], question_type, precision, recall
    else:
        if entities is None or ontologies is None:
            recall = 0.0
            return "-Linker_failed", [], question_type, precision, recall

        logger.info("start finding the minimal subgraph")

        entity_list = []
        for L in range(1, len(entities) + 1):
            for subset in itertools.combinations(entities, L):
                entity_list.append(subset)
        entity_list = entity_list[::-1]

        relation_list = []
        for L in range(1, len(ontologies) + 1):
            for subset in itertools.combinations(ontologies, L):
                relation_list.append(subset)
        relation_list = relation_list[::-1]

        combination_list = [(x, y) for x in entity_list for y in relation_list]

        generated_queries = []

        for comb in combination_list:
            if len(generated_queries) == 0:
                generated_queries, question_type = generate_query(question, comb[0], comb[1] ,question_type, kb, parser)
                if len(generated_queries) > 0:
                    ask_query = False
                    count_query = False

                    if int(question_type) == 2:
                        count_query = True
                    elif int(question_type) == 1:
                        ask_query = True
            else:
                break

        generated_queries.extend(generated_queries)
        if len(generated_queries) == 0:
            recall = 0.0
            return "sparql.raw_query", "answerset", question_type

        scores = []
        for s in generated_queries:
            scores.append(s['confidence'])

        scores = np.array(scores)
        inds = scores.argsort()[::-1]
        sorted_queries = [generated_queries[s] for s in inds]
        scores = [scores[s] for s in inds]

        used_answer = []
        uniqueid = []
        for i in range(len(sorted_queries)):
            if sorted_queries[i]['where'] not in used_answer:
                used_answer.append(sorted_queries[i]['where'])
                uniqueid.append(i)

        sorted_queries = [sorted_queries[i] for i in uniqueid]
        scores = [scores[i] for i in uniqueid]

        s_counter = Counter(sorted(scores, reverse=True))
        s_ind = []
        s_i = 0
        for k, v in s_counter.items():
            s_ind.append(range(s_i, s_i + v))
            s_i += v


        '''for idx in range(len(sorted_queries)):
            where = sorted_queries[idx]
            whereContents = where.split(" ")
            if (whereContents.size == 3):
                if (whereContents[0] == "?u_0" or whereContents[0] == "?u_0"):
                    if (whereContents[0] == "?u_0" or whereContents[0] == "?u_0"):
                        sorted_queries.remove[idx]
        '''

        where = sorted_queries[0]
        if "answer" in where:
            answerset = where["answer"]
            target_var = where["target_var"]
        else:
            target_var = "?u_" + str(where["suggested_id"])
            raw_answer = kb.query_where(where["where"], target_var, count_query, ask_query)
            answerset = AnswerSet(raw_answer, parser.parse_queryresult)

            # output_where[idx]["target_var"] = target_var
        sparql = SPARQL(kb.sparql_query(where["where"], target_var, count_query, ask_query), ds.parser.parse_sparql)

        if answerset == qapair.answerset:

            return (sparql.raw_query, question_type)

        else:
            if target_var == "?u_0":
                target_var = "?u_1"
            else:
                target_var = "?u_0"

            sparql = SPARQL(kb.sparql_query(where["where"], target_var, count_query, ask_query), ds.parser.parse_sparql)

            return (sparql.raw_query, question_type)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    utility.setup_logging()
    predictclass = Predict()
    ds = Qald(Qald.qald_7)
    ds.load()
    ds.parse()

    if not ds.parser.kb.server_available:
        logger.error("Server is not available. Please check the endpoint at: {}".format(ds.parser.kb.endpoint))
        sys.exit(0)

    output_file = 'qald9answer_output3'
    linker = Earl(path="data/QALD/entityww_qaldtestfromSQLtestclassquer.json")
    #linker = GoldLinker()

    base_dir = "./output"
    with open('mb_classifier', 'rb') as training_model:
        model = pickle.load(training_model)

    parser = LC_QaudParser()
    kb = parser.kb

    tmp = []
    output = []
    i = 0;
    for qapair in ds.qapairs:
        output_row = {"question": qapair.question.text,
                      "id": qapair.id,
                      "query": qapair.sparql.query,
                      "answer": "",
                      "question_type": None
                      }

        if qapair.answerset is None or len(qapair.answerset) == 0:
            output_row["answer"] = "-no_answer"
        else:
            #result, where, question_type, type_confidence, precision, recall = o.sort_query(linker, ds.parser.kb, ds.parser, qapair, question_type_classifier, True)
            sparql, type = sort_query(linker, ds.parser.kb, ds.parser, qapair, model, predictclass, True)

            #output_row["answer"] = answer
            output_row["generated_queries"] = sparql
            output_row["question_type"] = type
            print(qapair.question.text)
            print(qapair.id)
            i = i + 1
        output.append(output_row)

    with open("output/{}.json".format(output_file), "w") as data_file:
        json.dump(output, data_file, sort_keys=True, indent=4, separators=(',', ': '), cls=NumpyEncoder)