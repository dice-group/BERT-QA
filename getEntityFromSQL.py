
import common.utility.utility as utility

import numpy as np

from copynews.qald import Qald

import json

import logging
import requests
import json
import pandas as pd
import numpy as np
import json
from SPARQLWrapper import SPARQLWrapper, JSON
import requests
import itertools
import spotlight
import tagme
import inflect
import re
import sys
import requests
from nltk.stem.porter import *
stemmer = PorterStemmer()
p = inflect.engine()
tagme.GCUBE_TOKEN = ""

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def intilaize(query):

    result = {}
    result['question'] = query
    result['entities'] = []
    result['relations'] = []

    return result


def merge_entity(old_e, new_e):
    for i in new_e:
        exist = False
        for j in old_e:
            for k in j['uris']:
                if i['uris'][0]['uri'] == k['uri']:
                    #k['confidence'] = max(k['confidence'], i['uris'][0]['confidence'])
                    exist = True
        if not exist:
            old_e.append(i)
    return old_e


def get_nliwod_entities(query, hashmap):
    ignore_list = []
    entities = []
    singular_query = [stemmer.stem(word) if p.singular_noun(word) == False else stemmer.stem(p.singular_noun(word)) for
                      word in query.lower().split(' ')]

    string = ' '.join(singular_query)
    words = query.split(' ')
    indexlist = {}
    surface = []
    current = 0
    locate = 0
    for i in range(len(singular_query)):
        indexlist[current] = {}
        indexlist[current]['len'] = len(words[i])-1
        indexlist[current]['surface'] = [locate, len(words[i])-1]
        current += len(singular_query[i])+1
        locate += len(words[i])+1
    for key in hashmap.keys():
        if key in string and len(key) > 2 and key not in ignore_list:
            e_list = list(set(hashmap[key]))
            k_index = string.index(key)
            if k_index in indexlist.keys():
                surface = indexlist[k_index]['surface']
            else:
                for i in indexlist:
                    if k_index>i and k_index<(i+indexlist[i]['len']):
                        surface = indexlist[i]['surface']
                        break
            for e in e_list:
                r_e = {}
                #r_e['surface'] = surface
                r_en = {}
                r_en['uri'] = e
                #r_en['confidence'] = 0.3
                r_e['uris'] = [r_en]
                entities.append(r_e)
    return entities


def preprocess_relations(file, prop=False):
    relations = {}
    with open(file, encoding='utf-8') as f:
        content = f.readlines()
        for line in content:
            split_line = line.split()

            key = ' '.join(split_line[2:])[1:-3].lower()
            key = ' '.join([stemmer.stem(word) for word in key.split()])

            if key not in relations:
                relations[key] = []

            uri = split_line[0].replace('<', '').replace('>', '')

            if prop is True:
                uri_property = uri.replace('/ontology/', '/property/')
                relations[key].extend([uri, uri_property])
            else:
                relations[key].append(uri)
    return relations


def get_spotlight_entities(query):
    entities = []
    data = {
        'text': query,
        'confidence': '0.4',
        'support': '10'
    }
    headers = {"Accept": "application/json"}
    try:
        str = 'http://api.dbpedia-spotlight.org/en/annotate/?text='+data["text"]
        response = requests.post(str, headers=headers)
        response_json = response.text.replace('@', '')
        output = json.loads(response_json)
        if 'Resources' in output.keys():
            resource = output['Resources']
            for item in resource:
                entity = {}
                uri = {}
                uri['uri'] = item['URI']
                #uri['confidence'] = float(item['similarityScore'])
                entity['uris'] = [uri]
                #entity['surface'] = [int(item['offset']), len(item['surfaceForm'])]
                entities.append(entity)
    except:
        print('Spotlight: ', query)
    return entities


def get_falcon_entities(query):

    entities = []
    relations = []
    headers = {
        'Content-Type': 'application/json',
    }
    params = (
        ('mode', 'long'),
    )
    data = "{\"text\": \"" + query + "\"}"
    response = requests.post('https://labs.tib.eu/falcon/api', headers=headers, params=params, data=data.encode('utf-8'))
    try:
        output = json.loads(response.text)
        for i in output['entities']:
            ent = {}
            #ent['surface'] = ""
            ent_uri = {}
            #ent_uri['confidence'] = 0.9
            ent_uri['uri'] = i[0]
            ent['uris'] = [ent_uri]
            entities.append(ent)
        for i in output['relations']:
            rel = {}
           # rel['surface'] = ""
            rel_uri = {}
           # rel_uri['confidence'] = 0.9
            rel_uri['uri'] = i[0]
            rel['uris'] = [rel_uri]
            relations.append(rel)
    except:
            print('get_falcon_entities: ', query)
    return entities, relations


if __name__ == "__main__":
    properties = preprocess_relations('dbpedia_3Eng_property.ttl', True)
    print('properties: ', len(properties))
    logger = logging.getLogger(__name__)
    utility.setup_logging()
    with open("D:\\downloads\\QA\\query.json", encoding='utf-8') as data_file:
        data = json.load(data_file)

    linked_data = []
    dic = data["queries"]

    output = []
    for i in dic:
        print("******start*****")
        query = (i["query"])
        question = (i["nlu"])
        #print(query)
        query = "SELECT (Time Zone) FROM 105 WHERE City = salt lake city"
        x = query.split("= ")
        y = x[1].split(" ")
        str = ""
        if(len(y) > 1):
            for l in y:
                str = str +" "+ l.capitalize()
                str = str.lstrip()
                str = str.rstrip()
                str = str.strip()

        else:
            str = x[1].capitalize()
            print(str)
        #print(str)
        print(x[0])
        table = query.split("(")
        table2 = table[1].split(")")
        print(table2[0])

        classvariable = table2[1]
        classvariable2 = classvariable.split("WHERE")
        classvariable3 = classvariable2[1].split("=")
        classname = classvariable3[0].strip()
        print("*******end*******")
        #table name start
        tablename = table2[0]
        #tagme_e = get_tag_me_entities(tablename)
        earl = intilaize(tablename)


        #if len(tagme_e) > 0:
        #    earl['entities'] = merge_entity(earl['entities'], tagme_e)

        nliwod = get_nliwod_entities(tablename, properties)

        if len(nliwod) > 0:
            earl['relations'] = merge_entity(earl['relations'], nliwod)

        e_falcon, r_falcon = get_falcon_entities(tablename)
        if len(e_falcon) > 0:
            earl['entities'] = merge_entity(earl['entities'], e_falcon)
        if len(r_falcon) > 0:
            earl['relations'] = merge_entity(earl['relations'], r_falcon)



        if (classname != 'Name'):
            nliwod = get_nliwod_entities(classname, properties)

            if len(nliwod) > 0:
                earl['relations'] = merge_entity(earl['relations'], nliwod)

            e_falcon, r_falcon = get_falcon_entities(classname)
            if len(e_falcon) > 0:
                earl['entities'] = merge_entity(earl['entities'], e_falcon)
            if len(r_falcon) > 0:
                earl['relations'] = merge_entity(earl['relations'], r_falcon)

        #class name end

        #main res start
        query = str

        spot_e = get_spotlight_entities(query)
        if len(spot_e) > 0:
            earl['entities'] = merge_entity(earl['entities'], spot_e)

        e_falcon, r_falcon = get_falcon_entities(query)
        if len(e_falcon) > 0:
            earl['entities'] = merge_entity(earl['entities'], e_falcon)
        if len(r_falcon) > 0:
            earl['relations'] = merge_entity(earl['relations'], r_falcon)
        # main res end

        #class name 2 start
        query = classname

        if(query != 'Name'):
            spot_e = get_spotlight_entities(query)
            if len(spot_e) > 0:
                earl['entities'] = merge_entity(earl['entities'], spot_e)

            e_falcon, r_falcon = get_falcon_entities(query)
            if len(e_falcon) > 0:
                earl['entities'] = merge_entity(earl['entities'], e_falcon)
            if len(r_falcon) > 0:
                earl['relations'] = merge_entity(earl['relations'], r_falcon)


        earl['entities'] = list(earl['entities'])
        earl['relations'] = list(earl['relations'])

        earl['question'] = "What is the time zone of Salt Lake City?"
        linked_data.append(earl)



    with open('data/QALD/entityww_qaldtestfromSQLtestclassquer2.json', "w") as data_file:
        json.dump(linked_data, data_file, sort_keys=True, indent=4, separators=(',', ': '))











