# BERT-QA
BERT-based Question Answering Over Linked Data

Over the decades, the amount of information available on the web of data using semantic web 
standards has increased. Knowledge base such as DBpedia are becoming increasingly popular 
for various applications. The knowledge base is usually huge and not easily accessible to users. 
Therefore, it is particularly important to develop efficient methods for querying,
which require both specialized query languages (such as SPARQL) and an understanding of
the ontologies used by these knowledge bases. To solve this problem, Question Answering 
(QA) systems have been proposed. The main functionality of QA systems is to convert a 
natural language question into a corresponding SPARQL query and use the query to retrieve
the desired results from the underlying knowledge graphs. 
   
QA systems aim to build natural languageprocessing (NLP) pipelines that apply syntactic
and semantic analysis to the natural language question to generate the SPARQL query. 
The NLP pipeline components include parts-of-speech tagging, named-entity recognition, 
dependency parsing, etc. 
 
## To get SQL queries
1. Place all the data in the folder `data_and_model`
2. To train and test model run `train.py`

###Data
The json file looks like

Input questions are modified as
```
"table_id": "169",
  "question": "Show a list of soccer clubs that play in the Bundesliga.",
  "sql":{
    "sel":0,
    "conds":[[1,0,"Soccer"],[3,0,"Bundesliga"]],
    "agg":0
    },
    "wvi_corenlp":[[4,4],[10,10]],
    "bertindex_knowledge": [0, 0, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0],
    "header_knowledge": [1, 2, 0, 2, 0, 0]
```

Table data 

```
  "id": "169",
  "header": ["Club","Sport","Country","League","Manager","Coach"],
  "types": ["text", "text", "text", "text", "text", "text"], 
   "rows": [["BVB", "Soccer", "Germany", "Bundesliga", "Edin Terzic`", "Wesrfalenstadion"]],
   "rows": [["FC Bayern Munich", "Soccer", "Germany", "Bundesliga","Hans-Dieter Flick", "Allianz Arena"]],
   "name": "table_169"
```

## To get SPARQL queries from SQL

**Pre-Process the dataset**
Run ```python learning/treelstm/preprocess_lcquad.py``` to pre-process the LC-QuAD dataset for training the ranking model
Run ```python learning/treelstm/preprocess_qald.py``` to pre-process the QALD dataset for training the ranking model

**Train the BERT based ranking model**
Run ```python learning/treelstm/main.py``` to train model.

**Entity ranking process**
Run ```python getEntityFromSQL.py``` to get mappings

**Generating output process**
Run ```python testfile.py```


