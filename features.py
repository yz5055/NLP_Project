from nltk.parse import corenlp
from nltk.corpus import wordnet as wn
import os

path_to_jar = os.path.join(
    os.getcwd(), 
    'models/stanford-corenlp-4.2.0/stanford-corenlp-4.2.0.jar'
)
path_to_models_jar = os.path.join(
    os.getcwd(), 
    'models/stanford-corenlp-4.2.0/stanford-corenlp-4.2.0-models.jar'
)

server = corenlp.CoreNLPServer(path_to_jar, path_to_models_jar)
dep_parser = corenlp.CoreNLPDependencyParser()

wh_words = [
    "what",
    "which",
    "when",
    "where",
    "who",
    "will",
    "is there",
    "how long",
    "how", 
    "can",
    "would",
    "why",
]

# Find the root word of the sentence
def dep_parse(sentence):
    parse, = dep_parser.raw_parse(sentence)
    conll = parse.to_conll(4)
    tree = parse.tree()
    
    feats = { }

    for line in conll.split("\n")[:-1]:
        word, tag, head, rel = line.split("\t")
        if rel == "nsubj": feats["subject"] = word
        elif rel == "ROOT": feats["head_word"] = word 

    return feats

def build_features(series):
    question = series["Question"]
    label = series["Category"]

    features = { 
        "wh_word": None,
    }

    # wh_word
    for wh_word in wh_words:
        if wh_word in question:
            features["wh_word"] = wh_word
            break

    features.update(dep_parse(question))

    return (features, label)

