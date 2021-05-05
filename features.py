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
    "how long",
    "are",
    "will",
    "is there",
    "how", 
    "can",
    "would",
    "why",
]

# Find the root word of the sentence
def dep_parse(sentence):

    # replace "special" vocabulary i.e covid
    to_parse = make_substitutions(sentence)

    parse, = dep_parser.raw_parse(to_parse)
    conll = parse.to_conll(4)
    tree = parse.tree()
    
    feats = { }

    root_pos = -1
    covid_pos = -1

    for idx, line in enumerate(conll.split("\n")[:-1], start=1):
        word, tag, head, rel = line.split("\t")

        if rel == "ROOT": 
            root_pos = rel
            feats["root"] = word 

            if tag == "NN":
                feats["head_noun"] = word
        
        if word == "virus":
            covid_pos = idx + 1
    
    for idx, line in enumerate(conll.split("\n")[:-1], start=1):
        word, tag, head, rel = line.split("\t")

        # A dependent
        if head == str(covid_pos):
            feats["dependent"] = (word, tag)

    # If the root wasn't a noun, find the head noun
    # In addition, find key adjectives
    if "head_noun" not in feats:

        for idx, line in enumerate(conll.split("\n")[:-1], start=1):
            word, tag, head, rel = line.split("\t")

            if rel == "nsubj": 
                feats["head_noun"] = word
                break
            elif head == root_pos and tag == "NN": 
                feats["head_noun"] = word
                break

    hypernyms(feats)

    return feats

def hypernyms(feats):
    if "head_noun" in feats:
        word = feats["head_noun"]
        synsets = wn.synsets(word, wn.NOUN)

        if synsets:
            hs = synsets[0].hypernyms()
            if hs: feats["hypernym"] = hs[0].name()
            else: feats["hypernym"] = word

    if "dependent" in feats:
        word, tag = feats["dependent"]
        feats["dependent"] = word
        if tag[0] in penn_to_wn:
            synsets = wn.synsets(word, penn_to_wn[tag[0]])
        else:
            synsets = wn.synsets(word)

        if synsets:
            hs = synsets[0].hypernyms()
            if hs: feats["d_hypernym"] = hs[0].name()
            else: feats["d_hypernym"] = word



def build_features(series):
    question = series["Question"]
    label = series["Category"]

    features = { }

    # wh_word
    for wh_word in wh_words:
        if wh_word in question:
            features["wh_word"] = wh_word
            break

    features.update(dep_parse(question))

    return (features, label)


vocab_map = {
    "president trump": "the president",
    "trump": "the president",
    "bill gates": "he",
    "covid virus": "virus",
    "the covid": "the virus",
    "covid": "the virus",
    "the covid vaccine": "the vaccine",
}

penn_to_wn = {
    "N": wn.NOUN,
    "J": wn.ADJ,
    "V": wn.VERB,
    "R": wn.ADV,
}

def make_substitutions(sentence):
    for key, val in vocab_map.items():
        sentence = sentence.replace(key, val)
    return sentence
