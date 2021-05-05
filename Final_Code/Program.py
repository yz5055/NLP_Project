from nltk.tree import Tree

from  nltk.parse.stanford import StanfordParser

f=open("train20_augmented.csv","r")

stanford_parser_dir = '/mnt/c/NLP/final_project/QA/DataSet/COvid/stanford-parser-full-2015-04-20/stanford-parser-full-2015-04-20/'
eng_model_path = stanford_parser_dir  + "edu/stanford/nlp/models/lexparser/englishRNN.ser.gz"
my_path_to_models_jar = stanford_parser_dir  + "stanford-parser-3.5.2-models.jar"
my_path_to_jar = stanford_parser_dir  + "stanford-parser.jar"

parser=StanfordParser(model_path=eng_model_path, path_to_models_jar=my_path_to_models_jar, path_to_jar=my_path_to_jar)


#parser=StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")

wh_words=["what", "which", "when", "where", "who", "how", "why", "rest"]

Q_wh_word={}




def Q_head_word(question):
	#sentences=parser.raw_parse_sents((question))

	question_component=question.split(" ")
	#trees=nltk.parse(question_component)
	#print(trees)


for line in f:
	component=line.split(",")
	question=component[0]
	category=component[1].strip('\r\n')

	question_words=question.split(" ")
	Q_wh_word[question]="rest"
	for single_word in question_words:
		for wh_word in wh_words:
			if wh_word in single_word:
				Q_wh_word[question]=wh_word
	Q_head_word(question)
	sentences=parser.raw_parse_sents((question))








