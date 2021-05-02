

import nltk
import stop_list


f=open("train20_augmented.csv","r")
f_test=open("testA.csv","r")


output_file=open("training.feature","w")
output_file_test=open("test.feature","w")
answer=open("answer.chunk","w")

stop_list=stop_list.closed_class_stop_words



def head_finder(question):
	text=question
	isNoun = lambda pos: pos[:2] == 'NN'
	tokenized = nltk.word_tokenize(text)
	nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if isNoun (pos)] 
	print (nouns)
	return nouns
def unigram_words(question):
	unigram_list=[]
	question_component=question.split(" ")
	for word in question_component:
		if word not in stop_list:
			unigram_list.append(word)

	return unigram_list



wh_words=["what", "which", "when", "where", "who", "how", "why", "rest"]

Q_wh_word={}

Q_head_word={}

Q_wh_word_test={}

Q_head_word_test={}

unigram_word={}

unigram_word_test={}



for line in f:
	component=line.split(",")
	question=component[0]
	category=component[1].strip('\r\n')

	question_words=question.split(" ")
	unigram_word[question]=question_words


	Q_wh_word[question]="rest"
	#wh_word
	for single_word in question_words:
		for wh_word in wh_words:
			if wh_word in single_word:
				Q_wh_word[question]=wh_word



     ##head word
	Q_head_word[question]=head_finder(question)

	##unigram word

	unigram_word[question]=unigram_words(question)


for line in f_test:
	component=line.split(",")
	question=component[0]
	category=component[1].strip('\r\n')

	question_words=question.split(" ")
	unigram_word_test[question]=question_words


	Q_wh_word_test[question]="rest"
	#wh_word
	for single_word in question_words:
		for wh_word in wh_words:
			if wh_word in single_word:
				Q_wh_word_test[question]=wh_word



     ##head word
	Q_head_word_test[question]=head_finder(question)

	##unigram word

	unigram_word_test[question]=unigram_words(question)




f=open("train20_augmented.csv","r")


for line in f:

	component=line.split(",")
	question=component[0]
	category=component[1].strip('\r\n')
	

	output_file.write(question+"\t")
	output_file.write("wh-word=")
	output_file.write(Q_wh_word[question]+"\t")
	output_file.write("head word=")
	if (len(Q_head_word[question])>0):
		output_file.write(Q_head_word[question][0]+"\t")
	
	output_file.write("unigram word=")
	for word in unigram_word[question]:
		output_file.write(word+" ")
	
	output_file.write("\t")
	output_file.write(category)
	output_file.write("\n")
	


f_test=open("testA.csv","r")

for line in f_test:

	component=line.split(",")
	question=component[0]
	category=component[1].strip('\r\n')
	answer.write(question+"\t")
	answer.write(category+"\n")
	

	output_file_test.write(question+"\t")
	output_file_test.write("wh-word=")
	output_file_test.write(Q_wh_word_test[question]+"\t")
	output_file_test.write("head word=")
	if (len(Q_head_word_test[question])>0):
		output_file_test.write(Q_head_word_test[question][0]+"\t")
	
	output_file_test.write("unigram word=")
	for word in unigram_word_test[question]:
		output_file_test.write(word+" ")
	
	
	output_file_test.write("\n")