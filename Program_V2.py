

import nltk



f=open("train20_augmented.csv","r")





def head_finder(question):
	text=question
	isNoun = lambda pos: pos[:2] == 'NN'
	tokenized = nltk.word_tokenize(text)
	nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if isNoun (pos)] 
	print (nouns)
	return nouns


wh_words=["what", "which", "when", "where", "who", "how", "why", "rest"]

Q_wh_word={}

Q_head_word={}



unigram_word={}




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








