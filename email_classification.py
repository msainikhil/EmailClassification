from __future__ import print_function, division
import nltk
import os
import random
from collections import Counter
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import NaiveBayesClassifier, classify

# Once text is pre-processed, you can extract the features characterising spam and ham emails
#The first thing to notice is that some words like the, is or of appear in all emails, dont have much content to them 
#and are therefore not going to help you distinguish spam from non spam. Such words are called stopwords and they can be 
#disregarded during classification. NLTK has a corpus of stopwords for several languages including English.
stoplist = stopwords.words('english')

#You need to read in the files from the spam and ham subfolders and keep them in two separate lists
#To be able to iteratively read the files in a folder this fuction helps.
def init_lists(folder):
    a_list = []
    file_list = os.listdir(folder)
    for a_file in file_list:
        f = open(folder + a_file, 'r')
        a_list.append(f.read())
    f.close()
    return a_list
 
#To be able to use the words in these texts as features for your classifier, you need to preprocess the data and normalise it
#below fuction helps in that matter it main fuctions are:
#Splitting the text by white spaces and punctuation marks
#linking the different forms of the same word (for example, price and prices, is and are) to each other(lemmatizer).
#converting all words to lowercase so that the classifier does not treat People, people and PEOPLE as three separate features.
def preprocess(sentence):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(unicode(sentence, errors='ignore'))]

#this functions gets features it main algorithm is:
#read in the text of the email;
#preprocess it using the function preprocess(sentence) defined above;
#for each word that is not in the stopword list,
#either calculate how frequently it occurs in the text, or simply register the fact that the word occurs in the email. 
#The former approach is called the bag-of-words (bow, for short), and it allows the classifier to notice that certain keywords 
#may occur in both types of emails but with different frequencies, 
#for example the word Offer is much more frequent in spam than ham emails. Python Counter subclass allows to apply the bow model.
def get_features(text, setting):
    if setting=='bow':
        return {word: count for word, count in Counter(preprocess(text)).items() if not word in stoplist}
    else:
        return {word: True for word in preprocess(text) if not word in stoplist}


#this following code to the train function to train a model based on the training dataset
def train(features, samples_proportion):
    train_size = int(len(features) * samples_proportion)
    # initialise the training and test sets
    train_set, test_set = features[:train_size], features[train_size:]
    print ('Training set size = ' + str(len(train_set)) + ' emails')
    print ('Test set size = ' + str(len(test_set)) + ' emails')
    # train the classifier
    classifier = NaiveBayesClassifier.train(train_set)
    return train_set, test_set, classifier

#Evaluating your classifier performance
def evaluate(train_set, test_set, classifier):
    # check how the classifier performs on the training and test sets
    print ('Accuracy on the training set = ' + str(classify.accuracy(classifier, train_set)))
    print ('Accuracy of the test set = ' + str(classify.accuracy(classifier, test_set)))
    # check which words are most informative for the classifier
    classifier.show_most_informative_features(20)
 
if __name__ == "__main__":
    # initialise the data
    spam = init_lists('enron1/spam/')
    ham = init_lists('enron1/ham/')
    all_emails = [(email, 'spam') for email in spam]
    all_emails += [(email, 'ham') for email in ham]
    random.shuffle(all_emails)
    print ('Corpus size = ' + str(len(all_emails)) + ' emails')
 
    # extract the features
    all_features = [(get_features(email, ''), label) for (email, label) in all_emails]
    print ('Collected ' + str(len(all_features)) + ' feature sets')
    
 
    # train the classifier
    train_set, test_set, classifier = train(all_features, 0.8)
 
    # evaluate its performance
    evaluate(train_set, test_set, classifier)
