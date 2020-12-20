# Building a simple chatbot that reponse to COVID-19 general questions.
import nltk
# import the helper libraries
from newspaper import Article
import random
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# download the punt package for tokenization ( run the below two commands on your python terminal)
# >>> import nltk
# >>> nltk.download('punkt', quiet=True)
# if you are unable to run the above command, download the punkt package from here : https://www.nltk.org/nltk_data/,
# unpackage it and paste it at this location : /usr/local/share/ on your machine.

# Get the required information from a reliable medical source for the bot.
article = Article(
    'https://www.mayoclinic.org/diseases-conditions/coronavirus/symptoms-causes/syc-20479963')
article.download()
article.parse()
article.nlp()
corpus = article.text

# print the article text
# print(corpus)

text = corpus
# This will give us a list of sentenses required to train our bot
sentense_list = nltk.sent_tokenize(text)

# print the list of sentenses
# print(sentense_list)

# Function to return a random greeting response to user's greeting


def greeting_response(text):
    text = text.lower()
    # Bot's greeting response
    bot_greetings = ['howdy', 'hey', 'hi', 'hello', 'hola']
    # Users' greeting
    user_greetings = ['hi', 'hey', 'hello', 'holda', 'greetings', 'whatsup']

    for word in text.split():
        if word in user_greetings:
            return random.choice(bot_greetings)


def index_sort(list_var):
    length = len(list_var)
    list_index = list(range(0, length))
    x = list_var
    for i in range(length):
        for j in range(length):
            if x[list_index[i]] > x[list_index[j]]:
                # swap
                temp = list_index[i]
                list_index[i] = list_index[j]
                list_index[j] = temp
    return list_index

    # Create bot responses


def bot_response(user_input):
    user_input = user_input.lower()
    sentense_list.append(user_input)
    bot_response = ''
    cm = CountVectorizer().fit_transform(sentense_list)
    similarity_scores = cosine_similarity(cm[-1], cm)
    similarity_scores_list = similarity_scores.flatten()
    index = index_sort(similarity_scores_list)
    index = index[1:]
    response_flag = 0
    j = 0

    for i in range(len(index)):
        if similarity_scores_list[index[1]] > 0.0:
            bot_response = bot_response + ' '+sentense_list[index[i]]
            response_flag = 1
            j = j + 1
        if j > 2:
            break
    if response_flag == 0:
        bot_response = bot_response+' '+" I apologies I don't understand."

    sentense_list.remove(user_input)
    return bot_response


# Start the chat
print("I am your Doc Bot today ! I will answer your queries about COVID-19 disease. If you want to exit, please type bye.")
exit_list = ['exit', 'see you later', 'bye', 'quit', 'break']
while(True):
    user_input = input()
    if user_input.lower() in exit_list:
        print('Doc Bot will chat to you later. Stay safe.')
        break
    else:
        if greeting_response(user_input) != None:
            print('Doc Bot : '+greeting_response(user_input))
        else:
            print('Doc Bot : '+bot_response(user_input))
