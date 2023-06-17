#!/usr/bin/env python
# coding: utf-8

# # Importing Relevent Lib

# In[42]:


from nltk.tokenize import sent_tokenize, word_tokenize
import spacy
import pandas as pd
import requests
from spacy.matcher import matcher
import re

from nltk.util import ngrams
nlp = spacy.load('en_core_web_sm')


# # Making Set Of Stop_Words
# 

# In[43]:


stopword_files = ['StopWords_Auditor.txt','StopWords_Currencies.txt','StopWords_DatesandNumbers.txt','StopWords_Generic.txt','StopWords_GenericLong.txt','StopWords_Geographic.txt','StopWords_Names.txt']
for file in stopword_files :
    with open(file, 'r') as f:
        doc = f.read().splitlines()
        
        for ele in doc:
            nlp.Defaults.stop_words.add(ele)
            nlp.vocab[ele].is_stop = True


# # Making a Dict of Positve and Negative words

# In[44]:


with open('positive-words.txt', 'r') as f:
    positive_words = [line.strip() for line in f]

positive_dict = {word: 0 for word in positive_words}

with open('negative-words.txt', 'r') as f:
    negative_words = [line.strip() for line in f]

negative_dict = {word: 0 for word in negative_words}



# # Function to calculate_scores_matrix

# In[69]:


def count_syllables(word):
    word = word.lower()
    if word.endswith('es') or word.endswith('ed'):
        word = word[:-2]
    count = len(re.findall(r'[aeiouy]+', word))
    return max(count, 1)
def calculate_scores_metrics(text, positive_dict, negative_dict):
    tokens = word_tokenize(text)
    sentences = sent_tokenize(text)
    stop_words = set(nlp.Defaults.stop_words)
    words = [word for word in word_tokenize(text) if word.isalpha() and word not in stop_words]
    total_words = len(words)
    positive_score = sum(1 for token in tokens if token in positive_dict)
    negative_score = sum(1 for token in tokens if token in negative_dict)
    polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)
    subjectivity_score = (positive_score + negative_score) / (total_words + 0.000001)
    avg_sentence_length = total_words / len(sentences)
    complex_words = sum(1 for word in words if count_syllables(word) > 2)
    percentage_complex_words = complex_words / total_words
    fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)
    personal_pronouns = len(re.findall(r'\b(?:I|we|my|ours|us)\b', text, flags=re.IGNORECASE))
    avg_word_length = sum(len(word) for word in words) / total_words
    avg_sentence_length = total_words / len(sentences)
    for word in words:
        syllables_count = count_syllables(word)

    
    return {
        'positive score':positive_score,
        'negative score':negative_score,
        'polarity score':polarity_score,
        'subjectivity score':subjectivity_score,
        'avg sentence length' :avg_sentence_length,
        'percentage_complex_words':percentage_complex_words,
        'Fog Index': fog_index,
        'Average Sentence Length': avg_sentence_length,
        'Complex Word Count': complex_words,
        'Word Count': total_words,
        'syllables count':syllables_count,
        'Personal Pronouns': personal_pronouns,
        'Average Word Length': avg_word_length
           }


# # CREATING OUTPUTDF

# In[105]:


columns = ['URL', 'positive score','negative score','polarity score','Subjectivity Score', 'Average Sentence Length','percent of complex words','Fog Index','avg sentence length','Complex Word Count', 'Word Count','SYLLABLE PER WORD', 'Personal Pronouns', 'Average Word Length']
output_df= pd.DataFrame(columns=columns)


# # Reading the URLS from input excel and running the code
# 

# In[72]:


input_urls = pd.read_excel('Input.xlsx')

for url in input_urls['URL']:
    response = requests.get(url)
    doc = nlp(response.text)
    cleaned_text = ' '.join([token.text for token in doc if not token.is_stop])
    metrics = calculate_scores_metrics(cleaned_text, positive_dict, negative_dict)
    new_row = {'url': url, **metrics}
    output_df = output_df.append(new_row, ignore_index=True)


# In[73]:


output_df


# # converting the df into given datastructure

# In[ ]:


outdf = output_df.drop(['URL','Subjectivity Score','percent of complex words','SYLLABLE PER WORD'],axis=1)
outdf_new = outdf[['url','positive score','negative score','polarity score','subjectivity score','Average Sentence Length','percentage_complex_words','Fog Index','avg sentence length','Complex Word Count','Word Count','syllables count','Personal Pronouns','Average Word Length']]
outdf_new= outdf_new.rename({'url':'URL','positive score':'POSITIVE SCORE','negative score':'NEGATIVE SCORE','polarity score':'POLARITY SCORE','subjectivity score':'SUBJECTIVITY SCORE','Average Sentence Length':'AVG SENTENCE LENGTH','percentage_complex_words':'PERCENTAGE OF COMPLEX WORDS','Fog Index':'FOG INDEX','avg sentence length':'AVG NUMBER OF WORDS PER SENTENCE','Complex Word Count':'COMPLEX WORD COUNT','Word Count':'WORD COUNT','syllables count':'SYLLABLE PER WORD','Personal Pronouns':'PERSONAL PRONOUNS','Average Word Length':'AVG WORD LENGTH'},axis=1)


# # converting pandas df into excel

# In[103]:


outdf_new.to_excel('output.xlsx',index = False)

