import pandas as pd
import numpy as np
import os
import nltk
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models import LdaModel
from sklearn.feature_extraction.text import TfidfVectorizer
import pyLDAvis.gensim as gensimvis
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pyLDAvis
#pyLDAvis.enable_notebook()
#import warnings
#warnings.filterwarnings('ignore')

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    if isinstance(text, str):
        tokens = word_tokenize(text.lower())
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
        return tokens
    return []



df = pd.read_csv('https://github.com/saeed-saffari/LLM_education_topic_modeling/raw/refs/heads/main/Tastic%20AI%20-%20Opensource%20-%20AI%20tools.csv')

categories = [
    "AI tools for education",
    "AI writing tools",
    "AI tools for research",
    "AI summarizer tools",
    "AI coding tools",
    "AI presentation tools"
]

# Filter the DataFrame for rows where 'cat' matches any of the specified categories
filtered_df = df[df['cat'].isin(categories)]
filtered_df.head()

descriptions = filtered_df['desc'].tolist()
processed_descriptions = [preprocess(desc) for desc in descriptions]

# Create dictionary and corpus
dictionary = corpora.Dictionary(processed_descriptions)
corpus = [dictionary.doc2bow(text) for text in processed_descriptions]

# Load the model from the file
lda_model_group_8 = LdaModel.load("/Users/saeed/Library/CloudStorage/GoogleDrive-m.saeed.saffari@gmail.com/My Drive/Research/LLM_education_topic_modeling/lda_model_group_8.model")

# Use the loaded model
topics = lda_model_group_8.print_topics(num_words=10)
for idx, topic in topics:
    print(f"Topic {idx}: {topic}")


topic_name = ['Topic 0: AI Writing and Essay Tools',
              'Topic 1: Interactive Learning and Educational Videos',
              'Topic 2: Language Learning and Personalized Skill Development',
              'Topic 3: Data Analysis and Business Insights Tools',
              'Topic 4: AI Coding and Developer Tools',
              'Topic 5: AI Summarization and Research Assistance',
              'Topic 6: AI Tools for Writing and Text Processing',
              'Topic 7: AI Content Creation and Generation Tools'
              ]


# with 3 topic
doc_topics = lda_model_group_8.get_document_topics(corpus)
topic_1_mapping = []
topic_2_mapping = []
topic_3_mapping = []

for doc in doc_topics:
    sorted_topics = sorted(doc, key=lambda x: x[1], reverse=True)

    if len(sorted_topics) >= 3:
        top_3_topics = [topic[0] for topic in sorted_topics[:3]]
    else:
        top_3_topics = [topic[0] for topic in sorted_topics] + [None] * (3 - len(sorted_topics))

    topic_1_mapping.append(top_3_topics[0])
    topic_2_mapping.append(top_3_topics[1])
    topic_3_mapping.append(top_3_topics[2])

edu_with_topic_3 = filtered_df

edu_with_topic_3['Topic 1 Number'] = topic_1_mapping
edu_with_topic_3['Topic 2 Number'] = topic_2_mapping
edu_with_topic_3['Topic 3 Number'] = topic_3_mapping

edu_with_topic_3['Topic 1 Name'] = edu_with_topic_3['Topic 1 Number'].apply(lambda x: topic_name[int(x)] if pd.notna(x) else 'N/A')
edu_with_topic_3['Topic 2 Name'] = edu_with_topic_3['Topic 2 Number'].apply(lambda x: topic_name[int(x)] if pd.notna(x) else 'N/A')
edu_with_topic_3['Topic 3 Name'] = edu_with_topic_3['Topic 3 Number'].apply(lambda x: topic_name[int(x)] if pd.notna(x) else 'N/A')


