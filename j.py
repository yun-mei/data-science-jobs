import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("Jobs.csv")
df;

list_of_titles = df['title'].unique()
print('There are {} unique job titles for Data Scientist postings'.format(len(list_of_titles)));

df['title'] = df['title'].str.lower()
check1 = df['title'].str.contains('data science')
check2 = df['title'].str.contains('data scientist')

check = check1 | check2
df = df[check]

df['title'].unique();
print('There are {} unique job titles for Data Scientist postings'.format(len(df['title'].unique())));


# A function that replaces the job title if it contains a certain sub string.
def replace_title(title):
    # Lowercase the titles first to keep consistency    
    if 'lead' in title or 'principal' in title or 'founding' in title:
        return 'Lead Data Scientist'
    elif 'senior' in title or 'sr' in title:
        return 'Senior Data Scientist'
    elif 'manager' in title:
        return 'Data Science Manager'
    elif 'intern' in title:
        return 'Intern Data Scientist'
    elif 'vp' in title or 'vice president' in title:
        return 'VP of Data Science'
    elif 'director' in title:
        return 'Director of Data Science'
    elif 'staff' in title:
        return 'Staff Data Scientist'
    elif 'jr' in title or 'junior' in title:
        return 'Junior Data Scientist'
    elif 'data scientist' in title or 'data science' in title: 
        return 'Data Scientist'
    else:
        return title

# Apply function to title column
df['title'] = df['title'].apply(replace_title)

df['title'].unique();
print('There are {} unique job titles for Data Scientist postings'.format(len(df['title'].unique())));

df['word_count'] = df['description'].str.split().str.len()
summary = df['word_count'].describe(percentiles = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
summary

from fuzzywuzzy import fuzz

# list of skills to match
tools = ["Python","R","SQL","Jupyter","NumPy","Pandas","Matplotlib","Seaborn",
                      "SciPy","Scikit-Learn","TensorFlow","PyTorch","Keras","XGBoost","LightGBM",
                      "Plotly","Dask","Spark","Hadoop","AWS","Google Cloud",
                      "Azure","IBM Watson","NLTK","OpenCV","Gensim","StatsModels",
                      "Theano","Caffe","Keras-Tuner","Auto-Keras","Auto-Sklearn","Shap","ELI5","Bokeh",
                      "Folium","ggplot","plotnine","Geopandas","Datashader","Yellowbrick","H2O.ai","Flask",
                      "Dash","Streamlit","FastAPI","PySpark","TensorBoard","cuDF","NetworkX","BeautifulSoup",
                      "Scrapy","Numba","Cython", "Apache", "Git"]

def match_phrases(description, phrases):
    matched_phrase = [phrase for phrase in phrases if fuzz.partial_token_set_ratio(description, phrase) >= 90]
    # Only return matches once
    unique_matches = list(set(matched_phrase))
    return unique_matches

df['Tools'] = df['description'].apply(lambda x: match_phrases(x, tools))

import re 
from fuzzywuzzy import fuzz

def extract_years_of_experience(description):
    # Regular expression pattern to match the years of experience information
    pattern = re.compile(r'(\d+\+?\s*years?\s*of\s*experience)', re.IGNORECASE)
    
    # search for the pattern in the job description
    match = re.search(pattern, description)
    
    # if there is a match, return the matched string
    if match:
        return match.group(0)
    else:
        return "Not Specified"
    
df['Years_of_Experience'] = df['description'].apply(extract_years_of_experience)
df['Years_of_Experience'].value_counts()

import re 
from fuzzywuzzy import fuzz

def extract_years_of_experience(description):
    # Regular expression pattern to match the years of experience information
    pattern = re.compile(r'(\d+)\s*years?', re.IGNORECASE)
    
    # search for the pattern in the job description
    match = re.search(pattern, description)
    
    # if there is a match, return the matched string
    if match:
        return match.group(0)
    else:
        return "Not Specified"
    
df['Years_of_Experience'] = df['description'].apply(extract_years_of_experience)
df['Years_of_Experience'].value_counts()
# Extract just the numbers using regular expression
df['years'] = df['Years_of_Experience'].str.extract(r'(\d+)').fillna(0).astype(int)
df['years'].value_counts()
df['experience'] = df['years'].apply(lambda x: 'Not Specified' if x == 0 or x >= 18
                                    else ('1 to 3 years' if x >= 1 and x <= 3
                                    else ('4 to 6 years' if x >= 4 and x <= 6 
                                    else '6+ years')))
df['experience'].value_counts()


from fuzzywuzzy import fuzz

def extract_education_level(description):
    # Dictionary that maps education levels to their abbreviations
    education_levels = {
        'bachelor': ['bs', 'bachelor'],
        'master': ['ms', 'master'],
        'phd': ['phd'],
        'doctorate': ['doctorate']
    }
    # if none education = 0 
    education_level = None
    max_ratio = 0

    # iterate over the education levels and their abbreviations
    for level, abbreviations in education_levels.items():
        level_variants = [level] + abbreviations
        for variant in level_variants:

            # calculate the fuzzy matching ratio between the variant and the job description
            ratio = fuzz.partial_token_set_ratio(variant, description)
            if ratio > max_ratio:
                max_ratio = ratio
                education_level = level
    if max_ratio >= 80:
        return education_level
    else:
        return 'Not Specified'
    

df = df.drop(df.columns[:1], axis=1)
print(df.columns[:5])
df = df.drop('Years_of_Experience', axis=1)
df = df.drop('experience', axis=1)

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

words = df.description.str.split().explode().str.lower()
words = words[~words.isin(stop_words)]
most_common_words = words.value_counts().nlargest(30)


