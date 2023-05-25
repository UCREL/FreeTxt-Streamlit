import streamlit as st
import base64
import os
import string
import random
import pandas as pd
import numpy as np
import streamlit as st
from collections import Counter
import spacy
from spacy.tokens import Doc
from spacy.vocab import Vocab
import nltk
import io
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image as PilImage
from textblob import TextBlob
from nltk import word_tokenize, sent_tokenize, ngrams
from wordcloud import WordCloud, ImageColorGenerator
from nltk.corpus import stopwords
from labels import MESSAGES
from summarizer_labels import SUM_MESSAGES
from summa.summarizer import summarize as summa_summarizer
from langdetect import detect
nltk.download('punkt') # one time execution
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
import math
from pathlib import Path
from typing import List
import networkx as nx
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px #### pip install plotly.express
from dateutil import parser
import streamlit.components.v1 as components
from io import StringIO
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, ColumnsAutoSizeMode
from datetime import datetime
import plotly.graph_objects as go
import math
import random
from labels import MESSAGES
import tempfile
#########Download report
from io import BytesIO
from reportlab.lib.pagesizes import letter, landscape, A4
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image as ReportLabImage, Spacer, BaseDocTemplate, Frame, PageTemplate
from reportlab.lib.units import inch

def get_image_as_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')
def get_html_as_base64(path):
    with open(path, 'r') as file:
        html = file.read()
    return base64.b64encode(html.encode()).decode()

def save_uploaded_file(uploadedfile):
    with open(os.path.join("temp",uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())
    return st.success("Saved File:{} to temp".format(uploadedfile.name))

# Update with the Welsh stopwords (source: https://github.com/techiaith/ataleiriau)
en_stopwords = list(stopwords.words('english'))
cy_stopwords = open('welsh_stopwords.txt', 'r', encoding='iso-8859-1').read().split('\n') # replaced 'utf8' with 'iso-8859-1'
STOPWORDS = set(en_stopwords + cy_stopwords)
PUNCS = '''!→()-[]{};:'"\,<>./?@#$%^&*_~'''
pd.set_option('display.max_colwidth',None)

lang='en'
EXAMPLES_DIR = 'example_texts_pub'



# reading example and uploaded files
@st.cache(suppress_st_warning=True)
def read_file(fname, file_source):
    file_name = fname if file_source=='example' else fname.name
    if file_name.endswith('.txt'):
        data = open(fname, 'r', errors='ignore').read().split(r'[.\n]+') if file_source=='example' else fname.read().decode('utf8', errors='ignore').split('\n')
        data = pd.DataFrame.from_dict({i+1: data[i] for i in range(len(data))}, orient='index', columns = ['Reviews'])
        
    elif file_name.endswith(('.xls','.xlsx')):
        data = pd.read_excel(pd.ExcelFile(fname)) if file_source=='example' else pd.read_excel(fname)

    elif file_name.endswith('.tsv'):
        data = pd.read_csv(fname, sep='\t', encoding='cp1252') if file_source=='example' else pd.read_csv(fname, sep='\t', encoding='cp1252')
    else:
        return False, st.error(f"""**FileFormatError:** Unrecognised file format. Please ensure your file name has the extension `.txt`, `.xlsx`, `.xls`, `.tsv`.""", icon="🚨")
    return True, data

def get_data(file_source='example'):
    try:
        if file_source=='example':
            example_files = sorted([f for f in os.listdir(EXAMPLES_DIR) if f.startswith('Reviews')])
            fnames = st.sidebar.multiselect('Select example data file(s)', example_files, example_files[0])
            if fnames:
                return True, {fname:read_file(os.path.join(EXAMPLES_DIR, fname), file_source) for fname in fnames}
            else:
                return False, st.info('''**NoFileSelected:** Please select at least one file from the sidebar list.''', icon="ℹ️")
        
        elif file_source=='uploaded': # Todo: Consider a maximum number of files for memory management. 
            uploaded_files = st.sidebar.file_uploader("Upload your data file(s)", accept_multiple_files=True, type=['txt','tsv','xlsx', 'xls'])
            if uploaded_files:
                return True, {uploaded_file.name:read_file(uploaded_file, file_source) for uploaded_file in uploaded_files}
            else:
                return False, st.info('''**NoFileUploaded:** Please upload files with the upload button or by dragging the file into the upload area. Acceptable file formats include `.txt`, `.xlsx`, `.xls`, `.tsv`.''', icon="ℹ️")
        else:
            return False, st.error(f'''**UnexpectedFileError:** Some or all of your files may be empty or invalid. Acceptable file formats include `.txt`, `.xlsx`, `.xls`, `.tsv`.''', icon="🚨")
    except Exception as err:
        return False, st.error(f'''**UnexpectedFileError:** {err} Some or all of your files may be empty or invalid. Acceptable file formats include `.txt`, `.xlsx`, `.xls`, `.tsv`.''', icon="🚨")


def select_columns(data, key):
    layout = st.columns([7, 0.2, 2, 0.2, 2, 0.2, 3, 0.2, 3])
    selected_columns = layout[0].multiselect('Select column(s) below to analyse', data.columns, help='Select columns you are interested in with this selection box', key= f"{key}_cols_multiselect")
    start_row=0
    if selected_columns: start_row = layout[2].number_input('Choose start row:', value=0, min_value=0, max_value=5)
    
    if len(selected_columns)>=2 and layout[4].checkbox('Filter rows?'):
        filter_column = layout[6].selectbox('Select filter column', selected_columns)
        if filter_column: 
            filter_key = layout[8].selectbox('Select filter key', set(data[filter_column]))
            data = data[selected_columns][start_row:].dropna(how='all')
            return data.loc[data[filter_column] == filter_key].drop_duplicates()
    else:
        return data[selected_columns][start_row:].dropna(how='all').drop_duplicates()

def detect_language(df):
    detected_languages = []

    # Loop through all columns in the DataFrame
    for col in df.columns:
        # Loop through all rows in the column
        for text in df[col].fillna(''):
            # Use langdetect's detect_langs to detect the language of the text
            try:
                lang_probs =  detect_langs(text)
                most_probable_lang = max(lang_probs, key=lambda x: x.prob)
                detected_languages.append(most_probable_lang.lang)
            except Exception as e:
                print(f"Error detecting language: {e}")

    # Count the number of occurrences of each language
    lang_counts = pd.Series(detected_languages).value_counts()

    # Determine the most common language in the DataFrame
    if not lang_counts.empty:
        most_common_lang = lang_counts.index[0]
    else:
        most_common_lang = None
        print("No languages detected in the DataFrame.")

    return most_common_lang


###############PAGES########################################################################################

# ----------------
st.set_page_config(
     page_title='Welsh Free Text Tool',
     page_icon='🌐',
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'Get Help': "https://ucrel.lancs.ac.uk/freetxt/",
         'Report a bug': "https://github.com/UCREL/welsh-freetxt-app/issues",
                 'About': '''## The FreeTxt/TestunRhydd tool 
         FreeTxt was developed as part of an AHRC funded collaborative
    FreeTxt supporting bilingual free-text survey  
    and questionnaire data analysis
    research project involving colleagues from
    Cardiff University and Lancaster University (Grant Number AH/W004844/1). 
    The team included PI - Dawn Knight;
    CIs - Paul Rayson, Mo El-Haj;
    RAs - Ignatius Ezeani, Nouran Khallaf and Steve Morris. 
    The Project Advisory Group included representatives from 
    National Trust Wales, Cadw, National Museum Wales,
    CBAC | WJEC and National Centre for Learning Welsh.
    -------------------------------------------------------   
    Datblygwyd TestunRhydd fel rhan o brosiect ymchwil 
    cydweithredol a gyllidwyd gan yr AHRC 
    ‘TestunRhydd: yn cefnogi dadansoddi data arolygon testun 
    rhydd a holiaduron dwyieithog’ sy’n cynnwys cydweithwyr
    o Brifysgol Caerdydd a Phrifysgol Caerhirfryn (Rhif y 
    Grant AH/W004844/1).  
    Roedd y tîm yn cynnwys PY – Dawn Knight; 
    CYwyr – Paul Rayson, Mo El-Haj; CydY 
    – Igantius Ezeani, Nouran Khallaf a Steve Morris.
    Roedd Grŵp Ymgynghorol y Prosiect yn cynnwys cynrychiolwyr 
    o Ymddiriedolaeth Genedlaethol Cymru, Amgueddfa Cymru,
    CBAC a’r Ganolfan Dysgu Cymraeg Genedlaethol.  
       '''
     }
 )
###########################################Demo page#######################################################################
def demo_page():
    # Demo page content and layout
    # ...
    st.markdown("""
    <style>
        .stButton>button {
            display: inline-block;
            padding: 10px 20px;
            font-size: 20px;
            cursor: pointer;
            text-align: center;
            text-decoration: none;
            outline: none;
            color: #fff;
            background-color: #4CAF50;
            border: none;
            border-radius: 15px;
            box-shadow: 0 9px #999;
        }
        .stButton>button:hover {background-color: #3e8e41} /* Add a darker green color when the button is hovered */
        .stButton>button:active {
            background-color: #3e8e41;
            box-shadow: 0 5px #666;
            transform: translateY(4px);
        }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.image("img/FreeTxt_logo.png", width=300) 
    with col2:
        st.markdown("<h1 style='text-align: center; margin-top: 0px;'>Demo</h1>", unsafe_allow_html=True)
    with col3:
        st.image("img/FreeTxt_logo.png", width=300)
     
    coll1, coll2, coll3 = st.columns([2, 2, 2])
    with coll1:
        bt1,bt2,bt3 = st.columns([2,2,1])
        with bt1:
            if st.button('Home'):
                st.experimental_set_query_params(page=None)
            if st.button('Analysis'):
                st.experimental_set_query_params(page="analysis")
        
###########################################Analysis page#######################################################################
def analysis_page():
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
         
        st.image("img/FreeTxt_logo.png", width=300) 
        
    with col2:
        st.markdown("""
<h1 style='text-align: center; 
            margin-top: 20px; 
            color: #4a4a4a; 
            font-family: Arial, sans-serif; 
            font-weight: 300; 
            letter-spacing: 2px;'>
    Text Analysis
</h1>""", 
unsafe_allow_html=True)
    with col3:
        st.markdown("""
<style>
    .stButton>button {
        display: inline-block;
        padding: 10px 20px;
        font-size: 20px;
        cursor: pointer;
        text-align: center;
        text-decoration: none;
        outline: none;
        color: #fff;
        background-color: #4CAF50;
        border: none;
        border-radius: 15px;
        box-shadow: 0 9px #999;
    }
    .stButton>button:hover {background-color: #3e8e41} /* Add a darker green color when the button is hovered */
    .stButton>button:active {
        background-color: #3e8e41;
        box-shadow: 0 5px #666;
        transform: translateY(4px);
    }
</style>
""", unsafe_allow_html=True)
       
        st.image("img/FreeTxt_logo.png", width=300) 
    # Analysis page content and layout
    st.write("---")
    bt1,bt2,bt3,bt4,bt5,bt6 = st.columns([2,2,2,2,2,2])
    with bt1:
            if st.button('Home'):
                st.experimental_set_query_params(page=None)
    with bt2:
            if st.button('Demo'):
                st.experimental_set_query_params(page="demo")
    
    st.header("Start analysing your text")
    
    if 'uploaded_text' in st.session_state:
        st.text_area("Your text", value=st.session_state.uploaded_text)
    elif 'uploaded_file' in st.session_state:
        st.write(f"You've uploaded {st.session_state.uploaded_file.name}")
    else:
        text = st.text_area("Paste your text here")
        uploaded_file = st.file_uploader("Or upload a document", type=['txt', 'doc', 'docx', 'pdf'])

        if text:
            st.session_state.uploaded_text = text
        elif uploaded_file:
            save_uploaded_file(uploaded_file)
            st.session_state.uploaded_file = uploaded_file
###########################################Home page#######################################################################
def main():
    
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.image("img/FreeTxt_logo.png", width=300) 
    with col2:
        st.markdown("<h1 style='text-align: center; margin-top: 0px;'>Welcome to FreeTxt</h1>", unsafe_allow_html=True)
    with col3:
        st.image("img/FreeTxt_logo.png", width=300) 

    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html =True)
    st.markdown(
      """
<div style='background-color: lightgreen; padding: 10px; border-radius: 5px; color: black; font-size: 24px;'>
A free online text analysis and visualisation tool for English and Welsh. 
FreeTxt allows you to upload free-text feedback data from surveys, questionnaires etc., 
and to carry out quick and detailed analysis of the responses. FreeTxt will reveal 
common patterns of meaning in the feedback (e.g. what people are talking about/what they are saying about things),
common feelings about topics being discussed (i.e. their ‘sentiment’), 
and can produce simple summaries of the feedback provided. FreeTxt presents the results of 
analyses in visually engaging and easy to interpret ways, and has been designed to allow anyone in 
any sector in Wales and beyond to use it.
</div>
""",
    unsafe_allow_html=True,
    )  
    
    st.write("")

    col1, button_col1, button_col2,col2= st.columns([1,1,1, 1])

   
    with button_col1:
        if st.button("Start Analysis", key="analysis_button", help="Redirects to the Analysis page"):

            st.experimental_set_query_params(page="analysis")

    with button_col2:
        if st.button("Watch a Demo", key="demo_button", help="Redirects to the Demo page"):
            st.experimental_set_query_params(page="demo")
    
    st.write("---")
   
    st.markdown(
    f"""
    <style>
    .content-container {{
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        grid-template-rows: repeat(3, 1fr);
        gap: 10px;
        justify-items: center;
        align-items: center;
        padding: 10px;
        border-radius: 5px;
        background-color: white;
        color: white;
        text-align: center;
    }}
    
    .content-container > :nth-child(5) {{
        grid-column: 1 / -1;
    }}
    .a-image {{
        border-radius: 5px;
        transition: transform .2s;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.12), 0 1px 1px rgba(0, 0, 0, 0.24);
        position: relative;
    }}
    .a-image:hover {{
        transform: scale(1.1);
        box-shadow: 0 14px 28px rgba(0, 0, 0, 0.25), 0 10px 10px rgba(0, 0, 0, 0.22);
    }}
    .a-image:hover::after {{
        content: attr(title);
        position: absolute;
        top: -30px;
        left: 50%;
        transform: translateX(-50%);
        background-color: rgba(0, 0, 0, 0.8);
        padding: 5px 10px;
        border-radius: 3px;
        font-size: 14px;
        color: white;
    }}
    </style>
    <div class="content-container">
        <div>
            <h3>Word Collocation</h3>
            <iframe class="a-image" src="data:text/html;base64,{get_html_as_base64('img/analysis/network_output.html')}" width="550" height="350" title="Network Output"></iframe>
        </div>
        <div>
            <h3>Word Context</h3>
            <img class="a-image" src="data:image/png;base64,{get_image_as_base64('img/analysis/Keyword.png')}" alt="Keyword in Context" width="500" title="Keyword in Context">
        </div>
        <div>
            <h3>Positive and Negative Ratio<h3>
            <iframe class="a-image" src="data:text/html;base64,{get_html_as_base64('img/analysis/Sentiment_analysis_pie.html')}" width="500" height="400" title="Sentiment Analysis Pie"></iframe>
        </div>
        <div>
            <h3>Word Cloud</h3>
            <img class="a-image" src="data:image/png;base64,{get_image_as_base64('img/analysis/word_cloud.png')}" alt="Wordcloud" width="500" title="Wordcloud">
        </div>
        <div>
            <h3>Text Visualisation</h3>
            <iframe class="a-image" src="data:text/html;base64,{get_html_as_base64('img/analysis/scattertext_visualization.html')}" width="900" height="500" title="Scattertext Visualization"></iframe>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

    

    st.write("---")
    #st.header("The FreeTxt way-in")
    
   
    st.markdown(
      """
<div style='background-color: lightgrey; padding: 10px; border-radius: 5px; color: black; font-size: 14px;'>
FreeTxt was developed as part of an AHRC funded collaborative FreeTxt supporting bilingual free-text 
    survey and questionnaire data analysis research project involving colleagues from Cardiff University and 
    Lancaster University (Grant Number AH/W004844/1).

    The team included 
        PI - Dawn Knight; CIs - Paul Rayson, Mo El-Haj; 
        RAs - Ignatius Ezeani, Nouran Khallaf and Steve Morris.
    The Project Advisory Group included representatives from:
        National Trust Wales, Cadw, National Museum Wales, CBAC | WJEC and National Centre for Learning Welsh.

</div>
""",
    unsafe_allow_html=True,
    )     
    

    st.markdown("<br></br>", unsafe_allow_html=True) # Creates some space before logos


    st.markdown(
    f"""
    <style>
        .logo-container {{
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
        }}
        .logo {{
            width: 100px;
            height: 100px;
            margin: 10px;
            object-fit: contain;
            flex-grow: 1;
        }}
    </style>

    <div class="logo-container">
        <img class="logo" src="data:image/png;base64,{get_image_as_base64('img/cardiff.png')}" />
        <img class="logo" src="data:image/png;base64,{get_image_as_base64('img/Lancaster.png')}" />
        <img class="logo" src="data:image/png;base64,{get_image_as_base64('img/NTW.JPG')}" />
        <img class="logo" src="data:image/png;base64,{get_image_as_base64('img/Amgueddfa_Cymru_logo.svg.png')}" />
        <img class="logo" src="data:image/png;base64,{get_image_as_base64('img/Cadw.png')}" />
        <img class="logo" src="data:image/png;base64,{get_image_as_base64('img/NCLW.jpg')}" />
        <img class="logo" src="data:image/png;base64,{get_image_as_base64('img/ukri-ahrc-square-logo.png')}" />
        <img class="logo" src="data:image/png;base64,{get_image_as_base64('img/WJEC_CBAC_logo.svg.png')}" />
    </div>
    """,
    unsafe_allow_html=True,
     )


def app():
    query_params = st.experimental_get_query_params()
    page = query_params.get("page", [None])[0]


    if page == "demo":
        st.experimental_set_query_params(page="demo")
        demo_page()
    elif page == "analysis":
        st.experimental_set_query_params(page="analysis")
        analysis_page()
    else:
        main()


if __name__ == "__main__":
    app()
