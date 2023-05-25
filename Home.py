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
            fnames = st.multiselect('Select example data file(s)', example_files, example_files[0])
            if fnames:
                return True, {fname:read_file(os.path.join(EXAMPLES_DIR, fname), file_source) for fname in fnames}
            else:
                return False, st.info('''**NoFileSelected:** Please select at least one file from the sidebar list.''', icon="ℹ️")
        
        elif file_source=='uploaded': # Todo: Consider a maximum number of files for memory management. 
            uploaded_files = st.file_uploader("Upload your data file(s)", accept_multiple_files=True, type=['txt','tsv','xlsx', 'xls'])
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

###############################################Sentiment analysis###########################################
# --------------------Sentiments----------------------

###########Ployglot Welsh

def preprocess_text(text):
    # remove URLs, mentions, and hashtags
    text = re.sub(r"http\S+|@\S+|#\S+", "", text)

    # remove punctuation and convert to lowercase
    text = re.sub(f"[{re.escape(''.join(PUNCS))}]", "", text.lower())

    # remove stopwords
    text = " ".join(word for word in text.split() if word not in STOPWORDS)

    return text

# define function to analyze sentiment using Polyglot for Welsh language
@st.cache(allow_output_mutation=True)
def analyze_sentiment_welsh_polyglot(input_text):
    # preprocess input text and split into reviews
    reviews = input_text.split("\n")

    text_sentiment = []
    for review in reviews:
        review = preprocess_text(review)
        if review:
            text = Text(review, hint_language_code='cy')

            # calculate sentiment polarity per word
            sentiment_polarity_per_word = []
            for word in text.words:
                word_sentiment_polarity = word.polarity
                sentiment_polarity_per_word.append(word_sentiment_polarity)

            # calculate overall sentiment polarity
            overall_sentiment_polarity = sum(sentiment_polarity_per_word)

            # classify sentiment based on a threshold
            if overall_sentiment_polarity > 0.2:
                sentiment = "positive"
            elif overall_sentiment_polarity < -0.2:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            text_sentiment.append((review, sentiment, overall_sentiment_polarity))

    return text_sentiment

from textblob import TextBlob
# define function to analyze sentiment using TextBlob for Welsh language
@st.cache(allow_output_mutation=True)
def analyze_sentiment_welsh(input_text):
    # preprocess input text and split into reviews
    reviews = input_text.split("\n")

    text_sentiment = []
    for review in reviews:
        review = preprocess_text(review)
        if review:
            # analyze sentiment using TextBlob
            text_blob = TextBlob(review)

            # calculate overall sentiment polarity
            overall_sentiment_polarity = text_blob.sentiment.polarity

            # classify sentiment based on a threshold
            if overall_sentiment_polarity > 0.2:
                sentiment = "positive"
            elif overall_sentiment_polarity < -0.2:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            text_sentiment.append((review, sentiment, overall_sentiment_polarity))

    return text_sentiment


# --------------------Sentiments----------------------

###########Bert English
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
def preprocess_text(text):
    # remove URLs, mentions, and hashtags
    text = re.sub(r"http\S+|@\S+|#\S+", "", text)

    # remove punctuation and convert to lowercase
    text = re.sub(f"[{re.escape(''.join(PUNCS))}]", "", text.lower())

    # remove stopwords
    text = " ".join(word for word in text.split() if word not in STOPWORDS)

    return text



@st.cache(allow_output_mutation=True)
def analyze_sentiment(input_text,num_classes, max_seq_len=512):
    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

    # preprocess input text and split into reviews
    reviews = input_text.split("\n")

    # predict sentiment for each review
    sentiments = []
    for review in reviews:
        review = preprocess_text(review)
        if review:
            # Tokenize the review
            tokens = tokenizer.encode(review, add_special_tokens=True, truncation=True)

            # If the token length exceeds the maximum, split into smaller chunks
            token_chunks = []
            if len(tokens) > max_seq_len:
                token_chunks = [tokens[i:i + max_seq_len] for i in range(0, len(tokens), max_seq_len)]
            else:
                token_chunks.append(tokens)

            # Process each chunk
            sentiment_scores = []
            for token_chunk in token_chunks:
                input_ids = torch.tensor([token_chunk])
                attention_mask = torch.tensor([[1] * len(token_chunk)])

                # Run the model
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                scores = outputs.logits.softmax(dim=1).detach().numpy()[0]
                sentiment_scores.append(scores)

            # Aggregate the scores
            avg_scores = np.mean(sentiment_scores, axis=0)
            sentiment_labels = ['Very negative', 'Negative', 'Neutral', 'Positive', 'Very positive']
            sentiment_index = avg_scores.argmax()

            if num_classes == 3:
                sentiment_labels_3 = ['Negative', 'Neutral', 'Positive']
                if sentiment_index < 2:
                    sentiment_index = 0  # Negative
                elif sentiment_index > 2:
                    sentiment_index = 2  # Positive
                else:
                    sentiment_index = 1  # Neutral
                sentiment_label = sentiment_labels_3[sentiment_index]
            else:
                sentiment_label = sentiment_labels[sentiment_index]

            sentiment_score = avg_scores[sentiment_index]
            sentiments.append((review, sentiment_label, sentiment_score))

    return sentiments

#####
import plotly.graph_objs as go
import plotly.io as pio

def plot_sentiment(df):
    # count the number of reviews in each sentiment label
    counts = df['Sentiment Label'].value_counts()

    # create the bar chart
    data = [
        go.Bar(
            x=counts.index,
            y=counts.values,
            text=counts.values,
            textposition='auto',
            marker=dict(color='rgb(63, 81, 181)')
        )
    ]

    # set the layout
    layout = go.Layout(
        title='Sentiment Analysis Results',
        xaxis=dict(title='Sentiment Label'),
        yaxis=dict(title='Number of Reviews'),
        plot_bgcolor='white',
        font=dict(family='Arial, sans-serif', size=14, color='black'),
        margin=dict(l=50, r=50, t=80, b=50)
    )

    # create the figure
    fig = go.Figure(data=data, layout=layout)
    #Save the plot to an image
    
    pio.write_image(fig, 'Bar_fig.png', format='png', width=800, height=600, scale=2)
    # show the plot
    st.plotly_chart(fig)
    buffer = io.StringIO()
    fig.write_html(buffer, include_plotlyjs='cdn')
    html_bytes = buffer.getvalue().encode()

    st.download_button(
            label='Download Bar Chart',
            data=html_bytes,
            file_name='Sentiment_analysis_bar.html',
            mime='text/html'
        )





from streamlit_plotly_events import plotly_events
import plotly.express as px

def plot_sentiment_pie(df):

    # count the number of reviews in each sentiment label
    counts = df['Sentiment Label'].value_counts()

    # calculate the proportions
    proportions = counts / counts.sum()

    # create the pie chart
    data = [
        go.Pie(
            labels=proportions.index,
            values=proportions.values,
            hole=0.4,
            marker=dict(colors=px.colors.qualitative.Plotly)
        )
    ]

    # set the layout
    layout = go.Layout(
        title='Sentiment Analysis Results',
        plot_bgcolor='white',
        font=dict(family='Arial, sans-serif', size=14, color='black'),
        margin=dict(l=50, r=50, t=80, b=50),

    )

    fig = go.Figure(data=data, layout=layout)
    selected_points = plotly_events(fig, select_event=True)
    st.write('The figure displays the sentiment analysis of the data, you can press on any part of the graph to display the data')
    if selected_points:
        # filter the dataframe based on the selected point
        point_number = selected_points[0]['pointNumber']
        sentiment_label = proportions.index[point_number]
        df = df[df['Sentiment Label'] == sentiment_label]
        st.write(f'The proportion of " {sentiment_label} "')
        st.dataframe(df,use_container_width = True)
    
    # update the counts and proportions based on the filtered dataframe
    counts = df['Sentiment Label'].value_counts()
    proportions = counts / counts.sum()

    # update the pie chart data
    #fig.update_traces(labels=proportions.index, values=proportions.values)
    #Save the plot to an image
    pio.write_image(fig, 'Pie_fig.png', format='png', width=800, height=600, scale=2)
	
    buffer = io.StringIO()
    fig.write_html(buffer, include_plotlyjs='cdn')
    html_bytes = buffer.getvalue().encode()

    st.download_button(
        label='Download Pie Chart',
        data=html_bytes,
        file_name='Sentiment_analysis_pie.html',
        mime='text/html'
    )
    
nlp = spacy.load('en_core_web_sm-3.2.0')  
nlp.max_length = 9000000
######generate the scatter text 

def generate_scattertext_visualization(analysis):
    # Get the DataFrame with sentiment analysis results
    df = analysis
    # Parse the text using spaCy
    df['ParsedReview'] = df['Review'].apply(nlp)

    # Create a Scattertext Corpus
    corpus = tt.CorpusFromParsedDocuments(
        df,
        category_col="Sentiment Label",
        parsed_col="ParsedReview"
    ).build()
    
    
    term_scorer = tt.RankDifference()
    html = tt.produce_scattertext_explorer(
     corpus,
    category="Positive",
    not_categories=df["Sentiment Label"].unique().tolist(),
    minimum_term_frequency=5,
    pmi_threshold_coefficient=5,
    width_in_pixels=1000,
    metadata=df["Sentiment Label"],
    term_scorer=term_scorer
       ) 

    # Save the visualization as an HTML file
    with open("scattertext_visualization.html", "w") as f:
        f.write(html)
########## Generate the PDF#############
# Add a state variable to store the generated PDF data
generated_pdf_data = None
#---------------------------------------------------------------------------------------
def header(canvas, doc):
    # Add logo and title in a table
    logo_path = "img/FreeTxt_logo.png" 
    logo = PilImage.open(logo_path)
    logo_width, logo_height = logo.size
    aspect_ratio = float(logo_height) / float(logo_width)
    logo = ReportLabImage(logo_path, width=100, height=int(100 * aspect_ratio))
    title_text = "Sentiment Analysis Report"
    title_style = ParagraphStyle("Title", fontSize=18, alignment=TA_LEFT)
    title = Paragraph(title_text, title_style)
    header_data = [[logo, title]]
    header_table = Table(header_data)
    header_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (0, 0), 'LEFT'),
        ('ALIGN', (1, 0), (1, 0), 'LEFT'),
        ('VALIGN', (0, 0), (1, 0), 'TOP'),
        ('LEFTPADDING', (1, 0), (1, 0), 1),
    ]))
    w, h = header_table.wrap(doc.width, doc.topMargin)
    header_table.drawOn(canvas, doc.leftMargin, doc.height + doc.topMargin - h + 20)


#---------------------------------------------------------------------------------------

# Function to convert DataFrame to a CSV file and allow it to be downloaded
def download_csv(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="Sentiment-analysis.csv">Download CSV File</a>'
    return href
#-----------------------------------------------------------------------------------------



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
    bt1,bt2,bt3,bt4,bt5,bt6,bt7,bt8,bt9,bt10,bt11,bt12 = st.columns([2,2,2,2,2,2,2,2,2,2,2,2])
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
    #####---------------get the data
    option = st.radio(MESSAGES[lang][0], (MESSAGES[lang][1], MESSAGES[lang][2]))
    if option == MESSAGES[lang][1]: input_data = get_data()
    elif option == MESSAGES[lang][2]: input_data = get_data(file_source='uploaded')
    else: pass
    status, data = input_data
    tab1, tab2,tab3 = st.tabs(["📈 Meaning analysis",'💬 Keyword scatter','📥 Download pdf'])
    with tab1:
        if status:
        	num_classes = st.radio('How do you want to categorize the sentiments?', ('3 Class Sentiments (Positive, Neutral, Negative)', '5 Class Sentiments (Very Positive, Positive, Neutral, Negative, Very Negative)'))
        	num_classes = 3 if num_classes.startswith("3") else 5
        # With tabbed multiselect
        filenames = list(data.keys())
       
        for i in range(len(filenames)):
          
                _, df = data[filenames[i]]
                df = select_columns(df, key=i).astype(str)
                if df.empty:
                    st.info('''**NoColumnSelected 🤨**: Please select one or more columns to analyse.''', icon="ℹ️")
                else:
                        
                        input_text = '\n'.join(['\n'.join([str(t) for t in list(df[col]) if str(t) not in STOPWORDS and str(t) not in PUNCS]) for col in df])
                        
                        language = detect_language(df)
                      
                        
                        if language == 'en':
                            sentiments = analyze_sentiment(input_text,num_classes)
                            analysis = pd.DataFrame(sentiments, columns=['Review', 'Sentiment Label', 'Sentiment Score'])
                            plot_sentiment_pie(analysis)
                            plot_sentiment(analysis)
                      
                        elif language == 'cy':
                            #sentiments = analyze_sentiment_welsh(input_text)
                            sentiments = analyze_sentiment(input_text,num_classes)
                            analysis = pd.DataFrame(sentiments, columns=['Review', 'Sentiment Label', 'Sentiment Score'])
                            plot_sentiment_pie(analysis)
                            plot_sentiment(analysis)
                       
    with tab2:
                         #### interactive dataframe
                         gb = GridOptionsBuilder.from_dataframe(analysis)
                         gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
                         gb.configure_side_bar() #Add a sidebar
                         gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
                         gridOptions = gb.build()

                         grid_response = AgGrid(
                              analysis,
                              gridOptions=gridOptions,
                               data_return_mode='AS_INPUT', 
                            update_mode='MODEL_CHANGED', 
                             fit_columns_on_grid_load=False,
    
                                  enable_enterprise_modules=True,
                             height=350, 
                              width='100%',
                              reload_data=True
                                                )
                         data = grid_response['data']
                         selected = grid_response['selected_rows'] 
                         df = pd.DataFrame(selected) #Pass the selected rows to a new dataframe df
                         # Add a button to download the DataFrame as a CSV file
                         if st.button('Download CSV'):
                                    st.markdown(download_csv(analysis), unsafe_allow_html=True)

			
			
			###scattertext
                         st.header('Scatter Text')
                                                  # Copy the scattertext_visualization.html to a temporary file
                         st.write('For better reprentation we recommend selecting 3 sentiment classes')
                         st.write('The 2,000 most sentiment-associated uni grams are displayed as points in the scatter plot. Their x- and y- axes are the dense ranks of their usage in positive vs negative and neutral respectively.')
                         generate_scattertext_visualization(analysis)
                         scattertext_html_path='scattertext_visualization.html'
                         tmp_scattertext_path = "tmp_scattertext_visualization.html"
                         shutil.copyfile(scattertext_html_path, tmp_scattertext_path)

                         # Add a download button for the Scattertext HTML file
                         with open(tmp_scattertext_path, "rb") as file:
                                    scattertext_html_data = file.read()

                         st.download_button(
                             label="Download Scattertext Visualization HTML",
                             data=scattertext_html_data,
                          file_name="scattertext_visualization.html",
                            mime="text/html",
                                )
                         
                         
                         HtmlFile = open("scattertext_visualization.html", 'r', encoding='utf-8')
                         source_code = HtmlFile.read() 
                         print(source_code)
                         components.html(source_code,height = 1500,width = 800)
			 

                         
    with tab3:
                    try:
                     # Check if the DataFrame exists
                       if analysis is not None:
			#####pdf_generator
                        data_list_checkbox = st.checkbox("Include Data List as a Table")
                        sentiment_pie_checkbox = st.checkbox("Include Sentiment Pie Graph")
                        sentiment_bar_checkbox = st.checkbox("Include Sentiment Bar Graph")
                       # scatter_text_checkbox = st.checkbox("Include Scatter Text")
                        generate_pdf_checkbox = st.checkbox("Generate PDF report")
			
                        # Create the PDF
                            
                        buffer = BytesIO()
                        doc = BaseDocTemplate(buffer, pagesize=A4,topMargin=1.5 * inch, showBoundary=0)

                        # Create the frame for the content
                        frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='normal')

    
                        # Create a PageTemplate with the header
                        template = PageTemplate(id='header_template', frames=frame, onPage=header)
                        doc.addPageTemplates([template])
                        elements = []
                        # Add a spacer between header and input text
                        elements.append(Spacer(1, 20))
  

			# Add content based on selected checkboxes
                        if data_list_checkbox:
  

                            column_names = ['Review', 'Sentiment Label', 'Sentiment Score']
                            table_data = [column_names] + analysis[column_names].values.tolist()
                            col_widths = [200, 100, 100]  # Adjust these values according to your needs
                            wrapped_cells = []

                            styles = getSampleStyleSheet()
                            cell_style_normal = ParagraphStyle(name='cell_style_normal', parent=styles['Normal'], alignment=1)
                            cell_style_header = ParagraphStyle(name='cell_style_header', parent=styles['Normal'], alignment=1, textColor=colors.whitesmoke, backColor=colors.grey, fontName='Helvetica-Bold', fontSize=14, spaceAfter=12)

                            wrapped_data = []
                            for row in table_data:
                                  wrapped_cells = []
                                  for i, cell in enumerate(row):
                                       cell_style = cell_style_header if len(wrapped_data) == 0 else cell_style_normal
                                       wrapped_cell = Paragraph(str(cell), style=cell_style)
                                       wrapped_cells.append(wrapped_cell)
                                  wrapped_data.append(wrapped_cells)
                            table = Table(wrapped_data, colWidths=col_widths)

                            table.setStyle(TableStyle([
                                  ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                              ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),

                             ('ALIGN', (0, 0), (-1, -1), 'CENTER'),

                                  ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                  ('FONTSIZE', (0, 0), (-1, 0), 14),

                                   ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                           ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                                  ('GRID', (0, 0), (-1, -1), 1, colors.black),
       
                                 ('VALIGN', (0, 0), (-1, -1), 'TOP'),

                                         ]))
                            elements.append(table)
                            elements.append(Spacer(1, 20))


                        if sentiment_pie_checkbox:
                        # Add the sentiment pie graph
                        
                           pie_graph_path = "Pie_fig.png"
                           if os.path.exists(pie_graph_path):                
                              pie_graph = ReportLabImage(pie_graph_path, width= 325, height =250)
                              elements.append(pie_graph)
                              elements.append(Spacer(1, 20))
                           else:
                             st.error("Sentiment Pie Graph image not found")

                        if sentiment_bar_checkbox:
                         # Add the sentiment bar graph
                           
                           bar_graph_path = "Bar_fig.png"
                           if os.path.exists(bar_graph_path):
                               bar_graph = ReportLabImage(bar_graph_path, width= 325, height =250)
                               elements.append(bar_graph)
                               elements.append(Spacer(1, 20))
                           else:
                              st.error("Sentiment Bar Graph image not found")

                        #if scatter_text_checkbox:
                         # Add the scatter text
                               ###convert the html to image

                           #     scattertext_html_path = "scattertext_visualization.html"
                            #    scattertext_image_path = "scattertext_visualization.png"


                           #     html_to_image_weasyprint(scattertext_html_path, scattertext_image_path)


                               
                                
                                
                            #    scatter_text = ReportLabImage(scattertext_image_path, width=800, height=800)
                             #   elements.append(scatter_text)
                             #   elements.append(Spacer(1, 20))

                       else:
                           st.error("DataFrame not found")

                    except Exception as e:
                            st.error(f"An error occurred: {str(e)}")
                    if generate_pdf_checkbox:

        
                                # Build PDF
	
                            doc.build(elements)
                            buffer.seek(0)
                            generated_pdf_data = buffer.read()

   # Display the download button only after generating the report
                    if generated_pdf_data:
                              st.download_button("Download PDF", generated_pdf_data, "report_positiveandnegative.pdf", "application/pdf")
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
