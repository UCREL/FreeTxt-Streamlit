## The FreeTxt tool

Here is the [link to FreeTxt tool](https://nouran-khallaf-free-txt-home-c88nm3.streamlit.app/) which is currently under development. 

### Summary of features:
Below are the summary of the features currently available:
1.	**Data input feature**: At the moment, the tool manages two modes of input:

    a.	_Use example data_: These are a collection of example data files in different formats (.xlsx, .txt and .tsv) that are mostly for test and demo purposes.

    b.	_Upload data file_: This feature allows users to upload their data in any of the formats above.

In both cases, each file can be up to 200mb in size and multiple files upload is also allowed and can be processed simultaneously. For better memory and processing efficiency, users can select the sections of the data (i.e. columns) they wish to visualise or work with. Outputs from multiple files can be viewed in dynamically managed tabs.

2.	**Data Visualizer**: This is one of the key features of the tool and has three core components:
 
    a. _Data View_:  This allows the user to display and visualize the selected columns from the data file they wish to look at. The user can also dynamically modify the selection or the order of the columns as they wish before performing any other task on the selected columns
   
    b.	_Word Cloud_: This creates a word cloud from the content of the selected columns. It also allows the user to select the column(s) to build the word cloud from as well as the word cloud type – i.e. 'All words', 'Bigrams', 'Trigrams', '4-grams', 'Nouns', 'Proper nouns', 'Verbs', 'Adjectives', 'Adverbs', 'Numbers'
   
    c.	_Key word in Context and Collocation_: This extracts the keywords in the review text from the selected columns as well as the contexts within which they appeared in the text allowing the user to adjust the context window. It also shows the collocated words with the selected keywords

3.	**Text Summarizer**: This tool, adapted from the Welsh Summarization project, produces a basic extractive summary of the review text from the selected columns.

4.	**Sentiment Analyzer**: This feature performs sentiment classification on reviews from selected column(s) and displays a pie chart to visualize the output 

5.	**POS and Sematic Tagger** : This feature uses the PyMUSAS pipeline on Spacy to generate and display POS (CyTag) tags as well as semantic (USAS) tags. It currently works on the Ucrel-freetxt-VM as setting up Docker on the Streamlit cloud is a bit complex.

6.	**Language Identification**: We have implemented a basic language identification feature which can easily detect whether the text is written in English or Welsh.

### Contacts
- [Nouran Khallaf](https://github.com/Nouran-Khallaf)
- [Ignatius Ezeani](https://github.com/IgnatiusEzeani)
- [Paul Rayson](https://github.com/perayson)
- [Mahmoud El-Haj](https://github.com/drelhaj)
- [Dawn Knight](https://github.com/DawnKnight-Cardiff)

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons Licence" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a>
- This work with all the accompanying resources is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
