MESSAGES = {
    'cy.md': """
            - Mae’r adnodd hwn yn rhan o brosiect [Adnodd Creu Crynodebau](https://corcencc.org/acc/) (ACC)!
            - Mae’r adnodd echdynnol yn cynhyrchu crynodeb echdynnol syml gan ddefnyddio algorithm  [TextRank](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf).
            - Mae’r adnodd haniaethol yn ceisio 'deall' y testun er mwyn creu crynodeb heb gopïo’r testun gwreiddiol. Mae’n seiliedig ar seilwaith [Text-to-Text-Transfer-Tranformer (T5)](https://arxiv.org/pdf/1910.10683.pdf) ac fe’i chrewyd gan addasu model mT5 Google. Gan ystyried cymhlethdod yr adnodd hwn, mae angen datblygiad pellach arno.
            - Mae’r set ddata ar gael drwy [GitHub](https://github.com/UCREL/welsh-summarization-dataset).
            """,
    'en.md': """
            - This tool is part of the [Welsh Summarization Creator](https://corcencc.org/acc/) (WSC) project!
            - The *Extractive* tool produces a simple extractive summarisation with the [TextRank](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf) algorithm.
            - The *Abstractive* tool tries to 'understand' the text and create a summary without copying the original. This is based on the [Text-to-Text-Transfer-Tranformer (T5)](https://arxiv.org/pdf/1910.10683.pdf) architecture and was created by adapting the Google mT5 model. Given the complexity of this tool, it requires further development.
            - The dataset is available through [GitHub](https://github.com/UCREL/welsh-summarization-dataset).
             """,
    'cy.ext.md': '#### 🌷 Adnodd Creu Crynodebau Echdynnol',
    'en.ext.md': '#### 🌷 Extractive Summariser',
    'cy.abs.md': '#### 🌷 Adnodd Creu Crynodebau Haniaethol 0.1 (Alffa)',
    'en.abs.md': '#### 🌷 Abstractive Summariser 0.1 (Alpha)',
    'cy.sb.sl': 'Dewiswch gymhareb y crynodeb [10% i 50%]:',
    'en.sb.sl': 'Select summary ratio [10% to 50%]',
    'cy.button': 'Crynhoi👈',
    'en.button': 'Summarize👈',
    'cy.info.title': 'ℹ️ - Gwybodaeth am yr ap hwn',
    'en.info.title': 'ℹ️ - About this app',
    'cy.summary.type': 'Math o grynodeb',
    'en.summary.type': 'Summary type',
    'cy.abstractive': 'Haniaethol',
    'en.abstractive': 'Abstractive',
    'cy.extractive': 'Echdynnol',
    'en.extractive': 'Extractive',
    'cy.abs.warning': 'Gall hyn gymryd peth amser. Diolch am fod yn amyneddgar 😉.',
    'en.abs.warning': 'This may take a while. Please bear with us 😉.',
    'cy':['Sut ydych chi am fewnbynnu eich testun?',
          'Defnyddiwch destun enghreifftiol',
          'Rhowch eich testun eich hun',
          'Uwchlwythwch ffeil destun',
          'Dewiswch destun enghreifftiol:',
          'Crynhowch y testun enghreifftiol yn y blwch:',
          'Uwchlwythwch ffeil destun',
          "Crynhoi testun wedi'i uwchlwytho:",
          'Teipiwch neu gludwch eich testun yn y blwch testun',
          'Rhowch eich testun...',
          'Defnyddiwch destun enghreifftiol'],
    'en':['How do you want to input your text?',
          'Use an example data',
          'Upload data file',
          'Paste copied text',
          'Select example text:',          
          'Analyse example text in the box:',
          'Upload data file',
          'Analyse uploaded text:', 
          'Type or paste your text into the text box:',
          'Please enter your text...']}