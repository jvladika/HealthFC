# HealthFC
The code and data for the paper "**HealthFC: A Dataset of Health Claims for Evidence-Based Medical Fact-Checking**". The dataset can be found in the CSV file _healthFC_annotated.csv_


Explanation of columns in the dataset:

- **en_xyz**: English version of the dataset.
- **de_xyz**: German version of the dataset.

  
 _
- **claim**: Health claim posed in form of a research question.
- **text**: Main text of the fact-checking article.
- **studies**: Part of the fact-checking article detailing discovered clinical studies.
- **explanation**: Explanation of the final claim verdict in form of a short summary paragraph.

_
- **sentences**: A list of sentences of the full fact-checking article (combined text and studies).
- **ids**: IDs in the **sentences** list of manually annotated, gold evidence sentences.
- Indexing the sentences in the form of _sentences[ids]_ will yield the gold evidence sentences for each claim/article.

_
- **verdict**: Original verdict on the claim from the medical team.
- **label**: Verdict mapped to one of the three labels:
  -  Supported (0)
  -  Not enough information (1)
  -  Refuted (2)

_
- **title**: Original title of the article.
- **date**: Initial date the article was posted on. Keep in mind many articles get updated with time.
- **author**: Authors of the article.
- **url**: URL that contains the article.
