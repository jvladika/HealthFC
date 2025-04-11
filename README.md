Code of the research paper "[HealthFC: Verifying Health Claims with Evidence-Based Medical Fact-Checking](https://aclanthology.org/2024.lrec-main.709/)", accepted to LREC-COLING 2024, can be found in the _experiments_ folder.

The dataset _Medizin-transparent_ used for experiments in the paper can be found under _Datensatz.csv_. The dataset is free to use for research purposes under the license and terms described below.

Feel free to reach out for any questions or comments.


## Dataset Elements

- **en/de_claim**: Health claim posed in form of a research question
- **en/de_top_sentences**: Up to five most important sentences for determining claim veracity from the article text
- **en/de_explanation**: Explanation of the final claim verdict in form of a short summary paragraph

- **verdict**: Original verdict on the claim from the medical team
- **label**: Verdict mapped to one of the three labels: Supported (0), Not enough information (1), Refuted (2)

- **title**: Original title of the article
- **date**: Date of article creation or the latest update date.
- **author**: Authors of the article.
- **url**: URL that contains the full article text.


## Dataset License 
(EN) The dataset _Medizin-transparent_  is licensed under a
[Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International License][cc-by-nc-nd].

(DE) Der Datensatz _Medizin-transparent_ unterliegt den Bestimmungen einer 
[Creative Commons Namensnennung-Nicht kommerziell-Keine Bearbeitungen 4.0 International-Lizenz](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.de).

[![CC BY-NC-ND 4.0][cc-by-nc-nd-image]][cc-by-nc-nd] [![CC BY-NC-ND 4.0][cc-by-nc-nd-shield]][cc-by-nc-nd]

[cc-by-nc-nd-DE]: http://creativecommons.org/licenses/by-nc-sa/4.0/deed.de
[cc-by-nc-nd]: http://creativecommons.org/licenses/by-nc-nd/4.0/
[cc-by-nc-nd-image]: https://licensebuttons.net/l/by-nc-nd/4.0/88x31.png
[cc-by-nc-nd-shield]: https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg

## Dataset Attribution

(EN) The article texts used for construction of the dataset were written by:

- University for Continuing Education Krems (Danube University Krems))
- Medizin-transparent article authors included in the dataset: Bernd Kerschner, Jana Meixner, Teresa König, Iris Hinneburg, Julia Harlfinger, Claudia Christof, Jörg Wipplinger, Iris Mair, Verena Ahne, Tanja Wolf, Björn Bernitt, Lilith Teusch

Please ensure proper attribution when using this dataset by including the above information.

-----

(DE) Die Artikeltexte, die für die Erstellung des Datensatzes verwendet wurden, stammen von:

- Universität für Weiterbildung Krems (Donau-Universität Krems)
- Medizin-transparent-Artikel-Autor*innen, die im Datensatz vorkommen: Bernd Kerschner, Jana Meixner, Teresa König, Iris Hinneburg, Julia Harlfinger, Claudia Christof, Jörg Wipplinger, Iris Mair, Verena Ahne, Tanja Wolf, Björn Bernitt, Lilith Teusch

Bitte achten Sie bei der Verwendung dieses Datensatzes auf die korrekte Zuordnung der Daten und geben Sie die oben genannten Informationen an.

Medizin transparent wurde u.a. finanziert durch den Niederösterreichischen Gesundheits- und Sozialfonds (NÖGUS) sowie die Bundesgesundheitsagentur (BGA) in Österreich. Informationen zu diesen und weiteren Fördergebern unter https://medizin-transparent.at

## Study Citation

To cite the research study HealthFC in LaTeX bib, please use:
```
@inproceedings{vladika-etal-2024-healthfc-verifying,
    title = "{H}ealth{FC}: Verifying Health Claims with Evidence-Based Medical Fact-Checking",
    author = "Vladika, Juraj  and
      Schneider, Phillip  and
      Matthes, Florian",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italy",
    publisher = "ELRA and ICCL",
    url = https://aclanthology.org/2024.lrec-main.709,
    pages = "8095--8107",
}
```
