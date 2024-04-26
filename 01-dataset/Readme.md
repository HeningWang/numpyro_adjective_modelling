# Preprocessing Steps

This folder contains steps for data preprocessing and manipulation. It takes raw data and raw stimuli table as input, and output prepocessed data sets for further analysis.

Input:

Data from the slider rating experiment:
- 00-slider-data-raw.csv contains raw data
- 00-slider-subj-info.csv contains subject info. This is for data exclusion.

Data from the free production experiment:
- 00-production-data-raw.csv contains raw data
- 00-production-subj-info.csv contains subject info. This is for data exclusion.

00-stimuli-table-raw.csv contains a detailed experiment design. This is relevant for the encoding of properties of objects, used for modeling and model predictions.

Steps involved:
- Remove NA values
- Data exclusion
- Subset filler items
- Encode conditions
- Encode properties of objects

TODO: Write a detailed description of how properties are generated. Ideally in Math. Because this step is crucial for generating and interpreting random states.

Output:
- 01-slider-preprocessed.csv
- 01-production-preprocessed.csv

