# Can Whisper-Based Uncertainty Signals Reveal Error Sources? An Empirical Evaluation in Child Reading Assessment
This is the code resporitory of paper: Can Whisper-Based Uncertainty Signals Reveal Error Sources? An Empirical Evaluation in Child Reading Assessment

## Abastract
Trustworthy automated reading assessment for children requires distinguishing child-produced reading errors from ASR-induced errors. Misattributing ASR errors to children can lead to unfair or misleading feedback, yet it remains unclear whether Whisper-derived uncertainty signals encode information about error source in child speech. We analyze these signals for word-level error-source attribution across Dutch and English child reading datasets. Results show that individual signals contain measurable error-source information, and that aggregating multiple signals improves AUC over single uncertainty signals. However, higher AUC does not consistently improve probability calibration, measured by expected calibration error, and performance varies across child age groups. These findings highlight both the potential and limitations of uncertainty-derived signals for reliable and fair reading assessment.

## Data Split

Jasmin is a spoken Dutch corpus containing children, elderly and non-native Dutch speakers. We benchmark a standard Dutch ASR system (trained on non-children, non-elderly native speakers) in various types of speech in Jasmin. The Jasmin corpus can be found here: (https://www.aclweb.org/anthology/L06-1141/). It available for academic research. Please contact the developers for the access to this corpus.
We provide the training and testing filename list for replication

## Adjusted Phoneme-aware alignment(power alignment) and three-pass alignment
The documents to describe Adjusted Phoneme-aware alignment(power alignment) and three-pass alignment are in folder Power_alignment/ and three-pass-alignment
The code to achieve the algorithm are in the main folder
