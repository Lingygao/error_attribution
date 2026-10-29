# Can Whisper-Based Uncertainty Signals Reveal Error Sources? An Empirical Evaluation in Child Reading Assessment
This is the code resporitory of paper: Can Whisper-Based Uncertainty Signals Reveal Error Sources? An Empirical Evaluation in Child Reading Assessment

## Abastract
Trustworthy speech diagnosis requires distinguishing child-produced
reading errors from ASR-induced errors, especially in high-stakes
applications such as child speech assessment. Misattribution of
ASR errors to children can lead to unfair or misleading feedback.
It remains unclear whether Whisper-based uncertainty signals carry
information about error source. We develop a framework for evaluating Whisper-based uncertainty signals for word-level error-source
attribution in low-resource Dutch child reading data. Results show
that single uncertainty signals can carry measurable error-source
information, although their discriminative ability varies across attribution tasks and Whisper models. Aggregating complementary
signals generally improves discrimination over vanilla uncertainty
alone. Overall discriminative performance remains limited, however. These findings highlight both the potential and limitations of
uncertainty-based signals for reliable and fair child reading assessment.

## Data Split

Jasmin is a spoken Dutch corpus containing children, elderly and non-native Dutch speakers. For our project, we use only the native child read-speech subset. The Jasmin corpus can be found here: (https://www.aclweb.org/anthology/L06-1141/). It is available for academic research. Please contact the developers for the access to this corpus.

We provide the training and testing filename lists `train.txt` and `test.txt` for replication

## Adjusted Phoneme-aware alignment (power alignment) and three-pass alignment
The **documents** to describe the implementation logic, hypothesis to support the design, failure_mode and usage of both Adjusted Phoneme-aware alignment(power alignment) and three-pass alignment are in folder `/Power_alignment/` and `/three-pass-alignment/` repectively

The **code** to achieve the algorithm are in the main folder including adjusted power alignment: `/power-asr-master` and three-pass-alignment: `/build_alignment_label_table.py`, `/add_task_labels.py` etc

## Uncertainty signals

We try to find valid Whisper uncertainty signals based on Whisper output, speech audio and prompt:

Run the following scripts independently:

`batch_lm_word_scores_from_newsegments.py`
  -> Vanilla Whisper confidence (lm_confidence) and Vanilla entropy (lm_entropy).

`batch_whisperx_alignment_scores_from_fulltext.py`
  -> CTC mean probability (A_ctc) and CTC entropy (audio_entropy). 

`batch_decoding_consistency_from_newsegments.py`
  -> Decoding_consistency.

`prepare_jasmin.py`
  -> Build a manifest using existing PR-ASR alignments and ASR segments.
     Requires the project data layout and train.txt/test.txt.

`batch_ctc_contrast.py`
  -> Generate word_contrast.csv.
     Target-aware CTC contrast = CTC score(prompt) - CTC score(ASR), using the same
     audio window. It is generated for eligible substitution groups.

`batch_merge_speaker_signal_tables.py` to merge the feature

`batch_link_task_labels_with_signals.py` to connect the merged

`transcribe_vad_segments_beam.py` and `text_normalization.py` are supporting files

notes and notice: cross-attention feature is now deprecated, relvant codes `batch_cross_attention_from_newsegments.py` are also deprecated, those results are not included in the paper

## Evaluation

We used AUC with 95% bootstrap (paird boostrap) to evaluate the performance of each single uncertainty signal
and Aggregating all singals using logistic regression (LR)

To evaluate, run the following scripts when the table, containing each word with task label and corresponding signals, is ready:

`evaluate_six_signal_lr.py`
  -> Trains LR models and evaluate on different metrics including AUC and 95% speaker-bootstrap. 

`paired_all_vs_vanilla.py`
  -> This is used when comparing different settings of LR, extracts selected paired comparisons and checks them against the previously saved pairwise results

`six_signal_config.py`
  -> select signals

`auc_ece.ipynb` is supportive file
  **notice: the evaluation can provides results of three tasks instead of two in the paper: santity check task task (correct vs incorrect classification) (not included in the paper), insertion mistake task (human vs ASR), substitution mistake task (human vs ASR)

##  Acknowledgements
This work is part of the project Responsible AI for Voice Diagnostics (RAIVD) with file number NGF.1607.22.013 of the research program NGF AiNed Fellowship Grants, which is financed by the Dutch Research Council (NWO).

  ## Model
  Transcribing: `openai/whisper-large-v2`, `openai/whisper-large-v3`
  
  Audio-text forced alignment: `GroNLP/wav2vec2-dutch-large-ft-cgn`
