THREE-PASS ALIGNMENT
====================

Purpose
-------
Connect the prompt, the manual transcript, and the ASR output 
so that candidate ASR-detected reading errors can be attributed to the child or to the ASR.

PR = prompt: what the child was expected to read.
MT = manual transcript: the annotated realization of the child's speech.
AO = ASR output: what the ASR recognizer transcribed (in our project Whisper large v2/v3).

C = correct match; S = substitution; I = insertion; D = deletion.
These labels are relative to the reference and ASR hypothesis in each pass.


0. Research hypothesis
------------------------------------------------

H1. Decoding should be the main attempt of children in focused reading aloud task: motivation for filtering.
    In a prompted reading task, the child is assumed to attempt to decode
    the printed word into speech. Speech mistake mainly reflect an unsuccessful
    decoding attempt. Here "decoding" refers to the child's reading
    process, not ASR beam search. 
    Distraction like pronoucing prompt-agnostic words, 
    and other off-task speech are possible exceptions that are rare and can be filterd

H2. Transcibing speech mistake can have multiple correct transcriptions
    Since speech mistake can inherently admit multiple plausible transcriptions, 
    we adopt a more lenient selection criterion, 
    considering transcriptions acceptable as long as they are phonetically similar to the intended prompt word.
    Thus, small transcribing error for speech mistake is acceptable and only regarded as human mistake not ASR error

1. Implementation steps
-----------------------
Step 0: Normalize the inputs.
Step 1: Align PR (reference) with MT (hypothesis).
Step 2: Align PR (reference) with AO (hypothesis).
Step 3: Align MT (reference) with AO (hypothesis).
Step 4: Merge the three outputs.
For each AO token:
  a. Use its position in MT-AO to find its associated MT token(s).
  b. Use those MT token positions to retrieve their PR-MT labels.
  c. Use the AO token position to retrieve its PR-AO label.
  d. Write one combined row with the words, labels, positions, and sources.

2. Task labels
--------------
The merged table is converted to task labels as follows:

  PR-AO = I: eligible for task insertion attribution.
  PR-AO = S: eligible for task substitution attribution.

For either eligible task:
  Associated PR-MT labels are empty or all C:
    label 1 = ASR false alarm.
  At least one associated PR-MT label is non-C:
    label 0 = Human, meaning associated with a child reading error.

A Human label does not require AO to match MT exactly. 


3. Required code and resources
------------------------------
Keep these project-relative paths:

  /power-asr-master/
  /text_normalization.py 
    step 0
  /output/alignment/merged_lexicon.power.json
    Dutch pronunciation lexicon.
  /batch_power_alignment.py
    Runs PR-MT and PR-AO for ASR output( in our project Whisper output) 
  /batch_mt_v_alignment.py
    step 1, 2, 3
  /build_alignment_label_table.py
    step 4
  /batch_alignment_label_tables.py
    step 4
  /add_task_labels.py
    Core task-label rules. step 4
  scripts/batch_add_task_labels.py
    Batch wrapper for task labeling. step 4






