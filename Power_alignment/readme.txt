MODIFIED POWER ALIGNMENT: IMPLEMENTATION AND FAILURE HANDLING
============================================================

Scope
-----
This project uses a modified POWER implementation to align Dutch prompt
text (PR), manual transcripts (MT), and ASR output (AO). PR-MT, PR-AO,
and MT-AO are aligned independently. Their outputs are subsequently
combined into a three-way label table using token positions. This is not
a joint three-sequence alignment algorithm.

The settings below describe the current code defaults. Individual runs
may override these settings; consult their saved task configurations.

1. Dutch-language adaptation
----------------------------
Inputs are normalized using scripts/text_normalization.py before alignment.
The pipeline uses a Dutch pronunciation lexicon and strict lexicon mode
(--lexicon-fallback strict). If a required pronunciation is missing, the
aligner reports an error instead of generating an English pronunciation.
The missing lexicon entry must be supplied before rerunning that case.

2. Constrained multiword alignment
---------------------------------
The modified aligner exposes an automatic alignment mode supporting both
many-to-one and one-to-many mappings, with configurable limits.

Default settings:
  PR-MT: one-to-one alignment.
  PR-AO: up to three-to-one or one-to-three alignment.
  MT-AO: up to three-to-one or one-to-three alignment.

For multiword candidates, the ratio between the longer and shorter
concatenated character lengths must not exceed 1.4. These constraints
limit overly broad groupings while allowing differences in word segmentation.

3. Local repair and token preservation
--------------------------------------
Adjacent substitution, insertion, and deletion regions are considered for
constrained local realignment. Candidate alignments are evaluated using
phonetic alignment costs and a multiword penalty.

Phonetic refinements are accepted only if they preserve both original
token sequences. The final local-repair output is also checked against
both input sequences. If a check fails, the preceding alignment is retained.
These checks protect token coverage and order; they do not guarantee that
independent pairwise alignments select the same occurrence of a repeated word.

4. Automatic handling of expensive or failed jobs
------------------------------------------------
Long error regions:
  By default, if either side of an error region exceeds 25 words, its
  phonetic realignment is skipped. The initial word-level Levenshtein
  alignment is retained at this stage. The subsequent local multiword
  repair stage may still process the region, so this threshold is not a
  strict limit on all subsequent computation.

Timeout:
  Each speaker/task pair runs in a separate subprocess with a default
  timeout of 300 seconds. On timeout, the subprocess is terminated and
  the task is marked as failed. Processing continues with the remaining
  tasks. The main batch runner does not automatically retry the failed
  job or replace its output with a fallback alignment.

Other failures:
  Missing inputs are recorded as skipped tasks. Nonzero subprocess exit
  codes and other exceptions are recorded as failures. Batch summary and
  error CSV files record task status and diagnostic information. Standard
  output, standard error, and task configuration files are also written
  by the runner where execution permits.

Implementation limitation:
  TimeoutExpired may expose captured output as bytes even when text mode
  is enabled. The current timeout handler writes this output directly to
  a text file. If this raises a TypeError, the outer handler records a
  generic exception instead of a timeout, and timeout diagnostics may be
  incomplete. The batch still proceeds to the remaining tasks.

5. Separate recovery procedures
-------------------------------
These procedures must be invoked separately. They are not automatic
stages of the standard batch pipeline, and their presence does not prove
that they were used for a particular experimental result.

A. Rerun a difficult alignment

  scripts/special_pr_v_alignment.py
  scripts/special_mt_v_alignment.py

  Default recovery settings:
    Timeout:                       1500 seconds
    Maximum many-to-one mapping:    2 words to 1 word
    Maximum one-to-many mapping:    1 word to 2 words
    Maximum phonetic region size:   7 words on either side
    Maximum character-length ratio: 1.4

  These scripts reduce alignment flexibility and restrict expensive
  phonetic realignment while allowing more total runtime. The settings
  are configurable through command-line arguments.



6. Main implementation files
---------------------------
  /power-asr-master/power/aligner.py
      Alignment constraints, phonetic refinement, and local repair.
  /power-asr-master/power/pronounce.py
      Pronunciation lookup and strict lexicon handling.
  /power-asr-master/power.py
      POWER command-line options.
  /text_normalization.py
      Dutch text normalization.

  /Power_alignment/special_*_alignment.py



