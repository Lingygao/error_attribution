"""Current six-signal experiment definition, 2026-09-24."""
VANILLA=['vanilla_log_probability','lm_entropy']
BASE=VANILLA+['A_ctc','audio_entropy','decoding_consistency']
WORD='ctc_contrast_word'
GROUPS={
 'vanilla':VANILLA,
 'vanilla_plus_ctc_mean':VANILLA+['A_ctc'],
 'vanilla_plus_decoding_consistency':VANILLA+['decoding_consistency'],
 'all_without_contrast':BASE,
 'vanilla_plus_word_contrast':VANILLA+[WORD],
 'all_with_word_contrast':BASE+[WORD],
}
