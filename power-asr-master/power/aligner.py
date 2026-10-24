from __future__ import division
from collections import deque
from power.levenshtein import Levenshtein, ExpandedAlignment, AlignLabels
from power.phonemes import Phonemes
from power.pronounce import PronouncerType, PronouncerBase, PronouncerLex, LexiconFallback

class TokType:
    WordBoundary = 1
    SyllableBoundary = 2
    Phoneme = 3
    Empty = 0
    
    @staticmethod
    def checkAnnotation(tok):
        tt = TokType.Phoneme
        if not tok:
            tt = TokType.Empty
        elif tok == '|':
            tt = TokType.WordBoundary
        elif tok == '#':
            tt = TokType.SyllableBoundary
        return tt
    
class CharToWordAligner:
    def __init__(self, ref, hyp, lowercase=False):
        self.refwords = ref.strip()
        self.hypwords = hyp.strip()
        self.ref = ref
        self.hyp = hyp
        self.lowercase = lowercase
        self.char_align = None
        self.word_align = None
        
    def charAlign(self):
        ref_chars = [x for x in self.ref] + [' ']
        hyp_chars = [x for x in self.hyp] + [' ']
        
        lev = Levenshtein.align(ref_chars, hyp_chars, lowercase=self.lowercase, reserve_list=set([' ']))
        lev.editops()
        self.char_align = lev.expandAlign()
        return self.char_align
        
    def charAlignToWordAlign(self):
        if not self.char_align:
            raise Exception("char_align is None")
        
        ref_word_align = []
        hyp_word_align = []
        align_word = []
        
        tmp_ref_word = []
        tmp_hyp_word = []
        
        for i in range(len(self.char_align.align)):
            ref_char = self.char_align.s1[i]
            hyp_char = self.char_align.s2[i]
            align_char = self.char_align.align[i] 

            # check if both words are completed
            # There are a few of ways this could happen:
            if ((align_char == AlignLabels.correct and ref_char == ' ') or
                (align_char == AlignLabels.deletion and ref_char == ' ') or
                (align_char == AlignLabels.insertion and hyp_char == ' ')):

                ref_word = ''.join(tmp_ref_word)
                hyp_word = ''.join(tmp_hyp_word)
                
                if ref_word or hyp_word:
                    ref_word_align.append(ref_word)
                    hyp_word_align.append(hyp_word)
                    tmp_ref_word = []
                    tmp_hyp_word = []
                    
                    # Check align type
                    if ref_word and hyp_word:
                        if ref_word == hyp_word:
                            align_word.append(AlignLabels.correct)
                        else:
                            align_word.append(AlignLabels.substitution)
                    elif ref_word:
                        align_word.append(AlignLabels.deletion)
                    else:
                        align_word.append(AlignLabels.insertion)
                continue
            
            # Read current chars and check if one of the words is complete
            if ref_char == ' ':
                if len(tmp_ref_word) > 1:
                    # Probably a D
                    ref_word = ''.join(tmp_ref_word)
                    ref_word_align.append(ref_word)
                    hyp_word_align.append('')
                    tmp_ref_word = []
                    align_word.append(AlignLabels.deletion)
            else:
                tmp_ref_word.append(ref_char)
            
            if hyp_char == ' ':
                if len(tmp_hyp_word) > 1:
                    # Probably an I
                    hyp_word = ''.join(tmp_hyp_word)
                    ref_word_align.append('')
                    hyp_word_align.append(hyp_word)
                    tmp_hyp_word = []
                    align_word.append(AlignLabels.insertion)
            else:
                tmp_hyp_word.append(hyp_char)
            
        self.word_align = ExpandedAlignment(ref_word_align, hyp_word_align, align_word, lowercase=self.lowercase)
        return self.word_align	
            
class PowerAligner:
    class AlignmentMode:
        Auto = 'auto'
        ManyToOne = 'many-to-one'
        OneToMany = 'one-to-many'

        @classmethod
        def choices(cls):
            return (cls.Auto, cls.ManyToOne, cls.OneToMany)

    # Exclusive tokens that can only align to themselves; not other members in this set.
    reserve_list = set(['|', '#'])
    
    # R-sounds
    r_set = set.union(set('r'), Phonemes.r_vowels)
    exclusive_sets = [Phonemes.vowels, Phonemes.consonants, r_set]
    
    phoneDistPenalty    = 0.25
    phoneDistPenalt16ySet = set(['|'])
    repairMultiwordSlotPenalty = 6.0
    
    def __init__(self, ref, hyp, lowercase=False, verbose=False,
                pronounce_type=PronouncerType.Lexicon,
                lexicon=None,
                word_align_weights=Levenshtein.wordAlignWeights,
                alignment_mode=AlignmentMode.ManyToOne,
                max_many_to_one=None,
                max_one_to_many=None,
                max_multiword_char_ratio=None,
                max_power_region_words=None,
                multiword_slot_penalty=repairMultiwordSlotPenalty,
                lexicon_fallback=LexiconFallback.English):
        if not ref:
            raise Exception("No reference file.\nref: {0}\nhyp: {1}".format(ref, hyp))
        if alignment_mode not in self.AlignmentMode.choices():
            raise ValueError("alignment_mode must be one of: {0}".format(
                ', '.join(self.AlignmentMode.choices())))
        if max_many_to_one is not None and max_many_to_one < 1:
            raise ValueError("max_many_to_one must be a positive integer")
        if max_one_to_many is not None and max_one_to_many < 1:
            raise ValueError("max_one_to_many must be a positive integer")
        if max_multiword_char_ratio is not None and max_multiword_char_ratio < 1:
            raise ValueError("max_multiword_char_ratio must be at least 1")
        if max_power_region_words is not None and max_power_region_words < 1:
            raise ValueError("max_power_region_words must be a positive integer")
        if multiword_slot_penalty is not None and multiword_slot_penalty < 0:
            raise ValueError("multiword_slot_penalty must be non-negative")

        if pronounce_type == PronouncerType.Lexicon:
            self.pronouncer = PronouncerLex(lexicon, fallback=lexicon_fallback)
        else:
            self.pronouncer = PronouncerBase()
        
        self.ref = [x for x in ref.strip().split() if x]
        self.hyp = [x for x in hyp.strip().split() if x]
        self.refwords = ' '.join(self.ref)
        self.hypwords = ' '.join(self.hyp)
        
        self.lowercase = lowercase
        self.verbose = verbose
        self.alignment_mode = alignment_mode
        self.max_many_to_one = max_many_to_one
        self.max_one_to_many = max_one_to_many
        self.max_multiword_char_ratio = max_multiword_char_ratio
        self.max_power_region_words = max_power_region_words
        self.multiword_slot_penalty = multiword_slot_penalty
        
        # Perform word alignment
        lev = Levenshtein.align(self.ref, self.hyp, lowercase=self.lowercase, weights=word_align_weights)
        lev.editops()
        self.wer_alignment = lev.expandAlign()
        self.wer, self.wer_components = self.wer_alignment.error_rate()
        
        # Used for POWER alignment
        self.power_alignment = None
        self.power = None
        self.power_components = None
        
        # Used to find potential error regions
        self.split_regions = None
        self.error_indexes = None
        self.phonetic_alignments = None
        self.phonetic_lev = None
        
    def align(self):
        # Find the error regions that may need to be realigned
        self.split_regions, self.error_indexes = self.wer_alignment.split_error_regions()
        self.phonetic_alignments = [None] * len(self.split_regions)

        for error_index in self.error_indexes:
            seg = self.split_regions[error_index]
            ref_words = seg.s1_tokens()
            hyp_words = seg.s2_tokens()

            if (self.max_power_region_words is not None and
                    max(len(ref_words), len(hyp_words)) > self.max_power_region_words):
                self.phonetic_alignments[error_index] = None
                continue

            ref_phones = self.pronouncer.pronounce(ref_words)
            hyp_phones = self.pronouncer.pronounce(hyp_words)

            power_seg_alignment, self.phonetic_alignments[error_index] = PowerAligner.phoneAlignToWordAlign(ref_words, hyp_words, 
                ref_phones, hyp_phones, alignment_mode=self.alignment_mode,
                max_many_to_one=self.max_many_to_one,
                max_one_to_many=self.max_one_to_many,
                max_multiword_char_ratio=self.max_multiword_char_ratio)

            # Replace only if phone-to-word realignment preserves all source tokens.
            if (ref_words == power_seg_alignment.s1_tokens() and
                    hyp_words == power_seg_alignment.s2_tokens()):
                self.split_regions[error_index] = power_seg_alignment
            else:
                self.phonetic_alignments[error_index] = None

        # Merge the alignment segments back together.
        self.power_alignment = ExpandedAlignment(self.split_regions[0].s1, self.split_regions[0].s2, 
                self.split_regions[0].align, 
                self.split_regions[0].s1_map, self.split_regions[0].s2_map, lowercase=self.lowercase)
        for i in range(1, len(self.split_regions)):
            self.power_alignment.append_alignment(self.split_regions[i])
        unrepaired_power_alignment = self.power_alignment
        repaired_power_alignment = PowerAligner._merge_adjacent_multiword_errors(
            self.power_alignment, self.max_many_to_one, self.max_one_to_many,
            self.max_multiword_char_ratio, pronouncer=self.pronouncer,
            multiword_slot_penalty=self.multiword_slot_penalty)
        if (repaired_power_alignment.s1_string() == self.refwords and
                repaired_power_alignment.s2_string() == self.hypwords):
            self.power_alignment = repaired_power_alignment
        else:
            self.power_alignment = unrepaired_power_alignment
        
        # Get the alignment score
        self.power, self.power_components = self.power_alignment.error_rate()
                
        assert self.hypwords == self.power_alignment.s2_string(), "hyp mismatch:\n{0}\n{1}".format(self.hypwords, self.power_alignment.s2_string())
        assert self.refwords == self.power_alignment.s1_string(), "ref mismatch:\n{0}\n{1}".format(self.refwords, self.power_alignment.s1_string())
  
    # TODO: Make this simpler (and maybe recursive)
    @classmethod
    def phoneAlignToWordAlign(cls, ref_words, hyp_words, ref_phones, hyp_phones, break_on_syllables=True,
                              alignment_mode=AlignmentMode.ManyToOne,
                              max_many_to_one=None,
                              max_one_to_many=None,
                              max_multiword_char_ratio=None):
        if alignment_mode not in cls.AlignmentMode.choices():
            raise ValueError("alignment_mode must be one of: {0}".format(
                ', '.join(cls.AlignmentMode.choices())))
        if max_many_to_one is not None and max_many_to_one < 1:
            raise ValueError("max_many_to_one must be a positive integer")
        if max_one_to_many is not None and max_one_to_many < 1:
            raise ValueError("max_one_to_many must be a positive integer")
        if max_multiword_char_ratio is not None and max_multiword_char_ratio < 1:
            raise ValueError("max_multiword_char_ratio must be at least 1")

        ref_word_span = (0, len(ref_words))
        hyp_word_span = (0, len(hyp_words))
        
        # Perform Levenshtein Alignment
        lev = Levenshtein.align(ref=ref_phones, 
                            hyp=hyp_phones,
                            reserve_list=PowerAligner.reserve_list, 
                            exclusive_sets=PowerAligner.exclusive_sets,
                            weights=Levenshtein.wordAlignWeights) #, 
                            #dist_penalty=PowerAligner.phoneDistPenalty, dist_penalty_set=Levenshtein.wordAlignWeights)				
        phone_align = lev.expandAlignCompact()
        
        worklist = list()
        worklist.append((ref_word_span, hyp_word_span, phone_align))
        
        full_reference = list()
        full_hypothesis = list()
        full_alignment = list()
        full_phone_align = list()
        
        while worklist:
            # Take the next set of sequence boundaries off the worklist
            ref_word_span, hyp_word_span, phone_align = worklist.pop()
            ref_word_index, ref_word_limit = ref_word_span
            hyp_word_index, hyp_word_limit = hyp_word_span
            
            # TODO: Currently only checking in the forward direction
            ref_word_builder = []  # Temp storage of words in alignment span
            hyp_word_builder = []
            
            ref_word_iter = enumerate(ref_words[ref_word_span[0]:ref_word_span[1]])  # Iterates through the surface words
            hyp_word_iter = enumerate(hyp_words[hyp_word_span[0]:hyp_word_span[1]])
            
            ref_aligned = []  # Finalized alignments
            hyp_aligned = []
            alignment = []  # Finalized alignment labels
            
            ref_extra_syllable_word_index = None  # Used for marking words mapping to extra syllables in alignment.
            hyp_extra_syllable_word_index = None
            ref_syllable_count = 0
            hyp_syllable_count = 0
        
            ref_word_started = False  # Indicates whether a word is already accounted for in the alignment when a phoneme is reached.
            hyp_word_started = False
            
            advance_worklist = False
            commit_alignment = False
            
            for i in range(len(phone_align.align)):
                ref_type = TokType.checkAnnotation(phone_align.s1[i])
                hyp_type = TokType.checkAnnotation(phone_align.s2[i])
                
                # Check if word boundaries are reached, both on ref an hyp -- or the case where no more symbols can be read.
                if (i == len(phone_align.align) - 1) or (ref_type == TokType.WordBoundary and ref_type == hyp_type):
                    align_tok = None
                    # Only write outputs if either the ref or the hyp has scanned some words.
                    if ref_word_builder:
                        if hyp_word_builder:
                            align_tok = AlignLabels.substitution if ref_word_builder != hyp_word_builder else AlignLabels.correct
                        else:
                            align_tok = AlignLabels.deletion
                    elif hyp_word_builder:
                        align_tok = AlignLabels.insertion
    
                    if align_tok:
                        # Add the remainder to the worklist
                        ref_word_span_next = (ref_word_index + len(ref_word_builder), ref_word_limit)
                        hyp_word_span_next = (hyp_word_index + len(hyp_word_builder), hyp_word_limit)
                        phone_align_next = phone_align.subsequence(i, phone_align.length(), preserve_index=False)
                        worklist.append((ref_word_span_next, hyp_word_span_next, phone_align_next))
                        
                        # "Commit" the current alignment
                        if align_tok in (AlignLabels.correct, AlignLabels.substitution):
                            alignment.append(align_tok)
                            
                            # Check for syllable conflicts
                            if not break_on_syllables or not ref_extra_syllable_word_index:
                                ref_aligned.append(' '.join(ref_word_builder))
                                ref_syllable_count = 0
                                hyp_syllable_count = 0
                            else:
                                ref_aligned.append(' '.join(ref_word_builder[0:ref_extra_syllable_word_index]))
                                # The remaining words are deletions
                                for word in ref_word_builder[ref_extra_syllable_word_index:]:
                                    alignment.append(AlignLabels.deletion)
                                    ref_aligned.append(word)
                                    hyp_aligned.append('')
                                ref_syllable_count = 0
                                
                            if not break_on_syllables or not hyp_extra_syllable_word_index:
                                hyp_aligned.append(' '.join(hyp_word_builder))
                                ref_syllable_count = 0
                                hyp_syllable_count = 0
                            else:
                                hyp_aligned.append(' '.join(hyp_word_builder[0:hyp_extra_syllable_word_index]))
                                # The remaining words are insertions
                                for word in hyp_word_builder[hyp_extra_syllable_word_index:]:
                                    alignment.append(AlignLabels.insertion)
                                    ref_aligned.append('')
                                    hyp_aligned.append(word)
                                    hyp_syllable_count = 0
                            
                            if align_tok == AlignLabels.substitution:
                                # Check if you need to rework this alignment.
                                if len(ref_word_builder) != len(hyp_word_builder):
                                    # Word count mismatch in the alignment span. Is there a possibility that we need to re-align this segment?
                                    ref_word_span_curr = (ref_word_index, ref_word_index + len(ref_word_builder))
                                    hyp_word_span_curr = (hyp_word_index, hyp_word_index + len(hyp_word_builder))
                                    phone_align_curr = phone_align.subsequence(0, i+1, preserve_index=False)
                                    
                                    lev = Levenshtein.align(
                                        ref=phone_align_curr.s1_tokens(), 
                                        hyp=phone_align_curr.s2_tokens(),
                                        reserve_list=PowerAligner.reserve_list, 
                                        exclusive_sets=PowerAligner.exclusive_sets,
                                        weights=Levenshtein.wordAlignWeights) #, 
                                        #dist_penalty=PowerAligner.phoneDistPenalty, dist_penalty_set=Levenshtein.wordAlignWeights)
                                        
                                    phone_align_adjusted = lev.expandAlignCompact()
                                    
                                    if phone_align_curr.align != phone_align_adjusted.align:
                                        # Looks like we need to redo the phone-to-word alignment.
                                        worklist.append((ref_word_span_curr, hyp_word_span_curr, phone_align_adjusted))
                                    else:
                                        commit_alignment = True
                                else:
                                    commit_alignment = True
                                        
                        elif align_tok == AlignLabels.deletion:
                            for word in ref_word_builder:
                                alignment.append(align_tok)
                                ref_aligned.append(word)
                                hyp_aligned.append('')
                                
                            commit_alignment = True
                            ref_syllable_count = 0
                            
                        elif align_tok == AlignLabels.insertion:
                            for word in hyp_word_builder:
                                alignment.append(align_tok)
                                ref_aligned.append('')
                                hyp_aligned.append(word)
                                        
                            commit_alignment = True
                            hyp_syllable_count = 0
                            
                        if commit_alignment:
                            # Commit the alignment.
                            full_reference.extend(ref_aligned)
                            full_hypothesis.extend(hyp_aligned)
                            full_alignment.extend(alignment)
                            full_phone_align.append(phone_align.subsequence(0, i, preserve_index=False))
                            ref_aligned = []
                            hyp_aligned = []
                            alignment = []
                        break
                
                # Add words if word boundaries are reached.
                else:
                    if ref_type == TokType.WordBoundary:
                        ref_word_started = False
                        if hyp_type != TokType.WordBoundary and ref_word_builder and not hyp_word_builder:
                            # DELETION
                            # Ref word ended, but no hyp words have been added. Mark the current ref word(s) in the span as deletion errors.
                            # TODO: Dedupe this logic
                            for word in ref_word_builder:
                                alignment.append(AlignLabels.deletion)
                                ref_aligned.append(word)
                                hyp_aligned.append('')
                            ref_syllable_count = 0
                            
                            # Commit the alignment.
                            full_reference.extend(ref_aligned)
                            full_hypothesis.extend(hyp_aligned)
                            full_alignment.extend(alignment)
                            full_phone_align.append(phone_align.subsequence(0, i, preserve_index=False))
                            
                            # Add the remainder to the worklist
                            ref_word_span_next = (ref_word_index + len(ref_word_builder), ref_word_limit)
                            hyp_word_span_next = (hyp_word_index + len(hyp_word_builder), hyp_word_limit)
                            lev = Levenshtein.align(
                                ref=[x for x in phone_align.s1[i:] if x],
                                hyp=[x for x in phone_align.s2 if x],
                                reserve_list=PowerAligner.reserve_list, 
                                exclusive_sets=PowerAligner.exclusive_sets,
                                weights=Levenshtein.wordAlignWeights) #, 
                                #dist_penalty=PowerAligner.phoneDistPenalty, dist_penalty_set=Levenshtein.wordAlignWeights)
                            phone_align_next = lev.expandAlignCompact()
                            
                            worklist.append((ref_word_span_next, hyp_word_span_next, phone_align_next))
                            break					
                    elif ref_type == TokType.Phoneme and not ref_word_started:
                        ref_word_started = True
                        try:
                            ref_word_item = ref_word_iter.__next__()
                            ref_word_builder.append(ref_word_item[1])
                        except StopIteration:
                            pass
                    
                    if hyp_type == TokType.WordBoundary:
                        hyp_word_started = False
                        if ref_type != TokType.WordBoundary and hyp_word_builder and not ref_word_builder:
                            # INSERTION
                            # Hyp word ended, but no ref words have been added. Mark the current hyp word(s) in the span as insertion errors.
                            # TODO: Dedupe this logic
                            for word in hyp_word_builder:
                                alignment.append(AlignLabels.insertion)
                                ref_aligned.append('')
                                hyp_aligned.append(word)
                            hyp_syllable_count = 0
                            
                            # Commit the alignment.
                            full_reference.extend(ref_aligned)
                            full_hypothesis.extend(hyp_aligned)
                            full_alignment.extend(alignment)
                            full_phone_align.append(phone_align.subsequence(0, i, preserve_index=False))
                            
                            # Add the remainder to the worklist
                            ref_word_span_next = (ref_word_index + len(ref_word_builder), ref_word_limit)
                            hyp_word_span_next = (hyp_word_index + len(hyp_word_builder), hyp_word_limit)
                            lev = Levenshtein.align(
                                ref=[x for x in phone_align.s1 if x],
                                hyp=[x for x in phone_align.s2[i:] if x],
                                reserve_list=PowerAligner.reserve_list, 
                                exclusive_sets=PowerAligner.exclusive_sets,
                                weights=Levenshtein.wordAlignWeights) #, 
                                #dist_penalty=PowerAligner.phoneDistPenalty, dist_penalty_set=Levenshtein.wordAlignWeights)
                            phone_align_next = lev.expandAlignCompact()
                            
                            worklist.append((ref_word_span_next, hyp_word_span_next, phone_align_next))
                            break
                    elif hyp_type == TokType.Phoneme and not hyp_word_started:
                        hyp_word_started = True
                        try:
                            hyp_word_item = hyp_word_iter.__next__()
                            hyp_word_builder.append(hyp_word_item[1])
                        except StopIteration: 
                            pass
                        
                # Check for syllable mismatches
                if ref_type == TokType.SyllableBoundary:
                    ref_syllable_count += 1
                if hyp_type == TokType.SyllableBoundary:
                    hyp_syllable_count += 1
                    
                if (ref_type == TokType.SyllableBoundary == hyp_type or ref_syllable_count == hyp_syllable_count):
                    # No syllable conflicts here!
                    ref_extra_syllable_word_index = None
                    hyp_extra_syllable_word_index = None
                elif (ref_type == TokType.SyllableBoundary and 
                            not ref_extra_syllable_word_index and 
                            TokType.checkAnnotation(phone_align.s2[i - 1]) == TokType.WordBoundary):
                    # Extra syllable in hypothesis. We only care if the syllable immediately follows a word boundary. 
                    # This is because this indicates that a new word is being formed, which may likely be an insertion in hyp.
                    ref_extra_syllable_word_index = len(ref_word_builder) - 1
                    # print ref_word_builder
                    # print 'Syllable/word mismatch at', i
                    # print 'Extra hyp word:', ref_word_builder[ref_extra_syllable_word_index]
                elif (hyp_type == TokType.SyllableBoundary and 
                            not hyp_extra_syllable_word_index and 
                            TokType.checkAnnotation(phone_align.s2[i - 1]) == TokType.WordBoundary):
                    # This time there's an extra syllable in the ref, corresponding to a new ref word.
                    hyp_extra_syllable_word_index = len(hyp_word_builder) - 1
                    # print hyp_word_builder
                    # print 'Syllable/word mismatch at', i
                    # print 'Extra ref word:', hyp_word_builder[hyp_extra_syllable_word_index]
        # Concatenate all phoneme alignments
        fp_align = full_phone_align[0]
        for expand_align in full_phone_align[1:]:
            fp_align.append_alignment(expand_align)
            
        word_align = ExpandedAlignment(full_reference, full_hypothesis, full_alignment)
        word_align = cls._remove_many_to_many(
            word_align, alignment_mode, max_many_to_one, max_one_to_many,
            max_multiword_char_ratio)
        return word_align, fp_align

    @classmethod
    def _remove_many_to_many(cls, word_align, alignment_mode, max_many_to_one=None,
                             max_one_to_many=None, max_multiword_char_ratio=None):
        ref_aligned = []
        hyp_aligned = []
        alignment = []

        for ref_token, hyp_token, align_token in zip(word_align.s1, word_align.s2, word_align.align):
            ref_words = [x for x in ref_token.split() if x]
            hyp_words = [x for x in hyp_token.split() if x]

            if align_token != AlignLabels.substitution:
                ref_aligned.append(ref_token)
                hyp_aligned.append(hyp_token)
                alignment.append(align_token)
                continue

            is_many_to_one = len(ref_words) > 1 and len(hyp_words) >= 1
            is_one_to_many = len(hyp_words) > 1 and len(ref_words) >= 1
            exceeds_many_to_one = (
                max_many_to_one is not None and len(hyp_words) > 0 and
                len(ref_words) > len(hyp_words) * max_many_to_one)
            exceeds_one_to_many = (
                max_one_to_many is not None and len(ref_words) > 0 and
                len(hyp_words) > len(ref_words) * max_one_to_many)

            if not (is_many_to_one or is_one_to_many or exceeds_many_to_one or exceeds_one_to_many):
                ref_aligned.append(ref_token)
                hyp_aligned.append(hyp_token)
                alignment.append(align_token)
                continue

            if (alignment_mode == cls.AlignmentMode.ManyToOne or
                    (alignment_mode == cls.AlignmentMode.Auto and len(ref_words) >= len(hyp_words)) or
                    exceeds_many_to_one):
                cls._append_many_to_one(ref_words, hyp_words, ref_aligned, hyp_aligned, alignment,
                                        max_many_to_one=max_many_to_one,
                                        max_multiword_char_ratio=max_multiword_char_ratio)
            else:
                cls._append_one_to_many(ref_words, hyp_words, ref_aligned, hyp_aligned, alignment,
                                        max_one_to_many=max_one_to_many,
                                        max_multiword_char_ratio=max_multiword_char_ratio)

        return ExpandedAlignment(ref_aligned, hyp_aligned, alignment)

    @staticmethod
    def _split_evenly(items, group_count):
        if group_count <= 0:
            return []
        base = len(items) // group_count
        extra = len(items) % group_count
        groups = []
        start = 0
        for i in range(group_count):
            group_len = base + (1 if i < extra else 0)
            groups.append(items[start:start + group_len])
            start += group_len
        return groups

    @staticmethod
    def _char_ratio_ok(ref_words, hyp_words, max_multiword_char_ratio):
        if max_multiword_char_ratio is None:
            return True
        ref_len = len(''.join(ref_words))
        hyp_len = len(''.join(hyp_words))
        if ref_len == 0 or hyp_len == 0:
            return True
        return max(ref_len, hyp_len) / min(ref_len, hyp_len) <= max_multiword_char_ratio

    @staticmethod
    def _compact_words(words):
        return ''.join(words).lower()

    @staticmethod
    def _append_deletions(words, ref_aligned, hyp_aligned, alignment):
        for word in words:
            ref_aligned.append(word)
            hyp_aligned.append('')
            alignment.append(AlignLabels.deletion)

    @staticmethod
    def _append_insertions(words, ref_aligned, hyp_aligned, alignment):
        for word in words:
            ref_aligned.append('')
            hyp_aligned.append(word)
            alignment.append(AlignLabels.insertion)

    @classmethod
    def _phone_edit_distance(cls, ref_words, hyp_words, pronouncer):
        ref_phones = pronouncer.pronounce(ref_words)
        hyp_phones = pronouncer.pronounce(hyp_words)
        lev = Levenshtein.align(
            ref=ref_phones,
            hyp=hyp_phones,
            reserve_list=cls.reserve_list,
            exclusive_sets=cls.exclusive_sets,
            weights=Levenshtein.wordAlignWeights)
        return lev.dist

    @classmethod
    def _alignment_phone_score(cls, ref_slots, hyp_slots, labels, pronouncer,
                               multiword_slot_penalty=repairMultiwordSlotPenalty):
        score = 0.0
        for ref_slot, hyp_slot, label in zip(ref_slots, hyp_slots, labels):
            ref_words = [x for x in ref_slot.split() if x]
            hyp_words = [x for x in hyp_slot.split() if x]
            if label == AlignLabels.correct:
                continue
            if label == AlignLabels.substitution:
                score += cls._phone_edit_distance(ref_words, hyp_words, pronouncer)
                score += max(0, max(len(ref_words), len(hyp_words)) - 1) * multiword_slot_penalty
            elif label == AlignLabels.deletion:
                score += len(pronouncer.pronounce(ref_words)) * Levenshtein.wordAlignWeights[AlignLabels.deletion]
            elif label == AlignLabels.insertion:
                score += len(pronouncer.pronounce(hyp_words)) * Levenshtein.wordAlignWeights[AlignLabels.insertion]
        return score

    @classmethod
    def _candidate_is_better(cls, old_ref, old_hyp, old_labels,
                             new_ref, new_hyp, new_labels, pronouncer,
                             multiword_slot_penalty=repairMultiwordSlotPenalty):
        if pronouncer is None:
            return True
        try:
            old_score = cls._alignment_phone_score(
                old_ref, old_hyp, old_labels, pronouncer, multiword_slot_penalty)
            new_score = cls._alignment_phone_score(
                new_ref, new_hyp, new_labels, pronouncer, multiword_slot_penalty)
        except KeyError:
            return False
        return new_score < old_score

    @staticmethod
    def _candidate_signature(ref_slots, hyp_slots, labels):
        return tuple(zip(ref_slots, hyp_slots, labels))

    @staticmethod
    def _flat_slot_words(slots):
        words = []
        for slot in slots:
            words.extend([x for x in slot.split() if x])
        return words

    @classmethod
    def _add_candidate(cls, candidates, seen, ref_slots, hyp_slots, labels,
                       ref_words=None, hyp_words=None):
        if ref_words is not None and cls._flat_slot_words(ref_slots) != ref_words:
            return
        if hyp_words is not None and cls._flat_slot_words(hyp_slots) != hyp_words:
            return
        signature = cls._candidate_signature(ref_slots, hyp_slots, labels)
        if signature not in seen:
            seen.add(signature)
            candidates.append((ref_slots, hyp_slots, labels))

    @classmethod
    def _append_word_sequence(cls, words, ref_side, ref_slots, hyp_slots, labels):
        if ref_side:
            cls._append_deletions(words, ref_slots, hyp_slots, labels)
        else:
            cls._append_insertions(words, ref_slots, hyp_slots, labels)

    @classmethod
    def _compound_with_extra_tail_candidate(cls, ref_words, hyp_words, max_many_to_one=None,
                                           max_one_to_many=None,
                                           max_multiword_char_ratio=None):
        if len(ref_words) > 1 and len(hyp_words) > 1:
            max_hyp_words = max_one_to_many or len(hyp_words)
            if len(hyp_words) <= max_hyp_words:
                for ref_index in (0, len(ref_words) - 1):
                    ref_word = ref_words[ref_index]
                    if (cls._compact_words([ref_word]) == cls._compact_words(hyp_words) and
                            cls._char_ratio_ok([ref_word], hyp_words, max_multiword_char_ratio)):
                        ref_slots = []
                        hyp_slots = []
                        labels = []
                        cls._append_deletions(ref_words[:ref_index], ref_slots, hyp_slots, labels)
                        ref_slots.append(ref_word)
                        hyp_slots.append(' '.join(hyp_words))
                        labels.append(AlignLabels.substitution)
                        cls._append_deletions(ref_words[ref_index + 1:], ref_slots, hyp_slots, labels)
                        return ref_slots, hyp_slots, labels

            max_ref_words = max_many_to_one or len(ref_words)
            if len(ref_words) <= max_ref_words:
                for hyp_index in (0, len(hyp_words) - 1):
                    hyp_word = hyp_words[hyp_index]
                    if (cls._compact_words(ref_words) == cls._compact_words([hyp_word]) and
                            cls._char_ratio_ok(ref_words, [hyp_word], max_multiword_char_ratio)):
                        ref_slots = []
                        hyp_slots = []
                        labels = []
                        cls._append_insertions(hyp_words[:hyp_index], ref_slots, hyp_slots, labels)
                        ref_slots.append(' '.join(ref_words))
                        hyp_slots.append(hyp_word)
                        labels.append(AlignLabels.substitution)
                        cls._append_insertions(hyp_words[hyp_index + 1:], ref_slots, hyp_slots, labels)
                        return ref_slots, hyp_slots, labels

        return None

    @classmethod
    def _multiword_repair_candidates(cls, ref_words, hyp_words, max_many_to_one=None,
                                     max_one_to_many=None,
                                     max_multiword_char_ratio=None):
        candidates = []
        seen = set()

        max_hyp_words = max_one_to_many or len(hyp_words)
        max_ref_words = max_many_to_one or len(ref_words)

        if ref_words and len(hyp_words) > 1:
            for ref_index, ref_word in enumerate(ref_words):
                for hyp_start in range(len(hyp_words)):
                    for hyp_end in range(hyp_start + 2, len(hyp_words) + 1):
                        hyp_span = hyp_words[hyp_start:hyp_end]
                        if len(hyp_span) > max_hyp_words:
                            continue
                        if not cls._char_ratio_ok([ref_word], hyp_span, max_multiword_char_ratio):
                            continue
                        ref_slots = []
                        hyp_slots = []
                        labels = []
                        cls._append_deletions(ref_words[:ref_index], ref_slots, hyp_slots, labels)
                        cls._append_insertions(hyp_words[:hyp_start], ref_slots, hyp_slots, labels)
                        ref_slots.append(ref_word)
                        hyp_slots.append(' '.join(hyp_span))
                        labels.append(AlignLabels.substitution)
                        cls._append_insertions(hyp_words[hyp_end:], ref_slots, hyp_slots, labels)
                        cls._append_deletions(ref_words[ref_index + 1:], ref_slots, hyp_slots, labels)
                        cls._add_candidate(
                            candidates, seen, ref_slots, hyp_slots, labels,
                            ref_words=ref_words, hyp_words=hyp_words)

        if len(ref_words) > 1 and hyp_words:
            for hyp_index, hyp_word in enumerate(hyp_words):
                for ref_start in range(len(ref_words)):
                    for ref_end in range(ref_start + 2, len(ref_words) + 1):
                        ref_span = ref_words[ref_start:ref_end]
                        if len(ref_span) > max_ref_words:
                            continue
                        if not cls._char_ratio_ok(ref_span, [hyp_word], max_multiword_char_ratio):
                            continue
                        ref_slots = []
                        hyp_slots = []
                        labels = []
                        cls._append_insertions(hyp_words[:hyp_index], ref_slots, hyp_slots, labels)
                        cls._append_deletions(ref_words[:ref_start], ref_slots, hyp_slots, labels)
                        ref_slots.append(' '.join(ref_span))
                        hyp_slots.append(hyp_word)
                        labels.append(AlignLabels.substitution)
                        cls._append_deletions(ref_words[ref_end:], ref_slots, hyp_slots, labels)
                        cls._append_insertions(hyp_words[hyp_index + 1:], ref_slots, hyp_slots, labels)
                        cls._add_candidate(
                            candidates, seen, ref_slots, hyp_slots, labels,
                            ref_words=ref_words, hyp_words=hyp_words)

        compound_candidate = cls._compound_with_extra_tail_candidate(
            ref_words, hyp_words,
            max_many_to_one=max_many_to_one,
            max_one_to_many=max_one_to_many,
            max_multiword_char_ratio=max_multiword_char_ratio)
        if compound_candidate is not None:
            cls._add_candidate(
                candidates, seen, *compound_candidate,
                ref_words=ref_words, hyp_words=hyp_words)

        return candidates

    @classmethod
    def _best_phone_scored_candidate(cls, candidates, old_ref, old_hyp,
                                     old_labels, pronouncer,
                                     multiword_slot_penalty=repairMultiwordSlotPenalty):
        if not candidates:
            return None
        if pronouncer is None:
            return candidates[0]

        try:
            best_score = cls._alignment_phone_score(
                old_ref, old_hyp, old_labels, pronouncer, multiword_slot_penalty)
        except KeyError:
            return None

        best_candidate = None
        for candidate in candidates:
            ref_slots, hyp_slots, labels = candidate
            try:
                score = cls._alignment_phone_score(
                    ref_slots, hyp_slots, labels, pronouncer, multiword_slot_penalty)
            except KeyError:
                continue
            if score < best_score:
                best_score = score
                best_candidate = candidate
        return best_candidate

    @classmethod
    def _slot_phone_score(cls, ref_words, hyp_words, label, pronouncer,
                          multiword_slot_penalty=repairMultiwordSlotPenalty):
        if label == AlignLabels.correct:
            return 0.0
        if label == AlignLabels.substitution:
            score = cls._phone_edit_distance(ref_words, hyp_words, pronouncer)
            score += max(0, max(len(ref_words), len(hyp_words)) - 1) * multiword_slot_penalty
            return score
        if label == AlignLabels.deletion:
            return len(pronouncer.pronounce(ref_words)) * Levenshtein.wordAlignWeights[AlignLabels.deletion]
        if label == AlignLabels.insertion:
            return len(pronouncer.pronounce(hyp_words)) * Levenshtein.wordAlignWeights[AlignLabels.insertion]
        return 0.0

    @classmethod
    def _dp_multiword_repair_candidate(cls, ref_words, hyp_words, old_ref,
                                       old_hyp, old_labels, pronouncer,
                                       max_many_to_one=None, max_one_to_many=None,
                                       max_multiword_char_ratio=None,
                                       multiword_slot_penalty=repairMultiwordSlotPenalty):
        if pronouncer is None:
            return None

        n = len(ref_words)
        m = len(hyp_words)
        max_ref_words = max_many_to_one or n
        max_hyp_words = max_one_to_many or m
        try:
            old_score = cls._alignment_phone_score(
                old_ref, old_hyp, old_labels, pronouncer, multiword_slot_penalty)
        except KeyError:
            return None
        dp = {(0, 0): (0.0, [], [], [])}

        for i in range(n + 1):
            for j in range(m + 1):
                if (i, j) not in dp:
                    continue
                base_score, base_ref, base_hyp, base_labels = dp[(i, j)]
                transitions = []

                if i < n:
                    transitions.append(([ref_words[i]], [], AlignLabels.deletion, 1, 0))
                if j < m:
                    transitions.append(([], [hyp_words[j]], AlignLabels.insertion, 0, 1))
                if i < n and j < m:
                    label = AlignLabels.correct if ref_words[i] == hyp_words[j] else AlignLabels.substitution
                    transitions.append(([ref_words[i]], [hyp_words[j]], label, 1, 1))

                if i < n:
                    max_k = min(max_hyp_words, m - j)
                    for k in range(2, max_k + 1):
                        ref_span = [ref_words[i]]
                        hyp_span = hyp_words[j:j + k]
                        if cls._char_ratio_ok(ref_span, hyp_span, max_multiword_char_ratio):
                            transitions.append((ref_span, hyp_span, AlignLabels.substitution, 1, k))

                if j < m:
                    max_k = min(max_ref_words, n - i)
                    for k in range(2, max_k + 1):
                        ref_span = ref_words[i:i + k]
                        hyp_span = [hyp_words[j]]
                        if cls._char_ratio_ok(ref_span, hyp_span, max_multiword_char_ratio):
                            transitions.append((ref_span, hyp_span, AlignLabels.substitution, k, 1))

                for ref_span, hyp_span, label, di, dj in transitions:
                    try:
                        slot_score = cls._slot_phone_score(
                            ref_span, hyp_span, label, pronouncer, multiword_slot_penalty)
                    except KeyError:
                        continue
                    ref_text = ' '.join(ref_span)
                    hyp_text = ' '.join(hyp_span)
                    next_key = (i + di, j + dj)
                    next_score = base_score + slot_score
                    next_ref = base_ref + [ref_text]
                    next_hyp = base_hyp + [hyp_text]
                    next_labels = base_labels + [label]
                    current = dp.get(next_key)
                    if current is None or next_score < current[0]:
                        dp[next_key] = (next_score, next_ref, next_hyp, next_labels)

        result = dp.get((n, m))
        if result is None:
            return None
        score, ref_slots, hyp_slots, labels = result
        if cls._flat_slot_words(ref_slots) != ref_words or cls._flat_slot_words(hyp_slots) != hyp_words:
            return None
        has_multiword_slot = any(
            label == AlignLabels.substitution and
            (len(ref_slot.split()) > 1 or len(hyp_slot.split()) > 1)
            for ref_slot, hyp_slot, label in zip(ref_slots, hyp_slots, labels)
        )
        if not has_multiword_slot:
            return None
        if score < old_score:
            return ref_slots, hyp_slots, labels
        return None

    @classmethod
    def _merge_adjacent_multiword_errors(cls, word_align, max_many_to_one=None,
                                         max_one_to_many=None,
                                         max_multiword_char_ratio=None,
                                         pronouncer=None,
                                         multiword_slot_penalty=repairMultiwordSlotPenalty):
        ref_aligned = []
        hyp_aligned = []
        alignment = []
        i = 0

        while i < word_align.length():
            if word_align.align[i] not in (AlignLabels.substitution,
                                           AlignLabels.insertion,
                                           AlignLabels.deletion):
                ref_aligned.append(word_align.s1[i])
                hyp_aligned.append(word_align.s2[i])
                alignment.append(word_align.align[i])
                i += 1
                continue

            j = i
            ref_words = []
            hyp_words = []
            labels = []
            while j < word_align.length() and word_align.align[j] in (
                    AlignLabels.substitution, AlignLabels.insertion,
                    AlignLabels.deletion):
                labels.append(word_align.align[j])
                ref_words.extend([x for x in word_align.s1[j].split() if x])
                hyp_words.extend([x for x in word_align.s2[j].split() if x])
                j += 1

            old_ref = word_align.s1[i:j]
            old_hyp = word_align.s2[i:j]
            old_labels = word_align.align[i:j]

            dp_candidate = cls._dp_multiword_repair_candidate(
                ref_words, hyp_words, old_ref, old_hyp, old_labels, pronouncer,
                max_many_to_one=max_many_to_one,
                max_one_to_many=max_one_to_many,
                max_multiword_char_ratio=max_multiword_char_ratio,
                multiword_slot_penalty=multiword_slot_penalty)
            if dp_candidate is not None:
                cand_ref, cand_hyp, cand_labels = dp_candidate
                ref_aligned.extend(cand_ref)
                hyp_aligned.extend(cand_hyp)
                alignment.extend(cand_labels)
                i = j
                continue

            candidates = cls._multiword_repair_candidates(
                ref_words, hyp_words,
                max_many_to_one=max_many_to_one,
                max_one_to_many=max_one_to_many,
                max_multiword_char_ratio=max_multiword_char_ratio)
            best_candidate = cls._best_phone_scored_candidate(
                candidates, old_ref, old_hyp, old_labels, pronouncer,
                multiword_slot_penalty=multiword_slot_penalty)
            if best_candidate is not None:
                cand_ref, cand_hyp, cand_labels = best_candidate
                ref_aligned.extend(cand_ref)
                hyp_aligned.extend(cand_hyp)
                alignment.extend(cand_labels)
                i = j
                continue

            ref_aligned.extend(old_ref)
            hyp_aligned.extend(old_hyp)
            alignment.extend(old_labels)
            i = j

        return ExpandedAlignment(ref_aligned, hyp_aligned, alignment)

    @classmethod
    def _append_many_to_one(cls, ref_words, hyp_words, ref_aligned, hyp_aligned, alignment,
                            max_many_to_one=None, max_multiword_char_ratio=None):
        if len(ref_words) >= len(hyp_words):
            max_words = max_many_to_one or len(ref_words)
            mapped_ref_count = min(len(ref_words), len(hyp_words) * max_words)
            ref_groups = cls._split_evenly(ref_words[:mapped_ref_count], len(hyp_words))
            for ref_group, hyp_word in zip(ref_groups, hyp_words):
                if cls._char_ratio_ok(ref_group, [hyp_word], max_multiword_char_ratio):
                    ref_aligned.append(' '.join(ref_group))
                    hyp_aligned.append(hyp_word)
                    alignment.append(AlignLabels.substitution if ' '.join(ref_group) != hyp_word else AlignLabels.correct)
                else:
                    ref_aligned.append(ref_group[0])
                    hyp_aligned.append(hyp_word)
                    alignment.append(AlignLabels.substitution if ref_group[0] != hyp_word else AlignLabels.correct)
                    for ref_word in ref_group[1:]:
                        ref_aligned.append(ref_word)
                        hyp_aligned.append('')
                        alignment.append(AlignLabels.deletion)
            for ref_word in ref_words[mapped_ref_count:]:
                ref_aligned.append(ref_word)
                hyp_aligned.append('')
                alignment.append(AlignLabels.deletion)
            return

        ref_aligned.append(' '.join(ref_words))
        hyp_aligned.append(hyp_words[0])
        alignment.append(AlignLabels.substitution)
        for hyp_word in hyp_words[1:]:
            ref_aligned.append('')
            hyp_aligned.append(hyp_word)
            alignment.append(AlignLabels.insertion)

    @classmethod
    def _append_one_to_many(cls, ref_words, hyp_words, ref_aligned, hyp_aligned, alignment,
                            max_one_to_many=None, max_multiword_char_ratio=None):
        if len(hyp_words) >= len(ref_words):
            max_words = max_one_to_many or len(hyp_words)
            mapped_hyp_count = min(len(hyp_words), len(ref_words) * max_words)
            hyp_groups = cls._split_evenly(hyp_words[:mapped_hyp_count], len(ref_words))
            for ref_word, hyp_group in zip(ref_words, hyp_groups):
                hyp_group_text = ' '.join(hyp_group)
                if cls._char_ratio_ok([ref_word], hyp_group, max_multiword_char_ratio):
                    ref_aligned.append(ref_word)
                    hyp_aligned.append(hyp_group_text)
                    alignment.append(AlignLabels.substitution if ref_word != hyp_group_text else AlignLabels.correct)
                else:
                    ref_aligned.append(ref_word)
                    hyp_aligned.append(hyp_group[0])
                    alignment.append(AlignLabels.substitution if ref_word != hyp_group[0] else AlignLabels.correct)
                    for hyp_word in hyp_group[1:]:
                        ref_aligned.append('')
                        hyp_aligned.append(hyp_word)
                        alignment.append(AlignLabels.insertion)
            for hyp_word in hyp_words[mapped_hyp_count:]:
                ref_aligned.append('')
                hyp_aligned.append(hyp_word)
                alignment.append(AlignLabels.insertion)
            return

        ref_aligned.append(ref_words[0])
        hyp_aligned.append(' '.join(hyp_words))
        alignment.append(AlignLabels.substitution)
        for ref_word in ref_words[1:]:
            ref_aligned.append(ref_word)
            hyp_aligned.append('')
            alignment.append(AlignLabels.deletion)
            
