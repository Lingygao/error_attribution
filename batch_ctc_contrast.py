"""Sentence/segment and task3 substitution contrast on shared CTC emissions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from ctc_contrast_core import align,encode_words


def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--model',default='GroNLP/wav2vec2-dutch-large-ft-cgn');p.add_argument('--device',default='cuda:0');p.add_argument('--cache-dir',default=None);p.add_argument('--local-files-only',action='store_true')
 p.add_argument('--variants',nargs='+',default=['v2','v3']);p.add_argument('--speakers',nargs='*');p.add_argument('--limit',type=int);p.add_argument('--pad-seconds',type=float,default=.08);p.add_argument('--overwrite',action='store_true');a=p.parse_args()
 import torch
 from transformers import AutoProcessor,AutoModelForCTC
 import soundfile as sf
 from scipy.signal import resample_poly
 from math import gcd
 if a.pad_seconds<0:raise ValueError('negative padding')
 processor=AutoProcessor.from_pretrained(a.model,cache_dir=a.cache_dir,local_files_only=a.local_files_only)
 model=AutoModelForCTC.from_pretrained(a.model,cache_dir=a.cache_dir,local_files_only=a.local_files_only).to(a.device).eval()
 blank=model.config.pad_token_id
 records=[json.loads(s) for s in a.manifest.read_text().splitlines() if s.strip()]
 records=[r for r in records if r['variant'] in a.variants and (not a.speakers or r['speaker'] in a.speakers)]
 if a.limit is not None:records=records[:a.limit]
 if not records:raise ValueError('no_matching_manifest_records')
 a.output.mkdir(parents=True,exist_ok=True)
 provenance=dict(model=a.model,model_commit=getattr(model.config,'_commit_hash',None),manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),pad_seconds=a.pad_seconds,score='Mean over character positions of mean exp(logp) on Viterbi token frames; blanks and delimiters excluded',sign='prompt minus ASR',sentence_unit='manifest segment, not necessarily linguistic sentence',word_window='ASR forced alignment span plus padding, identical emission frame slice for both candidates',attention_used=False)
 config_path=a.output/'config.json'
 if config_path.exists() and json.loads(config_path.read_text())!=provenance and not a.overwrite:raise ValueError('output configuration mismatch; choose new output directory')
 config_path.write_text(json.dumps(provenance,indent=2)+'\n')
 cache_path=None;audio=None;allrows=[];segments=[]
 for r in records:
  key=f"{r['variant']}__{r['speaker']}__{r['segment_id']}";dest=a.output/(key+'.json')
  if dest.exists() and not a.overwrite and json.loads(dest.read_text()).get('status')=='ok':
   result=json.loads(dest.read_text())
  else:
   result={k:r[k] for k in ['speaker','variant','segment_id','start','end']};result['words']=[]
   try:
    if cache_path!=r['audio_path']:
     audio,sr=sf.read(r['audio_path'],dtype='float32',always_2d=True);audio=audio.mean(axis=1)
     if sr!=16000:
      divisor=gcd(sr,16000);audio=resample_poly(audio,16000//divisor,sr//divisor).astype('float32')
     cache_path=r['audio_path']
    lo=round(r['start']*16000);hi=round(r['end']*16000)
    if lo<0 or hi>len(audio) or hi<=lo:raise ValueError('invalid_audio_bounds')
    inp=processor(audio[lo:hi],sampling_rate=16000,return_tensors='pt')
    with torch.inference_mode():lp=model(**{k:v.to(a.device) for k,v in inp.items()}).logits[0].float().log_softmax(-1).cpu().numpy()
    def score(words,emissions):
     ids,owners=encode_words(words,processor.tokenizer);return align(emissions,ids,owners,blank)
    aa=score(r['asr_words'],lp);pp=score(r['prompt_words'],lp)
    result.update(status='ok',ctc_prompt=pp['score'],ctc_asr=aa['score'],ctc_contrast_sentence=pp['score']-aa['score'],prompt_viterbi_log_per_frame=pp['viterbi_log_score_per_frame'],asr_viterbi_log_per_frame=aa['viterbi_log_score_per_frame'])
    byindex={j:dict(speaker=r['speaker'],variant=r['variant'],segment_id=r['segment_id'],asr_index=j,asr_word=w,ctc_contrast_sentence=result['ctc_contrast_sentence'],ctc_contrast_word=None,word_status='not_substitution') for j,w in zip(r['asr_indices'],r['asr_words'])}
    pad=round(a.pad_seconds*len(lp)/(r['end']-r['start']))
    for g in r['substitutions']:
     details=dict(prompt_words=' '.join(g['prompt_words']),asr_group_words=' '.join(g['asr_words']),mapping=g['mapping'])
     try:
      spans=[aa['word_frames'][i] for i in g['local_indices']];start=max(0,min(x[0] for x in spans)-pad);end=min(len(lp),max(x[1] for x in spans)+pad)
      wp=score(g['prompt_words'],lp[start:end]);wa=score(g['asr_words'],lp[start:end])
      details.update(word_status='ok',ctc_prompt_word=wp['score'],ctc_asr_word=wa['score'],ctc_contrast_word=wp['score']-wa['score'],window_start=r['start']+start/len(lp)*(r['end']-r['start']),window_end=r['start']+end/len(lp)*(r['end']-r['start']))
     except (ValueError,KeyError) as e:details.update(word_status='failed',word_error=str(e))
     for j in g['asr_indices']:byindex[j].update(details)
    result['words']=list(byindex.values())
   except Exception as e:
    result.update(status='failed',error=f'{type(e).__name__}: {e}')
    result['words']=[dict(speaker=r['speaker'],variant=r['variant'],segment_id=r['segment_id'],asr_index=j,asr_word=w,ctc_contrast_sentence=None,ctc_contrast_word=None,word_status='segment_failed') for j,w in zip(r['asr_indices'],r['asr_words'])]
   tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(dest)
  print(key,result['status'],flush=True);allrows.extend(result['words']);segments.append({k:v for k,v in result.items() if k!='words'})
 for name,rows in [('word_contrast.csv',allrows),('sentence_contrast.csv',segments)]:
  cols=list(dict.fromkeys(k for row in rows for k in row))
  with (a.output/name).open('w') as out:
   writer=csv.DictWriter(out,fieldnames=cols);writer.writeheader();writer.writerows(rows)
 failures=sum(r['status']!='ok' for r in segments);print(json.dumps(dict(segments=len(segments),failed_segments=failures,word_rows=len(allrows))))
 if failures:raise SystemExit(2)

if __name__=='__main__':main()
