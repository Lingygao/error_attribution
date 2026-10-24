"""Prepare paired segment manifest using original PR-ASR alignment; no MT/labels."""
import argparse
import json
import sys
from pathlib import Path


def main():
 p=argparse.ArgumentParser();p.add_argument('--project-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--variants',nargs='+',default=['v2','v3']);args=p.parse_args()
 root=args.project_root.resolve()
 import num2words  # Required: do not silently leave Dutch numbers unnormalized.
 from text_normalization import text_norm_all
 from merge_speaker_signal_tables import NormalizedToken,collapse_repeated_tokens
 args.output.parent.mkdir(parents=True,exist_ok=True);audit=[];written=0
 speakers=(root/'notebooks/train.txt').read_text().split()+(root/'notebooks/test.txt').read_text().split()
 assert len(speakers)==len(set(speakers))
 with args.output.open('w') as out:
  for variant in args.variants:
   for sp in speakers:
    try:
     src=root/f'data/jasmin/{variant}_segment/{sp}/{sp}_newsegment.txt'
     segments=[json.loads(s) for s in src.read_text().splitlines() if s.lstrip().startswith('{')]
     ap=root/f'data/jasmin/output/alignment/pr_{variant}/{sp}/{sp}_pr_{variant}.power.json'
     slots=json.loads(ap.read_text())['alignments']
     tokens=collapse_repeated_tokens([NormalizedToken(w,i) for i,s in enumerate(segments) for w in text_norm_all(s['text']).split()])
     hyp=[w for slot in slots for w in slot['hyp'].split()]
     if [t.word for t in tokens]!=hyp:raise ValueError('ASR_stream_mismatch_no_automatic_remapping')
     owners=[t.row_index for t in tokens];indices={i:[] for i in range(len(segments))}
     for j,i in enumerate(owners):indices[i].append(j)
     slotinfo=[];offset=0
     for slot in slots:
      nh=len(slot['hyp'].split());ix=list(range(offset,offset+nh));offset+=nh
      slotinfo.append(dict(ref=slot['ref'].split(),hyp=slot['hyp'].split(),label=slot['align'],indices=ix,segments=set(owners[j] for j in ix)))
     prompt={i:[] for i in indices};groups={i:[] for i in indices};ambiguous=set()
     for k,slot in enumerate(slotinfo):
      segment_set=slot['segments']
      if not segment_set:
       left=next((s['segments'] for s in reversed(slotinfo[:k]) if s['segments']),set())
       right=next((s['segments'] for s in slotinfo[k+1:] if s['segments']),set())
       segment_set=left|right
      if len(segment_set)!=1:
       ambiguous.update(segment_set);continue
      i=next(iter(segment_set));prompt[i].extend(slot['ref'])
      if slot['label']=='S' and slot['ref'] and slot['indices']:
       local={j:q for q,j in enumerate(indices[i])}
       groups[i].append(dict(asr_indices=slot['indices'],local_indices=[local[j] for j in slot['indices']],prompt_words=slot['ref'],asr_words=slot['hyp'],mapping='PR_ASR_S_slot'))
     for i,s in enumerate(segments):
      status='ok'
      if i in ambiguous:status='ambiguous_prompt_segment_boundary'
      elif not indices[i] or not prompt[i]:status='empty_candidate'
      audit.append(dict(variant=variant,speaker=sp,segment_id=s['id'],status=status))
      if status!='ok':continue
      out.write(json.dumps(dict(speaker=sp,variant=variant,segment_id=s['id'],audio_path=str(root/f'data/jasmin/raw/{sp}/{sp}.wav'),start=float(s['start']),end=float(s['end']),prompt_words=prompt[i],asr_words=[hyp[j] for j in indices[i]],asr_indices=indices[i],substitutions=groups[i],source_alignment=str(ap),source_segments=str(src),boundary_source='existing_ASR_newsegment_not_guaranteed_linguistic_sentence'),ensure_ascii=False)+'\n');written+=1
    except (OSError,ValueError,KeyError) as e:audit.append(dict(variant=variant,speaker=sp,status='speaker_failed',error=str(e)))
 args.output.with_suffix('.audit.json').write_text(json.dumps(audit,indent=2)+'\n')
 print(json.dumps(dict(written_segments=written,failed_speakers=sum(r['status']=='speaker_failed' for r in audit),skipped_segments=sum(r['status'] not in ['ok','speaker_failed'] for r in audit))))

if __name__=='__main__':main()
