"""Conservative MT-anchored local C/C/I repair. Never changes input alignments."""
from collections import defaultdict
import copy
from build_alignment_label_table import expand_side

WEIGHTS={'C':0,'S':4,'I':3,'D':3}


def unique_alignment(ref,hyp):
    """Return slots and cost only when the minimum-cost path is unique."""
    n,m=len(ref),len(hyp)
    cost=[[0]*(m+1) for _ in range(n+1)]
    ways=[[0]*(m+1) for _ in range(n+1)];ways[0][0]=1
    previous={}
    for i in range(n+1):
        for j in range(m+1):
            if not i and not j:continue
            options=[]
            if i:options.append((cost[i-1][j]+3,i-1,j,'D'))
            if j:options.append((cost[i][j-1]+3,i,j-1,'I'))
            if i and j:options.append((cost[i-1][j-1]+(0 if ref[i-1]==hyp[j-1] else 4),i-1,j-1,'C' if ref[i-1]==hyp[j-1] else 'S'))
            best=min(t[0] for t in options);winners=[t for t in options if t[0]==best]
            cost[i][j]=best;ways[i][j]=min(2,sum(ways[a][b] for _,a,b,_ in winners));previous[i,j]=winners[0]
    if ways[n][m]!=1:return None,cost[n][m]
    slots=[];i,j=n,m
    while i or j:
        _,a,b,label=previous[i,j]
        slots.append(dict(align=label,ref=ref[i-1] if a<i else '',hyp=hyp[j-1] if b<j else ''))
        i,j=a,b
    return list(reversed(slots)),cost[n][m]


def slot_tags(slots,side):
    result=defaultdict(list)
    for t in expand_side(slots,side):result[t.slot_index].append(t)
    return result


def exact_pairs(slots):
    refs=slot_tags(slots,'ref');hyps=slot_tags(slots,'hyp')
    return {hyps[i][0].index:refs[i][0].index for i,s in enumerate(slots)
            if s['align']=='C' and len(refs[i])==len(hyps[i])==1 and refs[i][0].word==hyps[i][0].word}


def repair_alignment(pr_mt,mt_asr,pr_asr,max_window_words=80):
    # Shared streams must match exactly; mismatched sources are audited, never silently normalized.
    streams={name:[t.word for t in expand_side(slots,side)] for name,slots,side in
             [('pr1',pr_mt,'ref'),('pr2',pr_asr,'ref'),('mt1',pr_mt,'hyp'),('mt2',mt_asr,'ref'),('asr1',mt_asr,'hyp'),('asr2',pr_asr,'hyp')]}
    mismatch=[key for key in ['pr','mt','asr'] if streams[key+'1']!=streams[key+'2']]
    pm=exact_pairs(pr_mt);ma=exact_pairs(mt_asr);direct=exact_pairs(pr_asr)
    ah=expand_side(pr_asr,'hyp')
    bridge={a:pm[m] for a,m in ma.items() if m in pm}
    targets=sorted(a for a in bridge if ah[a].label=='I')
    if mismatch:
        return copy.deepcopy(pr_asr),[dict(asr_index=a,prompt_index=bridge[a],status='unresolved_shared_sequence_mismatch',detail=','.join(mismatch)) for a in targets],[]
    if not targets:return copy.deepcopy(pr_asr),[],[]
    stable=sorted((a,p,ah[a].slot_index) for a,p in bridge.items() if direct.get(a)==p)
    stable=[(-1,-1,-1)]+stable+[(len(ah),len(streams['pr1']),len(pr_asr))]
    proposed=[]
    for a in targets:
        p=bridge[a]
        left=max((t for t in stable if t[0]<a and t[1]<p),key=lambda t:t[0])
        right=min((t for t in stable if t[0]>a and t[1]>p),key=lambda t:t[0])
        start,end=left[2]+1,right[2]
        proposed.append(dict(start=start,end=end,left=left,right=right,targets=[a]))
    windows=[]
    for w in sorted(proposed,key=lambda w:w['start']):
        if windows and w['start']<windows[-1]['end']:
            old=windows[-1];old['targets']+=w['targets']
            if w['end']>old['end']:old['end']=w['end'];old['right']=w['right']
        else:windows.append(w)
    reports=[];changes=[];replacements=[]
    for w in windows:
        start,end=w['start'],w['end'];left,right=w['left'],w['right']
        old=pr_asr[start:end];a0,a1=left[0]+1,right[0];p0,p1=left[1]+1,right[1]
        reason=None;new=[];total=0
        if max(a1-a0,p1-p0)>max_window_words:reason='window_too_large'
        elif any(len(s['ref'].split())>1 or len(s['hyp'].split())>1 for s in old):reason='multiword_window_requires_review'
        else:
            anchors=sorted((p,a) for a,p in bridge.items() if a0<=a<a1 and p0<=p<p1)
            if any(anchors[i][0]>=anchors[i+1][0] or anchors[i][1]>=anchors[i+1][1] for i in range(len(anchors)-1)):
                reason='nonmonotonic_bridge'
            else:
                pi,ai=p0,a0
                for pp,aa in anchors+[(p1,a1)]:
                    segment,cost=unique_alignment(streams['pr1'][pi:pp],streams['asr1'][ai:aa])
                    if segment is None:reason='multiple_constrained_optima';break
                    new.extend(segment);total+=cost
                    if pp<p1:
                        if streams['pr1'][pp]!=streams['asr1'][aa]:raise ValueError('Nonexact anchor')
                        new.append(dict(align='C',ref=streams['pr1'][pp],hyp=streams['asr1'][aa]))
                    pi,ai=pp+1,aa+1
                old_cost=sum(WEIGHTS[s['align']] for s in old)
                if reason is None and total>old_cost:reason='higher_word_edit_cost'
        status='repaired' if reason is None else 'unresolved_'+reason
        for a in sorted(set(w['targets'])):
            reports.append(dict(asr_index=a,prompt_index=bridge[a],status=status,start_slot=start,end_slot_exclusive=end))
        if reason is None:
            assert ' '.join(s['ref'] for s in new).split()==' '.join(s['ref'] for s in old).split()
            assert ' '.join(s['hyp'] for s in new).split()==' '.join(s['hyp'] for s in old).split()
            replacements.append((start,end,new))
            changes.append(dict(start_slot=start,end_slot_exclusive=end,old_cost=old_cost,new_cost=total,old=old,new=new,targets=w['targets']))
    output=copy.deepcopy(pr_asr)
    for start,end,new in reversed(replacements):output[start:end]=new
    for side in ['ref','hyp']:
        assert [t.word for t in expand_side(output,side)]==[t.word for t in expand_side(pr_asr,side)]
    final=exact_pairs(output)
    for r in reports:
        if r['status']=='repaired':assert final[r['asr_index']]==r['prompt_index']
    return output,reports,changes
