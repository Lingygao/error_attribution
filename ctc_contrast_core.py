"""CTC Viterbi alignment on identical emissions; no blank/OOV dropping."""
import numpy as np


def align(logp, ids, owners, blank=0):
    logp=np.asarray(logp,dtype=float);ids=np.asarray(ids,dtype=int)
    if logp.ndim!=2 or not len(logp) or not len(ids):raise ValueError('empty_audio_or_text')
    if not np.isfinite(logp).all():raise ValueError('nonfinite_emissions')
    if len(owners)!=len(ids) or np.any(ids==blank):raise ValueError('invalid_target')
    if len(logp)<len(ids)+int(np.sum(ids[1:]==ids[:-1])):raise ValueError('insufficient_frames')
    states=np.full(2*len(ids)+1,blank,int);states[1::2]=ids
    n=len(states);skip=np.zeros(n,bool);skip[2:]=(states[2:]!=blank)&(states[2:]!=states[:-2])
    dp=np.full(n,-np.inf);dp[0]=logp[0,blank];dp[1]=logp[0,ids[0]]
    back=np.zeros((len(logp),n),np.uint8)
    for t in range(1,len(logp)):
        one=np.r_[-np.inf,dp[:-1]];two=np.r_[-np.inf,-np.inf,dp[:-2]];two[~skip]=-np.inf
        cand=np.stack([dp,one,two]);step=np.argmax(cand,axis=0)
        dp=cand[step,np.arange(n)]+logp[t,states];back[t]=step
    s=n-1 if dp[-1]>=dp[-2] else n-2
    if not np.isfinite(dp[s]):raise ValueError('alignment_unreachable')
    path_score=float(dp[s]);path=np.empty(len(logp),int)
    for t in range(len(logp)-1,-1,-1):
        path[t]=s
        if t:s-=int(back[t,s])
    token_scores=[];word_frames={}
    for j,(label,owner) in enumerate(zip(ids,owners)):
        frames=np.flatnonzero(path==2*j+1)
        if not len(frames):raise ValueError('unvisited_target')
        if owner>=0:
            token_scores.append(float(np.exp(logp[frames,label]).mean()))
            word_frames.setdefault(int(owner),[]).extend(frames.tolist())
    if not token_scores:raise ValueError('no_scored_characters')
    return dict(score=float(np.mean(token_scores)),viterbi_log_score_per_frame=path_score/len(logp),
                word_frames={w:(min(fs),max(fs)+1) for w,fs in word_frames.items()})


def encode_words(words, tokenizer):
    """Character CTC targets; delimiters align but do not contribute to mean score."""
    vocab=tokenizer.get_vocab();delimiter=tokenizer.word_delimiter_token
    if delimiter not in vocab:raise ValueError('missing_word_delimiter')
    ids=[];owners=[]
    for i,word in enumerate(words):
        if i:ids.append(vocab[delimiter]);owners.append(-1)
        for char in word:
            candidates=[char,char.upper(),char.lower()]
            token=next((c for c in candidates if c in vocab),None)
            if token is None:raise ValueError('unsupported_character:'+repr(char))
            ids.append(vocab[token]);owners.append(i)
    return ids,owners
