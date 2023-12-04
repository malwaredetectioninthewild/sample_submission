import numpy as np

from .trace_to_ngrams import FIELD_ORDER


def get_bounds(len_list, ns):

    helper = lambda n: n*len_list - (n*(n-1))/2

    all_indices = []

    for n in ns:
        if n > len_list:
            cur_indices = np.arange(helper(len_list-1), helper(len_list)).astype(int)
        else:
            cur_indices = np.arange(helper(n-1), helper(n)).astype(int)

        all_indices.extend(cur_indices)

    return sorted(np.unique(all_indices))

def get_len(num_ngrams, n):
    a = (2*num_ngrams)/n
    b = n - 1
    return int((a + b)/2)


def ngram_sequence_to_feature_counts(sequences, keep_fields=None, feat_n=2, keep_ns=[1,2], log_scaler=True):

    fvec_size = 14
    
    keep = list(FIELD_ORDER.keys()) if keep_fields is None else keep_fields
    keep_idx = [FIELD_ORDER[k] for k in keep]

    list_lens = {ii:get_len(ii, feat_n) for ii in np.arange(100)}
    keep_indices = {ll:set(get_bounds(ll, keep_ns)) for ll in list_lens.keys()}

    num_feats = 2**fvec_size if fvec_size < 20 else fvec_size

    fmat = np.zeros((len(sequences), num_feats))

    for ii in range(len(sequences)):

        for field_idx in keep_idx:

            field_acts = sequences[ii][field_idx]

            for act in field_acts:

                cur_list_len = list_lens[len(act)]
                cur_keep_indices = keep_indices[cur_list_len]

                for tii, tidx in enumerate(act): # ngrams
                    if tii in cur_keep_indices:
                        fmat[ii][tidx%num_feats] += 1

    if log_scaler:
        fmat = np.log2(fmat+1)

    return fmat

