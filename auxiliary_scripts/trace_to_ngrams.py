import pickle
import numpy as np
import hashlib


from .trace_tokenization import tokenize_report

FIELD_ORDER = {'regs_created':0, 'regs_deleted':1, 'mutexes_created':2, 'processes_created':3, 'files_created':4, 'processes_injected':5}
ORDER_FIELD = {0:'regs_created', 1:'regs_deleted', 2:'mutexes_created', 3:'processes_created', 4:'files_created', 5:'processes_injected'}


def get_tok_str(tok_type, tok):
    return f'{tok_type}::{tok}'


def extract_ngrams(report_paths, kept_tokens_dict, n=2):

    all_sequences = []

    for report_path in report_paths:

        with open(report_path, 'rb') as fp:
            report_file = pickle.load(fp)

        # first apply tokenization on the raw trace in the standardized format
        tokenized_report = tokenize_report(report_file['report'], add_depth=False, concat=False, return_token_types=True)
        cur_sequence = rep_to_sequence(tokenized_report, kept_tokens_dict)
        all_sequences.append(cur_sequence)
    
    ngrams = sequences_to_ngrams(all_sequences, n)

    return ngrams

# convert the tokenized trace into a sequence of numerical tokens (similar to how documents are processed in NLP)
def rep_to_sequence(tokenized_report, kept_tokens_info):

    rep_token_idxs = tokenized_report_to_token_idx(tokenized_report, kept_tokens_info)
    sequences = token_idx_to_sequence(rep_token_idxs)
    return sequences

# assign numerical indices to each string token
def tokenized_report_to_token_idx(tokenized_report, kept_tokens_info):

    tokenized_report_indices = {}

    rep_rares = []

    for field, acts in tokenized_report.items():
        
        tokenized_report_indices[field] = []

        for ii, act in enumerate(acts):

            cur_act = []

            for jj, (tok, type) in enumerate(act):

                s0, s1 = ('proc_f', 'proc_p')

                if type == 'proc_m':
                    if get_tok_str(s0, tok) in kept_tokens_info:
                        type = s0
                    elif get_tok_str(s1, tok) in kept_tokens_info:
                        type = s1
                    
                tok_str = get_tok_str(type, tok)

                if tok_str not in kept_tokens_info:
                    rep_rares.append((tok, field, ii, jj, type))
                    tok_str = get_tok_str(type, '<rare_singleton>')

                cur_act.append(tok_str)


            tokenized_report_indices[field].append(cur_act)

    # replace the rare tokens that occur multiple times in the report with special tokens 
    # (preserve non-random rare tokens)
    tok_fields = {}

    for r in rep_rares:
        if r[0] not in tok_fields:
            tok_fields[r[0]] = []
        tok_fields[r[0]].append(r[1])

    tok_fields = [(k, v) for k, v in tok_fields.items() if len(v) > 1]
    tok_fields = sorted(tok_fields, key=lambda r: (len(np.unique(r[1])), len(r[1])), reverse=True)[:25]
    multiples_idx = {r[0]:(idx+1) for idx, r in enumerate(tok_fields)}

    for tok, field, ii, jj, type in rep_rares:
        if tok in multiples_idx:
            tokenized_report_indices[field][ii][jj] = get_tok_str(type, f'<rare_{multiples_idx[tok]}>')

    return tokenized_report_indices


def token_idx_to_sequence(tokenized_report):

    sequence = []

    for field_idx in [0,1,2,3,4,5]:
        field_acts = tokenized_report[ORDER_FIELD[field_idx]]
        field_seq = []

        for act in field_acts:

            field_seq.append(np.asarray(act, dtype=str))
        
        sequence.append(np.asarray(field_seq, dtype=object))
        
    return np.asarray(sequence, dtype=object)

# take the sequences to tokens and create n-grams
def sequences_to_ngrams(sequences, n=2):   

    n = int(n)

    fvec_size = 32

    all_ngram_seq = []

    for seq in sequences:
        cur_ngram_seq = []

        for field_seq in seq:
            cur_ngram_field_seq = []
            # take the n-gram of each action seperately 
            # action order is not consistent between sandboxes so it doesn't matter
            for act in field_seq:
                ngrams, split_indices = find_ngrams(act, n)
                hash_indices = [get_token_md5(ngram, fvec_size, start_idx=0) for ngram in ngrams]
                cur_ngram_field_seq.append(np.asarray(hash_indices[:split_indices[n]], dtype=object))

            cur_ngram_field_seq = np.asarray(cur_ngram_field_seq, dtype=object)
            cur_ngram_seq.append(cur_ngram_field_seq)

        cur_ngram_seq = np.asarray(cur_ngram_seq, dtype=object)
        all_ngram_seq.append(cur_ngram_seq)

    return np.asarray(all_ngram_seq, dtype=object)

# we apply the hashing trick on each n-gram to map them to a fixed sized feature vector
def get_token_md5(cur_str, n_buckets, start_idx=0):

    md5 = int(hashlib.md5(cur_str.encode('utf-8')).hexdigest(), 16)
    return start_idx + (md5 % 2**n_buckets)

def find_ngrams(input_list, n):

    all_grams = []
    split_indices = {}
    for ii in range(1, n+1):
        n_grams =  list(zip(*[input_list[i:] for i in range(ii)]))
        n_grams = ['::'.join([str(t) for t in n_gram]) for n_gram in n_grams]
        all_grams.extend(n_grams)
        split_indices[ii] = len(all_grams)

    return all_grams, split_indices
