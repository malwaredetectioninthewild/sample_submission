import re

DEPTH_SEP = '@@'


MAX_LENGTH_PER_FIELD  = {'regs_created': 100, 'regs_deleted': 100, 
                         'mutexes_created': 60, 'processes_created': 15, 
                         'files_created': 250, 'processes_injected': 25}

MAX_LENGTH_PER_ACTION = {'regs_created': 14, 'regs_deleted': 9, 
                         'mutexes_created': 1, 'processes_created': 25, 
                         'files_created': 12, 'processes_injected': 2}

# token cleaner
def TOKEN_CLEANER(tok):
    return clean_up_token(tok, max_token_length=15)

def TOKEN_FILTER(tok):
    return False if get_token_length(tok) < 1 else True


def clean_up_token(token, max_token_length=10):

    # remove non-ascii
    new_token = re.sub(r'[^\x00-\x7F]', 'x', token)
    
    # remove all non alphanumeric characters
    # new_token = re.sub('[^a-z0-9]', '', new_token.lower())
    new_token = re.sub(r'[_]|[\s+]|,|\$|\?|!|\.|-|{|}|:|\[|\]|~|=|/|//|\"|\(|\)|\'|\\|\-', '', new_token.lower())

    return token_shortener(new_token, max_token_length)

def token_shortener(token, max_token_length=15):

    tok, depth = get_token_characters(token)

    short_tok = ''.join(tok[:max_token_length]) 

    short_tok = short_tok + f'{DEPTH_SEP}{depth}' if depth else short_tok

    return short_tok

def get_token_length(token):
    tok, _ = get_token_characters(token)
    return len(tok)

def get_token_characters(token):
    rem_characters = []
    cur_index = 0

    token_depth = token.split(DEPTH_SEP)
    if len(token_depth) > 1 and is_int(token_depth[-1]):
        tok, depth = DEPTH_SEP.join(token_depth[:-1]), token_depth[-1]
    else:
        tok, depth = token, None

    for match in re.finditer('<\w+>', tok):
        
        rem_characters.extend(tok[cur_index: match.span()[0]])
        rem_characters.append(match.group())
        cur_index = match.span()[1]

    rem_characters.extend(tok[cur_index:])

    return rem_characters, depth

def is_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False
    
def split_filename_extension(fname):

    # filename and extension
    tokens = fname.split('.')
    fname_toks = tokens[:1]

    exts = tokens[1:]
    exts = process_extensions(exts)

    fname_toks.extend([t for t in exts if len(t)>=5]) # long token, unlikely to be an extension
    exts = [t for t in exts if len(t) < 5 and len(t) > 0]

    if len(exts) == 0:
        exts = ['noextension']

    return '_'.join(fname_toks), exts

# ignore after any non alpha numeric character
def process_extensions(exts):
    return [re.split('[^a-z0-9]', ext)[0] for ext in exts]


def add_path_depth(path_toks, enabled=False):
    return [f'{t}{DEPTH_SEP}{ii+1}' for ii,t in enumerate(path_toks)] if enabled else path_toks

def tokenize_entry(entry, field, add_depth):
    if field == 'regs_created':
        # only take the key for now
        tokens = add_path_depth(entry.split('\\'), add_depth)
        tok_types = ['reg_c' for _ in tokens]

    elif field == 'regs_deleted':
        tokens = add_path_depth(entry.split('\\'), add_depth)
        tok_types = ['reg_d' for _ in tokens]

    elif field == 'mutexes_created':
        tokens = [entry] # no need to split
        tok_types = ['mtx_c']

    elif field == 'processes_created':
        tokens = []
        tok_types = []

        # first split with space
        ptokens = entry.split(' ')
        for t in ptokens:
            if any(re.findall(r'\b[a-z]:\\', t)) or '.exe' in t or '.dll' in t or '.bat' in t or '.ocx' in t or '.sys' in t: #path or executable
                path_toks = t.split('\\')
                path, fname = path_toks[:-1], path_toks[-1]
                tokens.extend(add_path_depth(path, add_depth))
                tok_types.extend(['proc_p' for _ in path])

                ftoks, fttypes = tokenize_filename(fname, 'proc')
                tokens.extend(ftoks)
                tok_types.extend(fttypes)
            
            # urls
            elif any(re.findall(r'(https?:\\)|(www.)|(ftps?:\\)', t)):
                matches = list(re.finditer(r'(https?:\\)|(www.)|(ftps?:\\)', t))
                url_type = re.split('\:|\.', matches[-1].group())[0]
                tokens.append(f'<{url_type}url>')
                tok_types.append('proc_m')

            # ipaddress:port
            elif any(re.findall(r'((localhost)|([0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}))(:[0-9]{1,5})?', t)):
                tname = '<'
                tname = tname + 'lochost' if 'localhost' in t else tname + 'ipaddr'
                tname = tname + 'port' if ':' in t else tname     
                tname = tname + '>'
                tokens.append(tname)
                tok_types.append('proc_m')

            else: # no path
                tokens.append(t)
                tok_types.append('proc_m') # miscellaneous

    elif field == 'files_created':
        tokens = []
        tok_types = []

        ptokens = entry.split('\\')
        path, fname = ptokens[:-1], ptokens[-1]
        tokens.extend(add_path_depth(path, add_depth))
        tok_types.extend(['file_p' for _ in path])

        ftoks, fttypes = tokenize_filename(fname, 'file')
        tokens.extend(ftoks)
        tok_types.extend(fttypes)  

    elif field == 'processes_injected':
        tokens = []
        tok_types = []
        ftoks, fttypes = tokenize_filename(entry, 'inj')
        tokens.extend(ftoks)
        tok_types.extend(fttypes)    

    assert len(tokens) == len(tok_types), 'entry: num tokens != num token types'

    return [(t,tt) for t, tt in zip(tokens, tok_types)]


def tokenize_filename(entry, act_type):
    fname, exts = split_filename_extension(entry)

    # handle the extension
    if len(exts) > 1: # multiple extensions
        merged_exts = ''.join(exts)
        ext_type = f'{act_type}_me'
    else:
        merged_exts = exts[0]
        ext_type = f'{act_type}_e'

    return [fname, merged_exts], [f'{act_type}_f', ext_type]


def tokenize_report(rep, rep_fields=None, add_depth=True, concat=False, return_token_types=True):
    
    tokenized_report = {}

    if rep_fields is None: # all fields
        collect_fields = ['regs_created', 'regs_deleted', 'mutexes_created', 'processes_created', 'files_created', 'processes_injected'] 
        
    else: # collect specified fields
        collect_fields = rep_fields
    

    for field in rep:
        if field not in collect_fields:
            continue
        tokenized_report[field] = []

        for entry in rep[field]:
            tokens = tokenize_entry(entry, field, add_depth)
            filtered_tokens, filtered_tok_types = [], []

            # clean up and filter the tokens
            for ii in range(len(tokens)):
                ct = TOKEN_CLEANER(tokens[ii][0])
                if TOKEN_FILTER(ct): # returns true for the tokens that will be kept
                    filtered_tokens.append(ct)
                    filtered_tok_types.append(tokens[ii][1])
            

            if return_token_types:
                tokens = list(zip(filtered_tokens, filtered_tok_types))
            else:
                tokens = filtered_tokens

            if len(tokens) == 0:
                continue

            tokens = tokens[:MAX_LENGTH_PER_ACTION[field]]

            if concat:
                tokenized_report[field].extend(tokens) # create a single list of tokens from all actions

            else:
                tokenized_report[field].append(tokens) # each action will be a different list of tokens

    tokenized_report = {field:entries[:MAX_LENGTH_PER_FIELD[field]] for field, entries in tokenized_report.items()}

    return tokenized_report
