# GloVe

Simple Python only implementation of Global Vectors (GloVe)

When researching GloVe, I noticed there were very few implementations freely available. Of those I could find, they usually involved other languages or just did not work.

This is an attempted replication of the original paper [GloVe: Global Vectors for Word Representation (2014)](https://nlp.stanford.edu/pubs/glove.pdf).
        

TODO:
+ Add support for Glove class to work with a text file directly

Example use:

```python
from   glove   import logger, Glove
from   utility import set_log_path

from   time    import process_time
import pandas  as pd
import numpy   as np
import os, gc

folder = 'dump-text8-2'
os.makedirs(folder, exist_ok = True)

set_log_path(folder)

logger('"glove-text8.py" started')
    
try:
    with open('text8') as f:
        words = f.read()
except:
    msg = 'Missing "text8" file!\nTry "wget https://data.deepai.org/text8.zip" and unzipping text8.zip then retrying this script!'
    raise Exception(msg)

def preprocessor(text, to_tokens = False):
    if to_tokens:
        return text.split()
    return [text.split()]

file = 'setup.npz'
if file in os.listdir(folder):
    logger(f'found saved {file}')
    glove  = Glove(None   , preprocessor, random_state = 2021, x_min = 2, x_max = 20)
    glove.load(f'{folder}/{file}')
else:
    start  = process_time()
    glove  = Glove([words], preprocessor, random_state = 2021, x_min = 2, x_max = 20)
    time   = process_time() - start
    glove.dump_co_occurance(f'{folder}/{file}', time = time)

del words; gc.collect()

print()
for dim in [2, 10, 50, 100, 200, 300, 400, 500, 600]:
    filename = f'glove-{dim}.npz'
    start    = process_time()
    glove.fit(dim, eta = 0.5, epochs = 500, optimiser = 'adam', decay = 1e-2)
    time     = process_time() - start
    glove.dump_vectors(f'{folder}/{filename}', time = time)
    print()
```
