# GloVe

Simple Python only implementation of Global Vectors (GloVe)

When researching GloVe, I noticed there were very few implementations freely available. Of those I could find, they usually involved other languages or just did not work.

This is an attempted replication of the original paper [GloVe: Global Vectors for Word Representation (2014)](https://nlp.stanford.edu/pubs/glove.pdf).
        

TODO:
+ Add support for Glove class to work with a text file directly

Example use:

```python
from   glove  import get_preprocessor, logger, Glove

import pandas as pd
import numpy  as np
import os
import gc

logger('"glove-example.py" started')

file = 'labeledTrainData.tsv.zip'

if file not in os.listdir():
    raise FileNotFoundError(f'Please download "{file}" from https://www.kaggle.com/c/word2vec-nlp-tutorial to run this script')
    
# Download data from https://www.kaggle.com/c/word2vec-nlp-tutorial
data    = pd.read_csv(file, header = 0, delimiter = '\t')
reviews = data['review'].values

del data; gc.collect()

preprocessor = get_preprocessor(stem = True, stop = True, min_length = 3)
folder       = 'dump'
os.makedirs(folder, exist_ok = True)

file = 'setup.npz'
if file in os.listdir(folder):
    logger(f'found saved {file}')
    glove  = Glove(None   , preprocessor, random_state = 2021)
    glove.load(f'{folder}/{file}')
else:
    glove  = Glove(reviews, preprocessor, random_state = 2021)
    glove.dump_co_occurance(f'{folder}/{file}')
    logger(f'saved {file}')

print()
for dim in [2, 10, 50, 100, 200, 300, 400, 500, 600]:
    # Needs more iterations to converge for higher dims
    glove.fit(dim, eta = 1e-2, epochs = 200 if dim < 300 else 1000, optimiser = 'adam', decay = 1e-2)
    filename = f'glove-{dim}.npz'
    glove.dump_vectors(f'{folder}/{filename}')
    logger(f'saved {filename}')
    print()
```
