from   glove       import get_preprocessor, Glove, logger
import pandas as pd
import numpy  as np
import os
import gc

logger('"glove-example.py" started')

file = 'labeledTrainData.tsv.zip'

if file not in os.listdir():
    raise FileNotFoundError(f'Please download "{file}" from https://www.kaggle.com/c/word2vec-nlp-tutorial to run this script')
    
# Download data from https://www.kaggle.com/c/word2vec-nlp-tutorial
data = pd.read_csv(file, header = 0, delimiter = '\t')

np.random.seed(2021)

n     = len(data)
idx   = np.random.permutation(n)
split = int(0.8 * n)
train = idx[:split]
test  = idx[split:]

X_train = data['review'].values[train]

del data; gc.collect()

preprocessor = get_preprocessor(stem = True, stop = True, min_length = 3)
folder       = 'dump'
os.makedirs(folder, exist_ok = True)

file       = 'setup.npz'
if file in os.listdir(folder):
    logger(f'found saved {file}')
    glove  = Glove(None   , preprocessor, random_state = 2021)
    glove.load(f'{folder}/{file}')
else:
    glove  = Glove(X_train, preprocessor, random_state = 2021)
    glove.dump_co_occurance(f'{folder}/{file}')
    logger(f'saved {file}')

del X_train; gc.collect()

print()
for dim in [2, 10, 50, 100, 200, 300, 400, 500, 600]:
    # Needs more iterations to converge for higher dims
    glove.fit(dim, eta = 1e-2, epochs = 200 if dim < 300 else 1000, optimiser = 'adam', decay = 1e-2)
    glove.dump_vectors(f'{folder}/glove-{dim}.npz')
    logger(f'saved {filename}')
    print()
    
