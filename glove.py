# Copyright (c) 2021, Jordan Taylor
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#  1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from   utility    import Update, logger, rprint
from   optimisers import get_optimiser
from   scipy      import sparse

import numpy as np
import gc

def get_preprocessor(html = True, lower = True, stop = False, stem = False, min_length = 0):
    """ Simplistic preprocessor for text """
    
    def preprocessor(sentences, to_tokens = False):
        from   bs4         import BeautifulSoup             
        from   nltk.corpus import stopwords
        from   nltk.stem   import PorterStemmer
        import re

        # 1. Remove HTML
        if html:
            sentences = BeautifulSoup(sentences, features = "html.parser").get_text() 

        # 2. Remove non-letters (^a-zA-Z means everything but lower and uppercase letters)
        sentences = re.sub(r"[^a-zA-Z\.]", " ", sentences).split('.')

        # 3. Convert to lower case and split
        if lower:
            sentences = [sentence.lower().split() for sentence in sentences]
        else:
            sentences = [sentence.split() for sentence in sentences]
            
        # 4. Stopwords
        if stop:
            stops     = set(stopwords.words("english"))
            sentences = [[token for token in sentence if not token in stops] for sentence in sentences]
            
        # 5. Stem
        if stem:
            ps        = PorterStemmer().stem
            sentences = [list(map(ps, sentence)) for sentence in sentences]
        
        # 6. Min length
        sentences = [[token for token in sentence if len(token) >= min_length] for sentence in sentences]
        
        # 7. Return list of all tokens or list of sentences
        if to_tokens:
            return sum(sentences, [])
        return sentences
    
    return preprocessor
    
class Glove():
    """
    Global Vector (GloVe) class in Python only
    
    Attempted replication of the original paper:
        Title   : "GloVe: Global Vectors for Word Representation"
        Authors : J. Pennington, R. Socher, C. D. Manning
        Year    : 2014
        Link    : https://nlp.stanford.edu/pubs/glove.pdf
    
    Modifications to the paper:
        • Generalised optimiser inplace of AdaGrad
        • Can preset x_max arbitrarily capping values higher
        • x_min option to ignore co-occurances with values below x_min
    
    Parameters
    ================
        documents    : array, list, tuple [OPTIONAL]
                       Collection of texts.
                       
        preprocessor : function [OPTIONAL]
                       Preprocessing function that should be applied to the text before computations are done.
                       
        alpha        : float [OPTIONAL]
                       See paper.
                       
        window_wise  : int [OPTIONAL]
                       See paper.
                       
        x_min        : float [OPTIONAL]
                       Co-occurances below this threshold will be treated as 0.
                       
        x_max        : float [OPTIONAL]
                       See paper.
                       
        random_state : int [OPTIONAL]
                       Random number generation for reproducibility via numpy.random.seed.
    """
    def __init__(self, documents = None, preprocessor = None, alpha = 0.75, window_size = 15, x_min = 1, x_max = 10, random_state = None):
        if preprocessor is None:
            preprocessor = lambda x : x.split('')
        
        self.preprocessor = preprocessor
        self.alpha        = alpha
        self.window_size  = window_size
        self.x_min        = x_min # If set, then treat co-occurance values below this limit as 0
        self.x_max        = x_max
        self.random_state = random_state
        
        if not isinstance(documents, type(None)):
            self.compute_coocurance(documents)
            
    def compute_coocurance(self, documents):
        """ Computes the sparse co-occurance matrix storing only the rows and and values """
        rprint('counting unique tokens')
        V = set()
        for document in documents:
            tokens = self.preprocessor(document, to_tokens = True)
            V     |= set(tokens)
        logger(f'counted {len(V):,d} unique tokens')
        
        # Vocabulary dictionary - map each token to an integer for indexing
        self.V = {k : v for v, k in enumerate(V)}
        
        sparse = {}
        N      = len(documents)
        u      = Update('computing co-occurance matrix : document', N)
        for n, document in enumerate(documents, 1):
            u.increment()
            for tokens in self.preprocessor(document):
                ntokens = len(tokens)
                for t, token1 in enumerate(tokens):
                    # Center token
                    i      = self.V[token1]
                    
                    # Window (forwards only)
                    window = range(t + 1, min(ntokens, t + self.window_size))
                    
                    for w in window:
                        increment = 1 / (w - t)
                        
                        # Token ahead
                        token2    = tokens[w]
                        j         = self.V[token2]
                        
                        # Increment forwards and backwards
                        if (i, j) in sparse:
                            sparse[(i, j)] += increment
                            sparse[(j, i)] += increment
                        else:
                            sparse[(i, j)]  = increment
                            sparse[(j, i)]  = increment
                            
            # Verbose updates every 1000 documents   
            if n % 1000 == 0:
                u.display()
                
        # Final update if not already given
        if n % 1000 != 0:
            u.increment()
            u.display()
        
        rprint('converting to sparse indices and values')
        # Store rows and values
        self.r, self.c = np.array(list(sparse)).T.astype('int32')
        self.x         = np.array(list(sparse.values())).astype('int32')
            
        logger(f'computed co-occurance matrix with {len(self.V) ** 2:,d} elements and {len(self.x):,d} interactions')
        
        self.compute_min_idx()
        
    def compute_min_idx(self):
        if self.x_min is not None:
            self._idx = np.where(self.x_min <= self.x)[0]
            logger(f'{len(self._idx):,d} interactions above x_min = {self.x_min}')
            
    def load(self, path):
        """ Loads vocabulary and sparse co-occurance matrix """
        npz        = np.load(path, allow_pickle = True)
        self.V     = npz['V'].tolist() # Vocabulary
        self.r     = npz['r']          # Rows of non-zero co-occurances
        self.c     = npz['c']          # Cols of non-zero co-occurances
        self.x     = npz['x']          # Vals of non-zero co-occurances
        logger(f'set co-occurance matrix with {len(self.V) ** 2:,d} elements, {len(self.x):,d} interactions, and {len(self.V):,d} unique tokens')
        
        self.compute_min_idx()
        
    def fit(self, vector_size, eta = 1e-4, epochs = 100, optimiser = 'adagrad', stop = None, tau = 1e-7, **optimiser_kwargs):
        
        if isinstance(optimiser, str):
            optimiser = get_optimiser(optimiser)
        
        logger(f'fitting with vector size = {vector_size:,d}')
        
        r, c, x    = self.r, self.c, self.x
        
        # Filter out not frequent enough co-occurances
        if self.x_min is not None:
            _r, _c, x = r[self._idx], c[self._idx], x[self._idx]
            ur        = {r : i for i, r in enumerate(np.unique(_r))}
            uc        = {c : i for i, c in enumerate(np.unique(_c))}
            r         = np.array([ur[r] for r in _r]).astype('int32')
            c         = np.array([uc[c] for c in _c]).astype('int32')
            
            # Free memory
            del _r, _c, ur, uc; gc.collect()
            
        # Compute max if not set, then cap values
        x_max      = x.max() if self.x_max is None else self.x_max
        if self.x_max is not None:
            rprint('setting x_max upper bound')
            _x     = np.minimum(x, x_max)
            
            rprint('precomputing f(X)')
            fx     = (_x / x_max) ** self.alpha
            
            # Free memory
            del _x; gc.collect()
        else:
            rprint('precomputing f(X)')
            fx     = (x / x_max) ** self.alpha
        
        rprint('precomputing log(X)')
        lx     = np.log(x)
        
        
        # Free memory
        del x; gc.collect()
        
        np.random.seed(self.random_state)
        
        shape     = len(np.unique(r)), vector_size
        
        rprint('initialising word vectors and bias vector variables')
        W1        = np.random.normal(scale = 0.5, size = shape).astype('float32')
        W2        = np.random.normal(scale = 0.5, size = shape).astype('float32')
        b1        = np.random.normal(scale = 0.5, size = shape[0]).astype('float32')
        b2        = np.random.normal(scale = 0.5, size = shape[0]).astype('float32')
        
        # As sparse matrix may have multiple entries per row, compute these entries before hand for later ease
        rprint('computing masks for optimisation')
        rmasks = {}
        cmasks = {}
        
        for d, masks in zip([r, c], [rmasks, cmasks]):
            for i, val in enumerate(d):
                if val not in masks:
                    masks[val] = []
                masks[val] += [i]
        
        # Free memory (masks is linked to cmasks so cannot delete it)
        del d; gc.collect()
        
        # Initialise optimisers (W1, W2, b)
        optim     = [optimiser(eta = eta, **optimiser_kwargs) for _ in range(3)]
        logger(f'initialised variables')
        
        u         = Update('optimising epoch', epochs)
        L         = self.L = np.ones(epochs + 1) * np.inf
        N         = fx.sum()
        lo        = np.inf
        for i in range(epochs):
            
            # Early stopping condition if over the last "stop" iterations there is a total variation of less than "tau"
            if stop is not None and i >= stop:
                if (L[i - stop: i].max() / L[i - stop: i].min() - 1) <= tau:
                    break
                
            delta           = (W1[r] * W2[c]).sum(axis = 1) + b1[r] + b2[c] - lx
            L[i]            = np.mean(fx * np.square(delta))
            
            # Store the best
            if L[i] < lo:
                best = [W1.copy(), W2.copy(), b1.copy(), b2.copy()]
                lo   = L[i]
    
            # Chain rule of loss function of the form L = fx * (delta ^ 2) w.r.t. delta (ignoring proportional constants)
            chain = (fx * delta)

            # Compute gradients to update W and b i.e. differentiate delta w.r.t W and b respectively
            #
            # Steps:
            #   • Compute adjusted gradients using optimiser
            #   • Aggregate gradients for each token (row of W)
            #   • Update parameter
            #   • Free space to reduce memory cost
            #
            # Do for W1 (optim[0]), W2 (optim[1]), b1 (optim[2]), b2 (optim[2])
            # Gradients for b1 and b2 are similar just with different aggregation masks r and c
            gw1   = optim[0](np.einsum('c,cv->cv', chain, W2[c]).astype('float32'))
            gW1   = np.zeros_like(W1)
            for j, mask in rmasks.items():
                gW1[j] += gw1[mask].mean(axis = 0)
            W1   -= gW1
            del gw1, gW1; gc.collect()
            
            gw2   = optim[1](np.einsum('c,cv->cv', chain, W1[r]).astype('float32'))
            gW2   = np.zeros_like(W2)
            for j, mask in cmasks.items():
                gW2[j] += gw2[mask].mean(axis = 0)
            W2   -= gW2
            del gw2, gW2; gc.collect()
            
            # Common gradients for b1 and b2 with different aggregations
            gb    = optim[2](chain.astype('float32'))
            
            gb1   = np.zeros_like(b1)
            for j, mask in rmasks.items():
                gb1[j] += gb[mask].mean(axis = 0)
            b1   -= gb1
            del gb1; gc.collect()
            
            gb2   = np.zeros_like(b2)
            for j, mask in cmasks.items():
                gb2[j] += gb[mask].mean(axis = 0)
            b2   -= gb2
            del chain, gb2; gc.collect()
            
            # Verbose update
            u.increment()
            u.display(loss = L[i], best = lo)
        else:
            # Enters the else statement only if the for loop completes without break
            i += 1
            
        delta  = (W1[r] * W2[c]).sum(axis = 1) + b1[r] + b2[c] - lx
        L[i]   = np.sum(fx * np.square(delta)) / N
        
        if L[i] == L.min():
            best = [W1.copy(), W2.copy(), b1.copy(), b2.copy()]
                
        self.W, self.Wc, self.b, self.bc = best
        self.L = L[:i + 1]
        logger(f'optimised over {i:,d} epochs (best loss = {min(L):,.3e}, final loss = {L[i]:,.3e})')

        return self
        
    def dump_co_occurance(self, path, **kwargs):
        """ Dumps the vocabulary and co-occurance matrix """
        np.savez(path, V = self.V, r = self.r, c = self.c, x = self.x, **kwargs)
        logger(f'dumped co-occurance at "{path}"')
        
    def dump_vectors(self, path, **kwargs):
        """ Dumps the word and context weight matrices and bias vectors """
        if self.x_min is not None:
            rprint('computing valid mask')
            sp    = sparse.csr_matrix((self.x, (self.r, self.c)))
            valid = np.where(sp.max(axis = 1).A.flatten() > self.x_min)[0]
            np.savez(path, W = self.W, Wc = self.Wc, b = self.b, bc = self.bc, L = self.L, valid = valid)
        else:
            np.savez(path, W = self.W, Wc = self.Wc, b = self.b, bc = self.bc, L = self.L, **kwargs)
        logger(f'dumped vectors at "{path}"')
