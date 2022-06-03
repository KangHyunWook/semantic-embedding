import pickle
import numpy as np

with open('word2vec.pkl', 'rb') as f:
    word2vec = pickle.load(f)

puppy_vec=word2vec['puppy']
#F5
print('====cosine similarity with puppy======')
for key in word2vec:
    if not key == 'puppy':
        cos_sim=np.dot(puppy_vec, word2vec[key])/(
                    np.linalg.norm(puppy_vec)*np.linalg.norm(word2vec[key]))
        print(key, cos_sim)
    


