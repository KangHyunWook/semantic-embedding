from sys import exit
import numpy as np
import pickle
emb_file_path=r'C:\example-code\glove.6B.300d.txt'

def cvt_str2float(vec):
    float_vec=[]
    for v in vec:
        float_vec.append(float(v.strip()))
    return np.array(float_vec)

candidates=['dog', 'tiger', 'baby', 'cat', 'chocolate', 'puppy']
word_vec_map={}
for line in open(emb_file_path, encoding='utf-8'):
    splits=line.split(' ')
    
    word=splits[0]
    if word in candidates:
        
        vec=splits[1:]
        vec=cvt_str2float(vec)
        word_vec_map[word]=vec
  
with open('word2vec.pkl', 'wb') as f:
    pickle.dump(word_vec_map, f, pickle.HIGHEST_PROTOCOL)
    
print('pickle saved')


    