import gensim
import gensim.downloader as gloader
import numpy as np
import tensorflow as tf

def load_glove():
    emb_dimension = 100
    download_path = "glove-wiki-gigaword-{}".format(emb_dimension)
    try:
        word_emb_model = gloader.load(download_path)
    except ValueError as e:
        print("Invalid embedding model name! Check the embedding dimension:")
        print("Glove: 50, 100, 200, 300")
        raise e

    return word_emb_model




## vocabulary class


import tqdm
import json

class Vocabulary(object):
  def __init__(self):
    self.char2idx = {'PAD':0, 'UNK': 1, ' ': 2}
    self.idx2char = {0: 'PAD',1: 'UNK', 2: ' '}


  def __len__(self):
    return len(self.char2idx)

  def __contains__(self, key):
    if type(key) == int:
      return key in self.idx2char
    elif type(key) == str:
      return key in self.char2idx

  def __getitem__(self, key):
    if type(key) == int:
      return self.idx2char.get(key, 0)
    elif type(key) == str:
      return self.char2idx.get(key, 0)

  def __setitem__(self, key, item):
    if type(key) == int and type(item) == str:
      self.idx2char[key] = item
    elif type(key) == str and type(item) == int:
      self.char2idx[key] = item
    else:
      raise RuntimeError('Invalid (key, item) types.')

  def add(self, token):
    if token not in self.char2idx:
      index = len(self.char2idx)
      self.char2idx[token] = index
      self.idx2char[index] = token

  def get_vocab_list(self):
      words = [self[k] for k in range(0, len(self))]
      return words

  def toidx(self, tokens):
        return [self[tok] for tok in tokens]


  def build(words):
    vocab = Vocabulary()
    for w in words: vocab.add(w)
    return vocab

  def save_json(self, name):
    with open(f'Vocab{name}.json', 'w+') as f:
    # this would  place the entire output on one line
    # use json.dump(lista_items, f, indent=4) to "pretty-print" with four spaces per indent
      json.dump(self.char2idx, f)

  def load_json(self, name):
    with open(name) as f:
      temp = json.load(f)
    self.char2idx = {char: int(idx) for char, idx in temp.items()}
    self.idx2char = {int(idx): char  for char, idx in temp.items()}

  #build vocab from context, question and answer char
  def build_on_df(df):
    vocab = Vocabulary()
    for i, t in tqdm.tqdm(df.iterrows(), total = len(df)):

      for c in t['context']: vocab.add(c)

      for c in t['question']: vocab.add(c)

      for c in t['text']: vocab.add(c)

    return vocab

def load_vocabulary(path):
    vocab = Vocabulary()
    vocab.load_json(path)
    return vocab


def prepare_data(df, emb_model, charVocab):

  word_context = []
  word_question  = []
  char_context = []
  char_question = []
  answer_start = []
  answer_end = []

  unk_word = {}



  for i, t in df.iterrows():
    temp_word = []
    temp_char  = []
    oov_terms = check_OOV_terms(emb_model, t['context_list'])

    # formatting context
    for word in t['context_list']:
      #word embedding
      if word in oov_terms:
        temp_word.append(np.zeros((100,)))
        unk_word[word] = unk_word.get(word,0)+1
      else:
        temp_word.append(emb_model.get_vector(word))
      #char emb
      temp_char.append(charVocab.toidx(word))

    temp_char = tf.keras.preprocessing.sequence.pad_sequences(temp_char,dtype='int32', maxlen=20, padding="post", value=0)

    word_context.append(temp_word)
    char_context.append(temp_char)

    # formatting question
    temp_word = []
    temp_char = []
    oov_terms = check_OOV_terms(emb_model, t['question_list'])

    for word in t['question_list']:
      if word in oov_terms:
        temp_word.append(np.zeros((100,)))
        unk_word[word] = unk_word.get(word,0)+1
      else:
        temp_word.append(emb_model.get_vector(word))
      temp_char.append(charVocab.toidx(word))

    temp_char = tf.keras.preprocessing.sequence.pad_sequences(temp_char, dtype='int32', maxlen=20, padding="post", value=0)

    word_question.append(temp_word)
    char_question.append(temp_char)

  word_context = tf.keras.preprocessing.sequence.pad_sequences(word_context, padding="post", dtype='float32', value=0.0)
  word_question = tf.keras.preprocessing.sequence.pad_sequences(word_question, padding="post", dtype='float32', value=0.0)
  char_context =  tf.keras.preprocessing.sequence.pad_sequences(char_context, padding="post", dtype='float32', value=0.0)
  char_question = tf.keras.preprocessing.sequence.pad_sequences(char_question, padding="post", dtype='float32', value=0.0)

  return  word_question, char_question, word_context, char_context

def check_OOV_terms(emb_model, word_listing):
    oov_list = []
    for w in word_listing:
      if w not in emb_model.key_to_index.keys():
        oov_list.append(w)
    return oov_list
