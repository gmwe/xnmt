from __future__ import division, generators

from batcher import *

class Embedder:
  '''
  A template class to embed a word or token.
  '''

  '''
  Takes a string or word ID and returns its embedding.
  '''

  def embed(self, x):
    raise NotImplementedError('embed must be implemented in Embedder subclasses')


class SimpleWordEmbedder(Embedder):
  'Simple word embeddings'

  def __init__(self, vocab_size, emb_dim, model):
    self.vocab_size = vocab_size
    self.emb_dim = emb_dim
    self.embeddings = model.add_lookup_parameters((vocab_size, emb_dim))
    self.serialize_params = [vocab_size, emb_dim, model]

  def embed(self, x):
    # single mode
    if not Batcher.is_batch_word(x):
      return self.embeddings[x]
    # minibatch mode
    else:
      return self.embeddings.batch(x)

  def embed_sentence(self, sentence):
    # single mode
    if not Batcher.is_batch_sentence(sentence):
      embeddings = [self.embed(word) for word in sentence]
    # minibatch mode
    else:
      embeddings = []
      for word_i in range(len(sentence[0])):
        embeddings.append(self.embed([single_sentence[word_i] for single_sentence in sentence]))

    return embeddings

class FeatVecNoopEmbedder(Embedder):
  def __init__(self, emb_dim, model):
    self.emb_dim = emb_dim
    self.serialize_params = [emb_dim, model]

  def embed(self, x):
    # single mode
    if not Batcher.is_batch_word(x):
      return dy.inputVector(x)
    # minibatch mode
    else:
      if len(x[0]) != self.emb_dim:
        raise RuntimeError("Input vec dim {} does not match embed_dim {}"\
                           .format(len(x[0]), self.emb_dim))
      # workaround from https://github.com/clab/dynet/issues/175
      concat_x = np.concatenate(x)
      return dy.reshape(
        dy.inputVector(concat_x),
        (len(x[0]),), batch_size=len(x))
#      return [inputVector(x_i) for x_i in x]
#      return self.embeddings.batch(x)

  def embed_sentence(self, sentence):
    # single mode
    if not Batcher.is_batch_sentence(sentence):
      embeddings = [self.embed(word) for word in sentence]
    # minibatch mode
    else:
      embeddings = []
      for word_i in range(len(sentence[0])):
        embeddings.append(self.embed([single_sentence[word_i] for single_sentence in sentence]))

    return embeddings

