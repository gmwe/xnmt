from __future__ import print_function

import dynet as dy
import segment_transducer
import linear
import six
import numpy
import expression_sequence

from enum import Enum
from model import HierarchicalModel
from decorators import recursive, recursive_assign
from reports import HTMLReportable

class SegmentingEncoderBuilder(HierarchicalModel, HTMLReportable):
  class SegmentingAction(Enum):
    READ = 0
    SEGMENT = 1
    DELETE = 2

  def __init__(self, embed_encoder=None, segment_transducer=None, learn_segmentation=True, model=None):
    super(SegmentingEncoderBuilder, self).__init__()
    # The Embed Encoder transduces the embedding vectors to a sequence of vector
    self.embed_encoder = embed_encoder
    self.P0 = model.add_parameters(segment_transducer.encoder.hidden_dim)
    self.learn_segmentation = learn_segmentation

    # The Segment Encoder decides whether to segment or not
    self.segment_transform = linear.Linear(embed_encoder.hidden_dim, len(self.SegmentingAction), model)

    # The Segment transducer predict a category based on the collected vector
    self.segment_transducer = segment_transducer

    self.train = True
    self.register_hier_child(segment_transducer)

  @recursive
  def set_train(self, train):
    self.train = train

  def set_html_input(self, *inputs):
    print("set_html_input", *inputs)

  def transduce(self, embed_sent):
    src = embed_sent
    num_batch = src[0].dim()[1]
    P0 = dy.parameter(self.P0)
    # Softmax + segment decision
    encodings = self.embed_encoder.transduce(embed_sent)
    if self.learn_segmentation:
      segment_logsoftmaxes = [dy.log_softmax(self.segment_transform(fb)) for fb in encodings]
      # Segment decision
      if self.train:
        segment_decisions = [log_softmax.tensor_value().categorical_sample_log_prob().as_numpy()[0] for log_softmax in segment_logsoftmaxes]
      else:
        segment_decisions = [log_softmax.tensor_value().argmax().as_numpy().transpose() for log_softmax in segment_logsoftmaxes]
    else:
      # TODO(philip30): Implement the reader for segment decision!
      segment_decision = embed_sent.segment_decision
    # Some checks
    assert len(encodings) == len(segment_decisions), \
           "Encoding={}, segment={}".format(len(encodings), len(segment_decisions))
    # The last segment decision should be equal to 1
    if len(segment_decisions) > 0:
      segment_decisions[-1] = numpy.ones(segment_decisions[-1].shape, dtype=int)
    # Buffer for output
    buffers = [[] for _ in range(num_batch)]
    outputs = [[] for _ in range(num_batch)]
    self.segment_transducer.set_input_size(num_batch, len(encodings))
    # Loop through all the frames (word / item) in input.
    for j, (encoding, segment_decision) in enumerate(six.moves.zip(encodings, segment_decisions)):
      # For each decision in the batch
      for i, decision in enumerate(segment_decision):
        # Get the particular encoding for that batch item
        encoding_i = dy.pick_batch_elem(encoding, i)
        # Append the encoding for this item to the buffer
        buffers[i].append(encoding_i)
        # If segment for this particular input
        decision = int(decision)
        if decision == self.SegmentingAction.SEGMENT.value:
          expr_seq = expression_sequence.ExpressionSequence(expr_list=buffers[i])
          transduce_output = self.segment_transducer.transduce(expr_seq)
          outputs[i].append(transduce_output)
          buffers[i] = []
        elif decision == self.SegmentingAction.DELETE.value:
          buffers[i] = []
        self.segment_transducer.next_item()
    # Padding
    max_col = max(len(xs) for xs in outputs)
    def pad(xs):
      deficit = max_col - len(xs)
      if deficit > 0:
        xs.extend([P0 for _ in range(deficit)])
      return xs
    outputs = dy.concatenate_to_batch(list(six.moves.map(lambda xs: dy.concatenate_cols(pad(xs)), outputs)))
    # Packing output together
    if self.train and self.learn_segmentation:
      self.segment_decisions = segment_decisions
      self.segment_logsoftmaxes = segment_logsoftmaxes
    # Return the encoded batch by the size of [(encode,segment)] * batch_size
    return outputs

  def calc_reinforce_loss(self, reward, lmbd):
    if self.learn_segmentation:
      segment_logprob = None
      for log_softmax, segment_decision in six.moves.zip(self.segment_logsoftmaxes, self.segment_decisions):
        ll = dy.pick_batch(log_softmax, segment_decision)
        if not segment_logprob:
          segment_logprob = ll
        else:
          segment_logprob += ll
      return (segment_logprob + self.segment_transducer.disc_ll()) * reward * lmbd
    else:
      return None

