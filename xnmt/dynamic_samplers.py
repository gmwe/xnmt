from xnmt.persistence import serializable_init, Serializable
import numpy as np



class DynamicSampler(Serializable):
  """
    Abstract dynamic sampler class

    https://arxiv.org/pdf/1805.00178.pdf
  """
  yaml_tag = '!DynamicSampler'

  @serializable_init
  def __init__(self):
    self._current_dict_idx = 0
    self.loss = [{},{}]
    self.diff = {}
    self.criterion = {}
    self.criterion_sum = 0
    self.max_diff = 0
    self.min_diff = 0

  @property
  def current_loss(self):
    return self.loss[self._current_dict_idx % 2]

  @property
  def previous_loss(self):
    return self.loss[(self._current_dict_idx + 1) % 2]

  @staticmethod
  def _access_by_objects(src, trg):
    # distinction needed for different input types
    if type(src[0]) == np.ndarray:
      tmp_src = str(tuple([list(e) for e in src]))
    else:
      tmp_src = str(tuple([e for e in src]))

    tmp_trg = str(tuple([e for e in trg]))
    return tmp_src, tmp_trg

  @staticmethod
  def _access(src, trg):
    return str(src), str(trg)

  def update_loss(self, src, trg, loss):
    self.current_loss[self._access(src, trg)] = loss

  def sample(self):
    raise NotImplementedError()

  def _next_epoch_special(self):
    pass

  def next_epoch(self):
    """ Update the criterion values after an epoch.
    """
    for k in self.current_loss.keys():
      if k not in self.previous_loss:
        continue
      self.diff[k] = (self.previous_loss[k] - self.current_loss[k]) \
                     / self.previous_loss[k]
    if len(self.diff) > 0:
      self.max_diff = max(self.diff.values())
      self.min_diff = min(self.diff.values())

    self.criterion_sum = 0
    for k in self.current_loss.keys():
      if k not in self.diff:
        continue
      self.criterion[k] = (self.diff[k] - self.min_diff) \
                          / (self.max_diff - self.min_diff)
      self.criterion_sum += self.criterion[k]

    self._next_epoch_special()
    self._current_dict_idx += 1



class WeightedSamplingDS(DynamicSampler):
  """ Performs weighted sampling as in
      https://arxiv.org/pdf/1805.00178.pdf
  """
  yaml_tag = '!WeightedSampling'

  @serializable_init
  def __init__(self, sampling_size=0.8):
    """

    Args:
      sampling_size: percentage of sentences sampled compared to full epoch
    """
    super().__init__()
    self.weight = {}
    self.sampling_size = sampling_size

  def sample(self):
    return self._sample_weighted()

  def _next_epoch_special(self):
    for k in self.current_loss.keys():
      if k not in self.criterion:
        continue
      self.weight[k] = self.criterion[k] / self.criterion_sum
    pass

  def _sample_weighted(self):
    """Perform a weighted sampling without any replacement.
    Returns:
      The sampled population where each entry is a data key.
    """
    population=range(len(self.weight))
    k=int(self.sampling_size * len(self.weight))
    w=list(self.weight.values())
    sample_population_idx = (np.random.choice(population,k, replace=False,p=w))
    weight_keys = list(self.weight.keys())
    sample_population = [weight_keys[i] for i in sample_population_idx]
    return sample_population


class ReviewMechanismDS(DynamicSampler):
  """ Performs sampling according to the review mechanism in
      https://arxiv.org/pdf/1805.00178.pdf
  """
  yaml_tag = '!ReviewMechanism'

  @serializable_init
  def __init__(self, sampling_size=0.8, low_sampling_size=0.1):
    """

    Args:
      sampling_size: percentage of dhigh entries with highest criterion that are kept after an epoch
      low_sampling_size: percentage of dlow entries that are sampled each epoch
    """
    super().__init__()
    self.dlow = []
    self.dhigh = None
    self.dhigh_sampling_size = sampling_size
    self.dlow_sampling_size = low_sampling_size

  def sample(self):
    if self.dhigh is None:
      self.dhigh = self.current_loss.keys()

    n=int(len(self.dhigh) * self.dhigh_sampling_size)

    # select top n
    dhigh_crit = {k: self.criterion[k] for k in self.dhigh}
    highest = sorted([(v, k) for k,v in dhigh_crit.items()],
                            reverse=True)
    self.dhigh = [k for l,k in highest[:n]]
    self.dlow = self.dlow + [k for l,k in highest[n:]]

    sample_dlow_idx = (np.random.choice(len(self.dlow),
                                        int(len(self.dlow) * self.dlow_sampling_size),
                                        replace=False))

    dlow_sample = [self.dlow[i] for i in sample_dlow_idx]

    return self.dhigh + dlow_sample
