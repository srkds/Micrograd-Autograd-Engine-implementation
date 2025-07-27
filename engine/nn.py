from .micrograd import *

class Linear:
  def __init__(self, in_dims, out_dims):
    """W : (layers, inputs)
        b: (layers, 1)
    """
    np.random.seed(1)
    self.W = Value(np.random.randn(out_dims, in_dims) / np.sqrt(in_dims) )
    self.b = Value(np.zeros((out_dims, 1)))
    self.Z = None

  def forward(self, X):
    # print("W: ", self.W.data.shape)
    # print("b: ", self.b.data.shape)
    # print("X: ", X.data.shape)
    assert self.W.data.shape[1] == X.data.shape[0]
    # assert (self.W.data.shape[0], X.data.shape[1]) == self.b.data.shape
    self.Z = self.W.dot(X) + self.b
    return self.Z

  def __call__(self, X):
    return self.forward(X)

