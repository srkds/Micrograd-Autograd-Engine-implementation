import numpy as np
import torch

class Value:
  def __init__(self, data, _children=(), _op=''):
    self.data = np.array(data, dtype=np.float32)
    self.grad = np.zeros_like(self.data)
    self._backward = lambda: None
    self._prev = set(_children)
    self._op = _op

  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    other.data = np.broadcast_to(other.data, self.data.shape)
    other.grad = np.zeros_like(other.data)
    out = Value(self.data + other.data, (self, other), "+")

    def _backward():
      self.grad += out.grad
      other.grad += out.grad

    out._backward = _backward
    return out

  def __neg__(self): # -self
    return self.data * -1
            
  def __sub__(self, other): # self - other
    return self + (-other)
  
  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    other.data = np.broadcast_to(other.data, self.data.shape)
    other.grad = np.zeros_like(other.data)
    out = Value(self.data*other.data, (self, other), "*")

    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad

    out._backward = _backward
    return out

  def dot(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(np.dot(self.data, other.data), (self, other), "dot")

    def _backward():
      self.grad += out.grad @ other.data.T
      # print("Grad: ", self.grad)
      other.grad += self.data.T @ out.grad
    out._backward = _backward
    return out

  def __repr__(self):
    return f"Value: {self.data}, Grad: {self.grad}"

  def sum(self):
    if not isinstance(self, Value):
      self = Value(self)
    out = Value(self.data.sum(), (self,), "sum")

    def _backward():
      self.grad += out.grad
    out._backward = _backward
    # print(self)
    return out

  def BCE(self, other):
    """
    Accepts (y, yh)
    TODO: atomic operation for divide, exp, log, etc
    """
    if not isinstance(self, Value):
      self = Value(self)
    if not isinstance(other, Value):
      other = Value(other)
    
    eps = 1e-8
    m = self.data.shape[1] # expecting (1,m) where m is no of training examples
    out = -np.sum(np.multiply(self.data, np.log(other.data + eps)) + np.multiply((1-self.data), np.log(1-other.data + eps))) / m
    out = Value(out, (self, other), "BCE")

    def _backward():
      # https://www.python-unleashed.com/post/derivation-of-the-binary-cross-entropy-loss-gradient
      # dAL = -(np.divide(Y, AL) - np.divide(1-Y, 1-AL))
      dyh = -(np.divide(self.data, other.data + eps) - np.divide(1-self.data, 1-other.data + eps))
      other.grad[:] += dyh
      # print(dyh)
    out._backward = _backward

    return out
    
    # m = Y.shape[1]
    # J = -np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1-Y), np.log(1-AL))) / m

  def ReLU(self):
    self = self if isinstance(self, Value) else Value(self)
    out = np.maximum(0, self.data)

    out = Value(out, (self,), "ReLU")

    def _backward():
      self.grad += out.grad * (self.data > 0)
      # mask = self.data <= 0
      # self.grad += np.array(out.grad, copy=True)
      # self.grad[mask] += 0
    out._backward = _backward
    return out

  def Sigmoid(self):
    self = self if isinstance(self, Value) else Value(self)
    out = 1 / (1 + np.exp(-self.data))
    # print(-self.data)
    out = Value(out, (self, ), "Sigmoid")

    def _backward():
      self.grad += out.data * (1 - out.data) * out.grad
    out._backward = _backward

    return out

  def backward(self):
    # DAG
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)

    if self.grad.ndim > 0:
       self.grad[:] = 1 
    else:
       self.grad = np.array(1) 
    for v in reversed(topo):
      v._backward()
      
if __name__ == "__main__":
    a = Value([3,5,6])
    o = a + 3 # support of broadcast
    print(o)

    b = Value([2, 1, 3])
    o1 = a * b # support of vectorization
    print(o1)

    # Validating Gradient
    x = Value([[-4.0, 2.]])
    w = Value([[.2],[1.]])
    c = Value([[1]])
    z = x.dot(w)
    d = z.dot(c)
    h = d + 2
    y = h
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([[-4.0, 2.]])
    w = torch.Tensor([[.2],[1.]])
    c = torch.Tensor([[1.]])
    x.requires_grad = True
    w.requires_grad = True
    c.requires_grad = True
    z = x@w
    d = z@c
    h = d + 2
    y = h
    y.backward()
    xpt, ypt = x, y

    # forward pass went well
    print(ymg.data, ypt.data.item())
    print(xmg.grad, xpt.grad)
    assert ymg.data == ypt.data.item()
    # backward pass went well
    assert xmg.grad[0][0] == xpt.grad[0][0]
    assert xmg.grad[0][1] == xpt.grad[0][1]