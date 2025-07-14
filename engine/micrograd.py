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
      print("Grad: ", self.grad)
      other.grad += self.data.T @ out.grad
    out._backward = _backward
    return out

  def __repr__(self):
    return f"Value: {self.data}, Grad: {self.grad}"

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

    self.grad[:] = 1
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