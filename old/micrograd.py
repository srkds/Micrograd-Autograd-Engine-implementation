import math
import numpy as np
import matplotlib.pyplot as plt

class Value():

    def __init__(self, data, _children=(), _op='', label=''):
        """
        data: data of the object
        grad: gradient of loss WRT current data
        _backward: function that computes gradient. dLoss/dCurrentNode. Initially it will be None.
        _prev: set of children nodes
        _op: operation
        label: variable name/label
        """
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f'Value(data={self.data})'

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out
    
    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other),'*')

        def _backward():
            self.grad += other.data * out.grad 
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
      assert isinstance(other, (float, int)), "Allowing other as only int or float"
      out = Value(self.data**other, (self, ), f"**{other}")

      def _backward():
        self.grad += (other * self.data**(other-1)) * out.grad
      out._backward = _backward
      return out

    def __rmul__(self, other):
        return self * other

    def tanh(self):
        e = math.e**(2*self.data)
        out = Value((e-1) / (e+1), (self,),'tanh')
            
        def _backward():
            self.grad += (1 - out.data**2) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        # (-other) overriding negative operation
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp') # exp function / forward pass

        def _backward():
            self.grad += out.data * out.grad # derivation of exp function multiply with global gradient
        out._backward = _backward 
        return out

    def backward(self):
        # Running Topological sort
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0

        # Call each node's _backward() function in reverse topological order to calculate gradient
        for node in reversed(topo):
            node._backward()
    