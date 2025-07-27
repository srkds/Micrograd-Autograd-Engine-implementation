from engine.nn import *

class Mymodel():
    def __init__(self, layer_dims=[3,5,3,1]):
        self.l1 = Linear(layer_dims[0], layer_dims[1])
        self.l2 = Linear(layer_dims[1], layer_dims[2])
        self.l3 = Linear(layer_dims[2], layer_dims[3])
        self.l4 = Linear(layer_dims[3], layer_dims[4])
        self.params = [self.l1.W, self.l1.b, self.l2.W, self.l2.b, self.l3.W, self.l3.b, self.l4.W, self.l4.b]

    def forward(self, X):

        o1 = self.l1(X).ReLU()
        o2 = self.l2(o1).ReLU()
        o3 = self.l3(o2).ReLU()
        o4 = self.l4(o3).Sigmoid()

        return o4

    def optimize(self, lr=0.01):
        for param in self.params:
            param.data = param.data - (param.grad*lr)

        # zero grad
        for param in self.params:
            param.grad[:] = 0
            
    def __call__(self, X):
        return self.forward(X)