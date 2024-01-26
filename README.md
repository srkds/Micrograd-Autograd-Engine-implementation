# Micrograd Autograd Engine Implementation

<img src="./assets/MLP2.png" width="700">

When you call `loss.backward()`, what happens behind the scenes? The model generates a directed acyclic graph of each forward pass for a single epoch. For each input training example fed to the network, it will generate a directed acyclic graph of the forward pass. Then, we calculate its loss; the loss node will be added on top of this DAG. When you call `loss.backward()`, it first generates a reversed topological order list of all the nodes in the DAG. Then, it starts calling each node (object's) `_backward()` (private) function that calculates the local gradient and multiplies it with the forward node's gradient, which is added to the current node's gradient. This process is done for all elements in the reversed topological list. Finally, it updates each parameter based on gradient values by passing all parameters to the optimizer using `model.parameters()`.

This is how backpropagation works when you call `loss.backward()` and then call `optimizer.step()`. That's the Autograd engine, the backbone of modern neural network libraries. Here is the link to [PyTorch's Autograd engine](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html).

Micrograd is a replica of the Autograd engine, containing the `Value` class that supports all these features. It was originally developed by [Andrej Karpathy ♥](https://github.com/karpathy/micrograd).

### Learnings

- How the gradients of loss with respect to each parameter of neural nets are generated.
- Generates a DAG, then a reversed topological order, and then calculates each parameter's gradients.
- Getting all parameters of the model using `.parameters()` and updating their values.

# Structure of Value Class

Can perform following operations

```py
>>> Value(1.0) + Value(2.0)
Value(3.0)

>>> Value(2.0) * Value(3.0)
Value(6.0)

>>> Value(2.9).tanh()
Value()
```

## Value Class

## Attributes

`data`

- Data of the object.

`grad`

- Contains the gradient.
- By default its 0.0

## Functions

`__rper__`

- Print formatted data of the `Value` object instead of object address.

### Operations

- add
- multiply

#### Negative (-a)

```py
   def __neg__(self):
        return self * -1
```

Negating the value.

eg

```py
a # 10
(-a) #-10
```

#### Subtraction (a-b)

```py
   def __sub__(self, other):
        return self + (-other) # (-other) is the negative operation
```

Reusing the `addition` operation by passing other value as its oposite value by negating `(-other)` to get subtracted value. And there's no need of differentiation method `._backward()` beacause we are reusing the `+` operation and it has `._backward()`.

Eg:

```py
a = 10
b = 5

a + (-b) # 5
a - b    # 5

```
