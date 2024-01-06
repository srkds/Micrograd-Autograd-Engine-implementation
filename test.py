from micrograd import Value

a = Value(2.0)
b = Value(3.0)
print('a is ', a)
print('b is ', b)

c = a+b; c.label='c'
d = a*b; d.label='d'
print(c.label, c, c._prev, c._op)
print(d.label, d, d._prev, d._op)
