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
