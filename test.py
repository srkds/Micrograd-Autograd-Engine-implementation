import pytest
from micrograd import Value

def test_add():
    r = Value(1.0) + Value(2.0)
    assert r.data == 3.0
    r = 10.0 + Value(2.0)
    assert r.data == 12.0

def test_mul():
    r = Value(3.0) * Value(2.0)
    assert  r.data == 6.0
    r = 10.0 * Value(2.0)
    assert r.data == 20.0

def test_tanh():
    r = Value(10.0)
    t = r.tanh()
    assert t.data == 0.9999999958776927
