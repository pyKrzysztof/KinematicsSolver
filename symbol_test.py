import math
from lib.matrixlib import Matrix as _M


class M(_M):

    def __str__(self):
        str_array = [[value for value in row] for row in self.array]
        # "List comprehension", syntax w pythonie pozwalający tworzenie nowej listy na bazie innej listy
        # pozwala na zagnieżdżanie, ale dla czytelności czasami z niego rezygnuje

        matrix_str = "\n".join([str(row) for row in str_array])
        return f"{self.size[0]}x{self.size[1]}:\n" + matrix_str + "\n"

def cos(val):
    try:
        return math.cos(val)
    except TypeError:
        s = Symbol(f"math.cos({val})")
        s.top_functions.append(math.cos)
        return s

def sin(val):
    try:
        return math.sin(val)
    except TypeError:
        return Symbol(f"math.sin({val})")


def tan(val):
    try:
        return math.tan(val)
    except TypeError:
        return Symbol(f"math.tan({val})")


def acos(val):
    try:
        return math.acos(val)
    except TypeError:
        return Symbol(f"math.acos({val})")


def asin(val):
    try:
        return math.asin(val)
    except TypeError:
        return Symbol(f"math.asin({val})")


def atan(val):
    try:
        return math.atan(val)
    except TypeError:
        return Symbol(f"math.atan({val})")


class Symbol:

    def __init__(self, name, value=None):
        self.name = name
        self.value = value
        self.operations = []
        self.linked = []
        self.top_functions = []

    def __float__(self):
        if self.value is not None:
            return float(self.value)
        return self.name

    def __int__(self):
        if self.value is not None:
            return int(self.__float__())
        return self.name

    def __add__(self, other):
        new = Symbol(f"{self.name}+{other}")
        new.value = self.value
        new.operations = self.operations
        if isinstance(other, int) or isinstance(other, float):
            if new.value is not None:
                new.value = new.value + other
                return new
            else:
                new.operations.append(f"(+{other})")
                return new
        if isinstance(other, Symbol):
            new.linked.append({"+": other})
            new.name = f"{self.name}+{other.name}"
            return self

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Symbol):
            new = Symbol(f"{self.name}-{other.name}", self.value)
            new.operations = self.operations
            new.linked.append({"-": other})
            return new
        return self.__add__(-1*other)

    def __round__(self, n=None):
        if self.value is not None:
            self.value = round(self.value, ndigits=M.PRECISION)
        return self

    # def __rsub__(self, other):
    #     if isinstance(other, Symbol):
    #         self.linked.append({"-": other})
    #
    #     pass  # operations = other - ({value}{operations})

    def __mul__(self, other):
        new = Symbol(f"{other}*{self.name}")
        new.operations = self.operations
        new.value = self.value
        if isinstance(other, int) or isinstance(other, float):
            if new.value is not None:
                new.value = new.value * other
                return new
            else:
                new.operations.append(f"*{other}")
                return new
        if isinstance(other, Symbol):
            new.linked.append({"*": other})
            new.operations.append(f"*{other}")
            new.name = f"{self.name}*{other.name}"
            return new

    def __rmul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return self.__mul__(other)

    def evaluate(self):
        if self.value is None:
            return self
        string = "".join(self.operations)
        if string == "":
            for f in self.top_functions:
                self.value = f(self.value)
            return self.value
        self.value = eval(f"{self.value}string")
        for f in self.top_functions:
            self.value = f(self.value)
        return self.value

    def set(self, value):
        self.value = value

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name