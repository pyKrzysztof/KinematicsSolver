from math import pi
import sympy as sm
from sympy import cos, sin, tan, acos, asin, atan
import numpy as np
import itertools
import matplotlib.pyplot as plt



class Range:

    def __init__(self, name: str, default: float, v1: float, v2: float, as_deg=False):
        if as_deg:
            default = Range._to_rad(default)
            v1 = Range._to_rad(v1)
            v2 = Range._to_rad(v2)
        self.name = name
        self.value = default
        self.minimum = v1
        self.maximum = v2

    def in_range(self, value, as_deg=False):
        if as_deg:
            value = Range._to_rad(value)
        return (value >= self.minimum) and (value <= self.maximum)

    def get_range(self, n):
        return np.linspace(self.minimum, self.maximum, n)

    def __str__(self):
        return f"<{self.minimum}; {self.maximum}>, default={self.value}"

    @staticmethod
    def _to_rad(value):
        return value * pi / 180


class Ranges:

    def __init__(self, *ranges):
        self.ranges = ranges
        self.values = [[r[0], None] for r in self.ranges]
        self.names = {}
        for idx, r in enumerate(ranges):
            self.names[r[1].name] = idx

    def get_symbols(self):
        r_symbols = [r[0] for r in self.ranges]
        # print(r_symbols)
        return r_symbols

    def __setitem__(self, key, value, as_deg=False):
        if isinstance(key, int):
            idx = key
        else:
            if key not in self.names:
                raise Exception(f"Key {key} is not in the range group.")
            idx = self.names[key]

        r = self.ranges[idx][1]
        if r.in_range(value, as_deg):
            self.values[idx][1] = value
        else:
            raise Exception(f"Value {value} out of range: <{r.minimum}, {r.maximum}>.")

    def __len__(self):
        return len(self.ranges)


class Result:

    def __init__(self, matrix):
        self.matrix = matrix

    @property
    def T(self):
        return self.matrix[3], self.matrix[7], self.matrix[11]

    def __str__(self):
        return repr(self.matrix)


class Link:

    # TRANSFORM = "M( " \
    #             f"[cos(self.tz),-1*sin(self.tz)*cos(self.tx),sin(self.tz)*sin(self.tx),self.x*cos(self.tz)]," \
    #             f"[sin(self.tz),cos(self.tz)*cos(self.tx),-1*cos(self.tz)*sin(self.tx),self.x*sin(self.tz)]," \
    #             f"[0, sin(self.tx), cos(self.tx), self.d], [0, 0, 0, 1] )"

    def __init__(self, data):
        self.data = data[0]  # [d, x, range(tx), range(tz)]
        self.idx = data[1]
        self.symbols = sm.symbols(f"d{self.idx}, x{self.idx}, thetaX{self.idx}, thetaZ{self.idx}")
        self.matrix = None

    def get_transform(self):
        d, x, tx, tz = self.symbols
        if not self.matrix:
            self.matrix = sm.Matrix( [[cos(tz), -sin(tz)*cos(tx), sin(tz)*sin(tx), x*cos(tz)],
                            [sin(tz), cos(tz)*cos(tx), -cos(tz)*sin(tx), x*sin(tz)],
                            [0, sin(tx), cos(tx), d], [0, 0, 0, 1]] )
        return self.matrix

    def evaluate(self, d=None, x=None, tx=None, tz=None):
        params = [d, x, tx, tz]
        subs = [ (sym, val) for sym, val in zip(self.symbols, params) if val is not None]
        return self.matrix.subs(subs)


class Solver:

    def __init__(self):
        self.links = []
        self.nlinks = 0
        self.initialized = False
        self.transform = sm.Matrix()
        self.matching_matrix = sm.Matrix()
        self._forward = (None, None)
        self._inverse = (None, None)

    def initialize(self):
        self.initialized = True
        self.get_transform_matrix()
        self.forward()
        # self.inverse()
        return self._forward[0], self._forward[1], self._inverse[0], self._inverse[1]

    def add_links(self, *parameters: list):
        for link in parameters:
            self.links.append(Link( [link, self.nlinks] ))
            self.nlinks += 1

    def get_transform_matrix(self):
        matrix = sm.eye(4)
        for link in self.links:
            matrix = matrix*link.get_transform()
        self.transform = matrix
        return self.transform

    def forward(self):
        matching = [None] * len(self.transform.free_symbols)
        ranges = [None] * len(self.transform.free_symbols)
        for link in self.links:
            for symbol in link.symbols:
                idx = list(self.transform.free_symbols).index(symbol)
                data = link.data[link.symbols.index(symbol)]
                if isinstance(data, Range):
                    ranges[idx] = list([list(self.transform.free_symbols)[idx], data])
                    continue
                matching[idx] = (list(self.transform.free_symbols)[idx], data)

        matching = [pair for pair in matching if pair]
        ranges = [pair for pair in ranges if pair]

        matrix = self.transform.subs(matching)
        self.matching_matrix = matrix

        def _forward_calculate(range_values: Ranges):
            if not len(ranges) == len(range_values):
                raise Exception("Wrong amount of range parameters.")
            return Result(matrix.subs(range_values.values))

        self._forward = (_forward_calculate, Ranges(*ranges))

    def inverse(self, ranged_symbols=None, h_transform=None):
        if not h_transform:
            h_transform = self.transform
        if not ranged_symbols:
            ranged_symbols = [link.symbols for link in self.links]
            print(ranged_symbols)

        a = sm.ones(3, self.nlinks)
        b = sm.ones(3, self.nlinks)
        column = h_transform[:-1, -1]
        links_transform = [link.get_transform() for link in self.links]
        for idx_v, value in enumerate(column):
            # print(idx_v, value)
            for idx_s, sym in enumerate(ranged_symbols):
                # print(idx_s, sym)
                a[idx_v, idx_s] = sm.diff(value, sym)
                b[:, idx_s] = links_transform[idx_s][:-1, 2]
        self._inverse = (sm.Matrix.vstack(a, b), ranged_symbols)
        return self._inverse[0], self._inverse[1]

    def working_area(self, n):
        if not self.initialized:
            raise Exception("Initialize the system before trying to get the working area.")

        f, fr = self._forward

        X, Y, Z = [], [], []
        permutations = list(itertools.product( *[r[1].get_range(n) for r in fr.ranges] ))

        for group in permutations:
            for idx in range(len(group)):
                fr[idx] = group[idx]
            x, y, z = f(fr).T
            X.append(x)
            Y.append(y)
            Z.append(z)

        plt.scatter(X, Y)
        plt.show()

        return X, Y, Z

    def pack(self):
        if not self.initialized:
            self.initialize()
        data = []
        data[0] = self.matching_matrix
