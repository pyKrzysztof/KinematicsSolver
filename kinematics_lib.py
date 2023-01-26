import itertools
import pickle
import matplotlib.pyplot as plt
import sympy as sm
from sympy import sin, cos
import numpy as np
import random
from math import pi


class Link:

    def __init__(self, data):
        self.data = data[0]  # [d, x, range(a), range(tz)]
        self.idx = data[1]
        self.symbols = sm.symbols(f"d{self.idx}, x{self.idx}, a{self.idx}, theta{self.idx}")
        self.matrix = None

    def get_transform(self):
        d, x, a, tz = self.symbols
        if not self.matrix:
            self.matrix = sm.Matrix( [[cos(tz), -sin(tz)*cos(a), sin(tz)*sin(a), x*cos(tz)], [sin(tz), cos(tz)*cos(a), -cos(tz)*sin(a), x*sin(tz)], [0, sin(a), cos(a), d], [0, 0, 0, 1]] )
        return self.matrix


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

    def __str__(self):
        return f"<{self.minimum}; {self.maximum}>, default={self.value}"

    @staticmethod
    def _to_rad(value):
        return value * pi / 180


class Kinematics:

    def __init__(self, filename, dimensions=2, overwrite=False):
        self.filename = filename
        if filename is not None and not overwrite:
            try:
                self.open(filename)
                return
            except FileNotFoundError:
                pass

        self.matching_matrix = None # transform matrix with constant values matched to the equations.
        self._symbols = None # a dictionary mapping string name of a symbol to it's symbol object, mostly for referencing its value ranges.
        self._ranges = None  # a list of value ranges for symbols in the form of [[name, minimum, value, maximum], ]
        self._link_transforms = None # a list of transform matrices from each link with matched constant values.
        self._dimensions = dimensions # 2D or 3D environment
        self._working_area = None # stores the results calculated by get_working_area
        self._jacobian = None # stores the calculated jacobian matrix
        self._inverse_jacobian = None # stores the calculated inverse jacobian matrix
        self._match_then_inv_jacobian_flag = False # flag whether to match unknown values to jacobian first and then calculate it's inverse. Saves time.

    def _get_permutations(self, n):
        return list(itertools.product(*[np.linspace(r[1], r[3], n) for r in self._ranges]))

    def open(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        self.matching_matrix = data[0]
        self._symbols = data[1]
        self._ranges = data[2]
        self._link_transforms = data[3]
        self._dimensions = data[4]
        self._working_area = data[5]
        self._jacobian = data[6]
        self._inverse_jacobian = data[7]
        self._match_then_inv_jacobian_flag = data[8]

    def close(self):
        data = [self.matching_matrix,
                self._symbols,
                self._ranges,
                self._link_transforms,
                self._dimensions,
                self._working_area,
                self._jacobian,
                self._inverse_jacobian,
                self._match_then_inv_jacobian_flag]

        with open(self.filename, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def set_links(self, *parameters: list):
        """
        list of DH Parameters: [D, X, a, Tz] where
        D = link offset,
        X = link length,
        a = Either value or a Range object for the Z axis twist starting from the PREVIOUS link,
        Tz = Either value or a Range object for the angle between previous and current coordinate anchors on the same plane XY
        """
        links = []
        matrix = sm.eye(4)
        for n, params in enumerate(parameters):
            link = Link( [params, n] )
            matrix = matrix * link.get_transform()
            links.append(link)

        free_symbols = matrix.free_symbols
        matching = []
        ranges = []
        for link in links:
            for symbol in link.symbols:
                idx = list(free_symbols).index(symbol)
                data = link.data[link.symbols.index(symbol)]
                if isinstance(data, Range):
                    ranges.append(list([list(free_symbols)[idx], data]))
                    continue
                matching.append((list(free_symbols)[idx], data))

        self.matching_matrix = matrix.subs(matching)
        self._link_transforms = [link.get_transform().subs(matching) for link in links]
        self._symbols = dict(zip([r[1].name for r in ranges], [r[0] for r in ranges]))
        self._ranges = [[r[1].name, r[1].minimum, r[1].value, r[1].maximum] for r in ranges]

    def forward(self, values):
        m = self.matching_matrix.subs(zip(self._symbols.values(), values))
        return m[3], m[7], m[9]

    def get_working_area(self, n=50, plot=False):
        if self._working_area is not None and self._working_area[0] == n:
            if plot and self._dimensions == 2:
                return self.plot2d(*self._working_area[1])
            if plot:
                return self.plot3d(*self._working_area[1])
            return self._working_area

        permutations = self._get_permutations(n)
        X, Y = [], []
        if self._dimensions == 2:
            for values in permutations:
                x, y, _ = self.forward(values)
                X.append(x)
                Y.append(y)
            self._working_area = [n, [X, Y]]
            if plot:
                return self.plot2d(X, Y)
            return X, Y

        Z = []
        for values in permutations:
            x, y, z = self.forward(values)
            X.append(x)
            Y.append(y)
            Z.append(z)
        self._working_area = [n, [X, Y, Z]]
        if plot:
            return self.plot3d(X, Y, Z)
        return X, Y, Z

    def calculate_jacobian(self):
        # TODO: categorize operations by joint types. currently assumes everything is revolute joint.
        # self._link_transforms[0][:3, :3] # Rot matrix
        # self._link_transforms[0][:3, 3:] # T matrix
        z_unit = sm.Matrix([0, 0, 1])
        final_tr = self.matching_matrix[:3, 3:]
        columns = []
        for link in self._link_transforms:
            link_rot = link[:3, :3]
            link_tr = link[:3, 3:]

            a = link_rot * z_unit
            columns.append(sm.Matrix.vstack(a.cross((final_tr - link_tr).transpose()), a))

        self._jacobian = sm.Matrix.hstack(*columns)

        if self._dimensions == 2:
            temp: sm.Matrix = self._jacobian[[0, 1], :]
            while temp.shape[0] != temp.shape[1]:
                temp = temp.row_insert(-1, sm.Matrix([[1]*temp.shape[1]]))
            self._inverse_jacobian = temp.inv()
        else:
            temp: sm.Matrix = self._jacobian[[0, 1, 2], :]
            while temp.shape[0] != temp.shape[1]:
                temp = temp.col_insert(-1, sm.Matrix([1]*temp.shape[0]))
            self._inverse_jacobian = temp
            self._match_then_inv_jacobian_flag = True

        return self._jacobian, self._inverse_jacobian

    def inverse(self, x, y, z=0, initial_inv_vars=None, timeout=300, error_margin=0.005, delta=0.1, plot=False):
        if self._jacobian is None:
            print("Creating inverse kinematics data.. This will happen only once. Remember to "
                  "close the solver with close() method to save the data")
            self.calculate_jacobian()
            print("Done.")

        # target position
        position_matrix = sm.Matrix([x, y, z])

        # if initial ranged values were given, use those.
        if not initial_inv_vars:
            init = [[self._symbols[r[0]], r[2]] for r in self._ranges]
        else:
            init = [[self._symbols[r[0]], initial_inv_vars[i]] for i, r in enumerate(self._ranges)]

        # for less typing
        ij: sm.Matrix = self._inverse_jacobian

        # to keep the rules of matrix multiplication (cols1 = rows2)
        if ij.shape[1] < 3:
            ij = ij.col_insert(2, sm.Matrix([1]*ij.shape[0]))
        if ij.shape[1] > 3:
            # TODO: add a flag to add extra filler column to the delta_target, will be needed for 4+ link robots.
            pass

        def update(prev_values, d):  # prev_values = [[symbol, value], ], d = delta
            # difference in position between current matrix and target position
            targetV = position_matrix - self.matching_matrix.subs(prev_values)[:3, -1]

            # normalized difference between those positions accounting for some small delta step.
            delta_target = d * (targetV / targetV.norm())

            # matched jacobian inverse matrix for previous ranged values.
            if not self._match_then_inv_jacobian_flag:
                ij_matched = ij.subs(prev_values)
            else:
                ij_matched = ij.subs(prev_values).inv()

            # velocity of ranged values = Inverse Jacobian * velocity of change of position
            deltaValues = ij_matched * delta_target

            # getting new ranged values
            new_values = []
            for dv, val in zip(deltaValues, prev_values):
                new_values.append([val[0], val[1] + dv])

            # check if new value of each ranged symbol is still in its range
            # if value is out of range, get new random starting point that's withing that range.
            for i, (s, v) in enumerate(new_values):
                name = [k for k, v in self._symbols.items() if v == s][0]
                r = [(s1[1], s1[2], s1[3]) for s1 in self._ranges if s1[0] == name][0]
                rmin, rmax = r[0], r[2]

                if rmin + error_margin <= v <= rmax + error_margin:
                    continue
                new_values[i][1] = random.uniform(rmin, rmax)

            # end update loop if the end position is close enough to the target.
            if (-error_margin < targetV[0, 0] < error_margin) and (-error_margin < targetV[1, 0] < error_margin):
                if self._dimensions == 2:
                    return prev_values, d, 1
                if -error_margin < targetV[2, 0] < error_margin:
                    return prev_values, d, 1

            # continue the update loop
            return new_values, d, 0

        t = 0
        current = init
        while t != timeout:
            t = t + 1
            current, delta, is_done = update(current, delta)
            if is_done:
                break

        if t == timeout:
            print("Operation timed out.")
            return None

        if plot:
            self.plot_inverse(current)

        return current

    def plot_inverse(self, values):
        first_M = self._link_transforms[0].subs(values)
        final_M = self.matching_matrix.subs(values)
        matrices = [first_M, ]
        coords = [first_M[:3, 3:], ]
        for link in self._link_transforms[1:-1]:
            temp = link.subs(values)
            for m in matrices:
                temp = m * temp
            matrices.append(temp)
            coords.append(temp[:3, 3:])

        coords.append(final_M[:3, 3:])

        x = [0, ]
        y = [0, ]
        z = [0, ]
        for c in coords:
            x.append(c[0])
            y.append(c[1])
            if self._dimensions == 3:
                z.append(c[2])

        if self._dimensions == 2:
            plt.plot(x, y)
        else:
            axis = plt.axes(projection='3d')
            axis.plot3D(x, y, z)
        plt.gca().set_aspect("equal")
        plt.show()

    @staticmethod
    def plot2d(x, y):
        plt.scatter(x, y)
        plt.gca().set_aspect("equal")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    @staticmethod
    def plot3d(x, y, z):
        axis = plt.figure().add_subplot(projection="3d")
        axis.set_xlabel('X')
        axis.set_ylabel('Y')
        axis.set_zlabel('Z')
        axis.scatter(x, y, z)
        plt.gca().set_aspect("equal")
        plt.show()
