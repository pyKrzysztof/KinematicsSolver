# from solver import Solver
import itertools
import pickle
import matplotlib.pyplot as plt
import sympy as sm
import numpy as np


class OptimizedSolver:

    def __init__(self, pkl):

        self.filename = pkl
        with open(pkl, 'rb') as f:
            data = pickle.load(f)
        self.matching_matrix = data[0]
        self._symbols: dict = data[1]
        self._ranges = data[2]
        self._link_transforms = data[3]
        self._dimensions = data[4]
        self._working_area = data[5]
        self._jacobian = data[6]
        try:
            self._inverse_jacobian = data[7]
        except IndexError:
            self._inverse_jacobian = None
        self._permutations = None

    def _get_permutations(self, n):
        return list(itertools.product(*[np.linspace(r[1], r[3], n) for r in self._ranges]))

    def _optimized_forward(self, values):
        m = self.matching_matrix.subs(zip(self._symbols.values(), values))
        return m[3], m[7], m[9]

    def in_range(self, values):
        for r, v in zip(self._ranges, values):
            if v < r[0] or v > r[1]:
                return 0
        return 1

    def forward(self, values):
        if self.in_range(values):
            return self._optimized_forward(values)

    def get_working_area(self, n=50, plot=False):
        if self._working_area is not None and self._working_area[0] == n:
            if plot and self._dimensions == 2:
                return self.plot2d(*self._working_area[1])
            if plot:
                return self.plot3d(*self._working_area[1])
            return self._working_area

        self._permutations = self._get_permutations(n)
        X, Y = [], []
        if self._dimensions == 2:
            for values in self._permutations:
                x, y, _ = self._optimized_forward(values)
                X.append(x)
                Y.append(y)
            self._working_area = [n, [X, Y]]
            if plot:
                return self.plot2d(X, Y)
            return X, Y

        Z = []
        for values in self._permutations:
            x, y, z = self._optimized_forward(values)
            X.append(x)
            Y.append(y)
            Z.append(z)
        self._working_area = [n, [X, Y, Z]]
        if plot:
            return self.plot2d(X, Y, Z)
        return X, Y, Z

    def close_and_update(self):
        data = [self.matching_matrix,
                self._symbols,
                self._ranges,
                self._link_transforms,
                self._dimensions,
                self._working_area,
                self._jacobian,
                self._inverse_jacobian]

        with open(self.filename, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def calculate_jacobian(self, recalculate=False):
        # TODO: categorize operations by joint types. currently assumes everything is revolute joint.
        # maybe store link type with it's transform?

        if self._jacobian is not None and recalculate is False:
            return self._jacobian, self._inverse_jacobian

        # self._link_transforms[0][:3, :3] # Rot matrix
        # self._link_transforms[0][:3, 3:] # T matrix
        nlinks = len(self._link_transforms)
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
            self._inverse_jacobian = self._jacobian[:max(2, nlinks), :].inv()
        else:
            self._inverse_jacobian = self._jacobian[:max(3, nlinks), :].pinv()
        return self._jacobian, self._inverse_jacobian

    def inverse(self, x, y, z=None, initial_inv_vars=None, timeout=300, error_margin=0.005):
        nlinks = len(self._link_transforms)
        inv_variables = []
        for link in self._link_transforms:
            for var in link.free_symbols:
                inv_variables.append(var)

        if z is None:
            position_matrix = sm.Matrix([x, y])
        else:
            position_matrix = sm.Matrix([x, y, z])

        if not initial_inv_vars:
            init = [[self._symbols[r[0]], r[2]] for r in self._ranges]
        else:
            init = initial_inv_vars

        ji_template = self._jacobian[:max(2, nlinks), :].inv()

        def update(prev_values, delta):

            targetV = position_matrix - self.matching_matrix.subs(prev_values)[:position_matrix.shape[0], -1]
            delta_target = delta * (targetV / targetV.norm())

            ji = ji_template.subs(prev_values)
            deltaValues = ji * delta_target
            new_values = []
            # print(deltaValues)
            for dv, val in zip(deltaValues, prev_values):
                new_values.append([val[0], val[1] + dv])

            # print(targetV)
            if (-error_margin < targetV[0,0] < error_margin) and (-error_margin < targetV[1,0] < error_margin):
                if self._dimensions == 2:
                    return prev_values, 1
                if [-error_margin < targetV[2, 0] < error_margin]:
                    return prev_values, 1


            return new_values, 0

        i = 0
        curr = init
        while i != timeout:
            i = i + 1
            curr, is_done = update(curr, 0.01)
            if is_done:
                return curr
        return None






        # result = self._inverse_jacobian * position_matrix



        # print(repr(target_variables))

    @staticmethod
    def plot2d(x, y):
        plt.scatter(x, y)
        plt.show()

    @staticmethod
    def plot3d(x, y, z):
        pass
