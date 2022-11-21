# from solver import Solver
import pickle
import matplotlib.pyplot as plt


class OptimizedSolver:

    def __init__(self, pkl):
        data = pickle.load(pkl)
        self.matching_matrix = data[0]
        self._symbols = data[1]
        self._ranges = data[2]
        self._permutations = data[3]
        self._link_transforms = data[3]
        self._dimensions = data[4]
        # [matrix with matching already fixed in place, list of all permutations, list of all ranged symbols, list of N transforms]

    def forward(self, values):
        if self.in_range(values):
            return self._optimized_forward(values)
        return [None] * self._dimensions

    def _optimized_forward(self, values):
        m = [self.matching_matrix.subs(zip(self._symbols, values))]
        if self._dimensions == 3:
            return m[3], m[7], m[9]
        return m[3], m[7]

    def in_range(self, values):
        for r, v in zip(self._ranges, values):
            if v < r[0] or v > r[1]:
                return 0
        return 1

    def get_working_area(self, plot=False):
        X, Y = [], []
        if self._dimensions == 2:
            for values in self._permutations:
                x, y = self._optimized_forward(values)
                X.append(x)
                Y.append(y)
            if plot:
                return self.plot2d(X, Y)

        Z = []
        for values in self._permutations:
            x, y, z = self._optimized_forward(values)
            X.append(x)
            Y.append(y)
            Z.append(z)

    @staticmethod
    def plot2d(x, y):
        plt.scatter(x, y)
        plt.show()

    @staticmethod
    def plot3d(x, y, z):
        pass