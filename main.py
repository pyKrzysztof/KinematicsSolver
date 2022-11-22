from solver import Range, Link, Solver
from optimized import OptimizedSolver
from math import pi
import matplotlib.pyplot as plt


def main():
    # solver = Solver()
    # solver.add_links([0, 0.4, 0, Range("theta0", pi/2, 0, pi/2)],
    #                  [0, 0.4, 0, Range("theta1", 0, 0, pi)])
    #
    # solver.pack("2ArmManipulator.pkl", dimensions=2)
    # f, fr, i, ir = solver.initialize()
    # i, ir = solver.inverse(fr.get_symbols())
    # print(repr(i))

    solver = OptimizedSolver("2ArmManipulator.pkl")
    # solver.get_working_area(40, plot=True)
    # solver.calculate_jacobian()
    # solver.close_and_update()

    t1, t0 = solver.inverse(-0.2, 0.4)
    # print(t0)
    a = solver._link_transforms[0].subs(*t0)[:2, -1]
    b = solver._link_transforms[1].subs(*t1)[:2, -1]
    c = solver.matching_matrix.subs([t0, t1])[:2, -1]
    # print(c)
    x1 = [0, a[0], c[0]]
    y1 = [0, a[1], c[1]]
    # print(x1, y1)
    plt.plot(x1, y1)
    plt.show()
    # print(a)
    # print(solver.inverse(0.6, 0.4))



main()
