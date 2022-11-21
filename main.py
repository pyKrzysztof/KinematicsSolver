from solver import Range, Link, Solver
from optimized import OptimizedSolver
from math import pi


def main():
    solver = Solver()
    solver.add_links([0, 0.4, 0, Range("theta0", 0, 0, pi/2)],
                     [0, 0.4, 0, Range("theta1", 0, 0, pi)])

    # solver.pack("2ArmManipulator.pkl", dimensions=2)
    f, fr, i, ir = solver.initialize()
    i, ir = solver.inverse(fr.get_symbols())
    print(repr(i))

    solver = OptimizedSolver("2ArmManipulator.pkl")
    # solver.get_working_area(40, plot=True)
    # solver.close_and_update()
    j = solver.calculate_jacobian()
    print(repr(j))


main()
