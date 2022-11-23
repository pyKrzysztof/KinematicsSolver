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

    solver = OptimizedSolver("2ArmManipulator.pkl")
    # solver.get_working_area(40, plot=True)
    # solver.calculate_jacobian()
    # solver.close_and_update()

    solver.inverse(-0.2, 0.4, timeout=1000, error_margin=0.05, delta=0.04, plot=True)
    solver.inverse(0.0, 0.6, timeout=1000, error_margin=0.05, delta=0.04, plot=True)
    solver.inverse(0.7, 0.3, timeout=1000, error_margin=0.01, delta=0.04, plot=True)


main()
