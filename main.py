from solver import Range, Link, Solver
from optimized import OptimizedSolver
from math import pi


def Manipulator2():
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
    # solver.inverse(0.0, 0.6, timeout=1000, error_margin=0.05, delta=0.04, plot=True)
    # solver.inverse(0.7, 0.3, timeout=1000, error_margin=0.01, delta=0.04, plot=True)


def Manipulator3():
    # solver = Solver()
    # solver.add_links([0, 0.4, 0, Range("theta0", pi/2, 0, pi/2)],
    #                  [0, 0.4, 0, Range("theta1", 0, 0, pi)],
    #                  [0, 0.1, 0, Range("theta2", 0, -pi, pi)])
    #
    # solver.pack("3ArmManipulator.pkl", dimensions=2)

    solver = OptimizedSolver("3ArmManipulator.pkl")
    # solver.get_working_area(20, plot=True)
    # solver.calculate_jacobian()
    # solver.close_and_update()
    solver.inverse(0.2, 0.6, timeout=1000, error_margin=0.05, delta=0.04, plot=True)


def Manipulator4():
    # solver = Solver()
    # solver.add_links([0, 0.4, 0, Range("theta0", pi/2, 0, pi/2)],
    #                  [0, 0.4, 0, Range("theta1", 0, 0, pi)],
    #                  [0, 0.2, 0, Range("theta2", 0, -pi/4, pi/4)],
    #                  [0, 0.1, 0, Range("theta3", 0, -pi, pi)])

    # solver.pack("4ArmManipulator.pkl", dimensions=2)

    solver = OptimizedSolver("4ArmManipulator.pkl")
    # solver.get_working_area(20, plot=True)
    # solver.calculate_jacobian()
    # solver.close_and_update()
    solver.inverse(0.2, 0.6, timeout=1000, error_margin=0.05, delta=0.04, plot=True)


def main():
    Manipulator2()
    Manipulator3()
    # Manipulator4()

main()
