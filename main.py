from kinematics_lib import Kinematics, Range
from math import pi


def manip_1():
    solver = Kinematics("DATA_Robot1_2ArmPlanar", dimensions=2)
    solver.set_links([0, 0.4, 0, Range("theta0", pi / 2, 0, pi / 2)],
                     [0, 0.4, 0, Range("theta1", 0, 0, pi)])

    solver.get_working_area(n=25, plot=True)
    solver.inverse(-0.2, 0.4, timeout=1000, error_margin=0.05, delta=0.04, plot=True)
    solver.close()


def manip_2():
    solver = Kinematics("DATA_Robot2_axis_twist", dimensions=3)
    solver.set_links([0, 0.4, Range("a0", 0, 0, pi/4), Range("theta0", pi/2, 0, pi/2)],
                     [0, 0.4, 0, Range("theta1", 0, 0, pi)])

    solver.get_working_area(n=10, plot=True)
    solver.inverse(-0.2, 0.4, 0.05, timeout=1000, error_margin=0.06, delta=0.04, plot=True)
    solver.close()


def manip_3():
    solver = Kinematics("DATA_Robot3_axis_twist_on_base", dimensions=3)
    solver.set_links([0, 0, Range("a0", pi/4, 0, pi/4), 0],
                     [0, 0.4, 0, Range("theta0", pi/2, 0, pi/2)],
                     [0, 0.4, 0, Range("theta1", 0, 0, pi)])

    solver.get_working_area(n=5, plot=True)
    solver.inverse(0.2, 0.6, 0.1, timeout=1000, error_margin=0.06, delta=0.04, plot=True)
    solver.inverse(-0.1, 0.3, 0, timeout=1000, error_margin=0.06, delta=0.04, plot=True)
    solver.inverse(0.2, 0.6, 0.3, timeout=1000, error_margin=0.06, delta=0.04, plot=True)
    solver.close()


def main():
    manip_1()
    manip_2()
    manip_3()


main()
