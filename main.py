# from lib.matrixlib import Matrix as _M
from solver import Range, Link, Solver


def main():

    l1 = 0.4
    l2 = 0.4

    solver = Solver()

    solver.add_links([0, l1, 0, Range("theta0", 0, 0, pi/2)],
                     [0, l2, 0, Range("theta1", 0, 0, pi)])

    f, f_r, inv, inv_r = solver.initialize()

    # print(f_r.get_symbols())

    # inv, inv_r = solver.inverse(h_transform=solver.matching_transform, ranged_symbols=f_r.get_symbols())
    # print(repr(inv))
    # print(repr(solver.get_transform_matrix()))

    f, fr, i, ir = solver.initialize()
    print(f(fr))
    #
    # # fr["theta0"] = pi/2
    # # fr["theta1"] = pi
    # # m = f(fr).matrix
    # # print(repr(m))

    X, Y, Z = solver.working_area(30)


main()
