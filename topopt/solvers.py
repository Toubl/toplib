"""
Solvers to solve topology optimization problems.

Todo:
    * Make TopOptSolver an abstract class
    * Rename the current TopOptSolver to MMASolver(TopOptSolver)
    * Create a TopOptSolver using originality criterion
"""
from __future__ import division

import numpy
import nlopt

from topopt.problems import Problem
from topopt.filters import Filter
from topopt.filters import DensityBasedFilter

from topopt.guis import GUI
import time


class TopOptSolver:
    """Solver for topology optimization problems using NLopt's MMA solver."""

    def __init__(self, problem: Problem, volfrac: float, filter: Filter,
                 gui: GUI, constraints, constraints_f, maxeval=2000, ftol_rel=1e-3):
        """
        Create a solver to solve the problem.

        Parameters
        ----------
        problem: :obj:`topopt.problems.Problem`
            The topology optimization problem to solve.
        volfrac: float
            The maximum fraction of the volume to use.
        filter: :obj:`topopt.filters.Filter`
            A filter for the solutions to reduce artefacts.
        gui: :obj:`topopt.guis.GUI`
            The graphical user interface to visualize intermediate results.
        maxeval: int
            The maximum number of evaluations to perform.
        ftol: float
            A floating point tolerance for relative change.

        """
        self.problem = problem
        self.filter = filter
        self.gui = gui

        n = problem.nelx * problem.nely * problem.nelz
        self.opt = nlopt.opt(nlopt.LD_MMA, n)
        self.xPhys = numpy.ones(n)

        # set bounds on the value of x (0 ≤ x ≤ 1)
        self.opt.set_lower_bounds(numpy.zeros(n))
        self.opt.set_upper_bounds(numpy.ones(n))

        # set stopping criteria
        self.maxeval = maxeval
        self.ftol_rel = ftol_rel

        # set objective and constraint functions
        self.volfrac = volfrac  # max volume fraction to use


        if (len(constraints) == 0):
            self.opt.set_min_objective(self.objective_function)
            self.opt.add_inequality_constraint(self.volume_function, 0)
        elif (len(constraints) != 0):
            self.opt.set_min_objective(self.volume_function)
            self.opt.add_inequality_constraint(self.constraint_function_1, 0)
            self.constraint_1_f = constraints_f[0]
            self.constraint_1 = constraints[0]
            if len(constraints) > 1:
                self.opt.add_inequality_constraint(self.constraint_function_2, 0)
                self.constraint_2_f = constraints_f[1]
                self.constraint_2 = constraints[1]
            if len(constraints) > 2:
                self.opt.add_inequality_constraint(self.constraint_function_3, 0)
                self.constraint_3_f = constraints_f[2]
                self.constraint_3 = constraints[2]
            if len(constraints) > 3:
                self.opt.add_inequality_constraint(self.constraint_function_4, 0)
                self.constraint_4_f = constraints_f[3]
                self.constraint_4 = constraints[3]
            if len(constraints) > 4:
                self.opt.add_inequality_constraint(self.constraint_function_5, 0)
                self.constraint_5_f = constraints_f[4]
                self.constraint_5 = constraints[4]
            if len(constraints) > 5:
                self.opt.add_inequality_constraint(self.constraint_function_6, 0)
                self.constraint_6_f = constraints_f[5]
                self.constraint_6 = constraints[5]
            if len(constraints) > 6:
                self.opt.add_inequality_constraint(self.constraint_function_7, 0)
                self.constraint_7_f = constraints_f[6]
                self.constraint_7 = constraints[6]
            if len(constraints) > 7:
                self.opt.add_inequality_constraint(self.constraint_function_8, 0)
                self.constraint_8_f = constraints_f[7]
                self.constraint_8 = constraints[7]
            if len(constraints) > 8:
                self.opt.add_inequality_constraint(self.constraint_function_9, 0)
                self.constraint_9_f = constraints_f[8]
                self.constraint_9 = constraints[8]
            if len(constraints) > 9:
                self.opt.add_inequality_constraint(self.constraint_function_10, 0)
                self.constraint_10_f = constraints_f[9]
                self.constraint_10 = constraints[9]
            if len(constraints) > 10:
                self.opt.add_inequality_constraint(self.constraint_function_11, 0)
                self.constraint_11_f = constraints_f[10]
                self.constraint_11 = constraints[10]
            if len(constraints) > 11:
                self.opt.add_inequality_constraint(self.constraint_function_12, 0)
                self.constraint_12_f = constraints_f[11]
                self.constraint_12 = constraints[11]
            if len(constraints) > 12:
                self.opt.add_inequality_constraint(self.constraint_function_13, 0)
                self.constraint_13_f = constraints_f[12]
                self.constraint_13 = constraints[12]
            if len(constraints) > 13:
                self.opt.add_inequality_constraint(self.constraint_function_14, 0)
                self.constraint_14_f = constraints_f[13]
                self.constraint_14 = constraints[13]
            if len(constraints) > 14:
                self.opt.add_inequality_constraint(self.constraint_function_15, 0)
                self.constraint_15_f = constraints_f[14]
                self.constraint_15 = constraints[14]
            if len(constraints) > 15:
                self.opt.add_inequality_constraint(self.constraint_function_16, 0)
                self.constraint_16_f = constraints_f[15]
                self.constraint_16 = constraints[15]
            if len(constraints) > 16:
                self.opt.add_inequality_constraint(self.constraint_function_17, 0)
                self.constraint_17_f = constraints_f[16]
                self.constraint_17 = constraints[16]
            if len(constraints) > 17:
                self.opt.add_inequality_constraint(self.constraint_function_18, 0)
                self.constraint_18_f = constraints_f[17]
                self.constraint_18 = constraints[17]
            if len(constraints) > 18:
                self.opt.add_inequality_constraint(self.constraint_function_19, 0)
                self.constraint_19_f = constraints_f[18]
                self.constraint_19 = constraints[18]
            if len(constraints) > 19:
                self.opt.add_inequality_constraint(self.constraint_function_20, 0)
                self.constraint_20_f = constraints_f[19]
                self.constraint_20 = constraints[19]
            if len(constraints) > 20:
                self.opt.add_inequality_constraint(self.constraint_function_21, 0)
                self.constraint_21_f = constraints_f[20]
                self.constraint_21 = constraints[20]
            if len(constraints) > 21:
                self.opt.add_inequality_constraint(self.constraint_function_22, 0)
                self.constraint_22_f = constraints_f[21]
                self.constraint_22 = constraints[21]

            self.volfrac = 0


        # setup filter
        self.passive = problem.bc.passive_elements
        if self.passive.size > 0:
            self.xPhys[self.passive] = 0
        self.active = problem.bc.active_elements
        if self.active.size > 0:
            self.xPhys[self.active] = 1

    def __str__(self):
        """Create a string representation of the solver."""
        return self.__class__.__name__

    def __format__(self, format_spec):
        """Create a formated representation of the solver."""
        return "{} with {}".format(str(self.problem), str(self))

    def __repr__(self):
        """Create a representation of the solver."""
        return ("{}(problem={!r}, volfrac={:g}, filter={!r}, ".format(
            self.__class__.__name__, self.problem, self.volfrac, self.filter)
            + "gui={!r}, maxeval={:d}, ftol={:g})".format(
                self.gui, self.opt.get_maxeval(), self.opt.get_ftol_rel()))

    @property
    def ftol_rel(self):
        """:obj:`float`: Relative tolerance for convergence."""
        return self.opt.get_ftol_rel()

    @ftol_rel.setter
    def ftol_rel(self, ftol_rel):
        self.opt.set_ftol_rel(ftol_rel)

    @property
    def maxeval(self):
        """:obj:`int`: Maximum number of objective evaluations (iterations)."""
        return self.opt.get_maxeval()

    @maxeval.setter
    def maxeval(self, ftol_rel):
        self.opt.set_maxeval(ftol_rel)

    def optimize(self, x: numpy.ndarray) -> numpy.ndarray:
        """
        Optimize the problem.

        Parameters
        ----------
        x:
            The initial value for the design variables.

        Returns
        -------
        numpy.ndarray
            The optimal value of x found.

        """
        self.iter = 0
        self.xPhys = x.copy()
        x = self.opt.optimize(x)
        return x

    def filter_variables(self, x: numpy.ndarray) -> numpy.ndarray:
        """
        Filter the variables and impose values on passive/active variables.

        Parameters
        ----------
        x:
            The variables to be filtered.

        Returns
        -------
        numpy.ndarray
            The filtered "physical" variables.

        """
        self.filter.filter_variables(x, self.xPhys)
        if self.passive.size > 0:
            self.xPhys[self.passive] = 0
        if self.active.size > 0:
            self.xPhys[self.active] = 1
        return self.xPhys

    def constraint_function_1(
            self, x: numpy.ndarray, dobj: numpy.ndarray) -> float:
        print(self.iter)
        self.iter += 1
        self.problem.f[:6] = self.problem.f[:6] * 0
        self.problem.f[:6] = self.constraint_1_f
        self.init = 1
        constraint_1 = self.constraint_function(x, dobj) - self.constraint_1
        if constraint_1 > 0:
            print('constraint 1')
        return constraint_1

    def constraint_function_2(
            self, x: numpy.ndarray, dobj: numpy.ndarray) -> float:
        self.problem.f[:6] = self.problem.f[:6] * 0
        self.problem.f[:6] = self.constraint_2_f
        self.init = 0
        constraint_2 = self.constraint_function(x, dobj) - self.constraint_2
        if constraint_2 > 0:
            print('constraint 2')
        return constraint_2

    def constraint_function_3(
            self, x: numpy.ndarray, dobj: numpy.ndarray) -> float:
        self.problem.f[:6] = self.problem.f[:6] * 0
        self.problem.f[:6] = self.constraint_3_f
        self.init = 0
        constraint_3 = self.constraint_function(x, dobj) - self.constraint_3
        if constraint_3 > 0:
            print('constraint 3')
        return constraint_3

    def constraint_function_4(
            self, x: numpy.ndarray, dobj: numpy.ndarray) -> float:
        self.problem.f[:6] = self.problem.f[:6] * 0
        self.problem.f[:6] = self.constraint_4_f
        self.init = 0
        constraint_4 = self.constraint_function(x, dobj) - self.constraint_4
        if constraint_4 > 0:
            print('constraint 4')
        return constraint_4

    def constraint_function_5(
            self, x: numpy.ndarray, dobj: numpy.ndarray) -> float:
        self.problem.f[:6] = self.problem.f[:6] * 0
        self.problem.f[:6] = self.constraint_5_f
        self.init = 0
        constraint_5 = self.constraint_function(x, dobj) - self.constraint_5
        if constraint_5 > 0:
            print('constraint 5')
        return constraint_5

    def constraint_function_6(
            self, x: numpy.ndarray, dobj: numpy.ndarray) -> float:
        self.problem.f[:6] = self.problem.f[:6] * 0
        self.problem.f[:6] = self.constraint_6_f
        self.init = 0
        constraint_6 = self.constraint_function(x, dobj) - self.constraint_6
        if constraint_6 > 0:
            print('constraint 6')
        return constraint_6

    def constraint_function_7(
            self, x: numpy.ndarray, dobj: numpy.ndarray) -> float:
        self.problem.f[:6] = self.problem.f[:6] * 0
        self.problem.f[:6] = self.constraint_7_f
        self.init = 0
        constraint_7 = self.constraint_function(x, dobj) - self.constraint_7
        if constraint_7 > 0:
            print('constraint 7')
        return constraint_7

    def constraint_function_8(
            self, x: numpy.ndarray, dobj: numpy.ndarray) -> float:
        self.problem.f[:6] = self.problem.f[:6] * 0
        self.problem.f[:6] = self.constraint_8_f
        self.init = 0
        constraint_8 = self.constraint_function(x, dobj) - self.constraint_8
        if constraint_8 > 0:
            print('constraint 8')
        return constraint_8


    def constraint_function_9(
            self, x: numpy.ndarray, dobj: numpy.ndarray) -> float:
        self.problem.f[:6] = self.problem.f[:6] * 0
        self.problem.f[:6] = self.constraint_9_f
        self.init = 0
        constraint_9 = self.constraint_function(x, dobj) - self.constraint_9
        if constraint_9 > 0:
            print('constraint 9')
        return constraint_9

    def constraint_function_10(
            self, x: numpy.ndarray, dobj: numpy.ndarray) -> float:
        self.problem.f[:6] = self.problem.f[:6] * 0
        self.problem.f[:6] = self.constraint_10_f
        self.init = 0
        constraint_10 = self.constraint_function(x, dobj) - self.constraint_10
        if constraint_10 > 0:
            print('constraint 10')
        return constraint_10

    def constraint_function_11(
            self, x: numpy.ndarray, dobj: numpy.ndarray) -> float:
        self.problem.f[:6] = self.problem.f[:6] * 0
        self.problem.f[:6] = self.constraint_11_f
        self.init = 0
        constraint_11 = self.constraint_function(x, dobj) - self.constraint_11
        if constraint_11 > 0:
            print('constraint 11')
        return constraint_11

    def constraint_function_12(
            self, x: numpy.ndarray, dobj: numpy.ndarray) -> float:
        self.problem.f[:6] = self.problem.f[:6] * 0
        self.problem.f[:6] = self.constraint_12_f
        self.init = 0
        constraint_12 = self.constraint_function(x, dobj) - self.constraint_12
        if constraint_12 > 0:
            print('constraint 12')
        return constraint_12

    def constraint_function_13(
            self, x: numpy.ndarray, dobj: numpy.ndarray) -> float:
        self.problem.f[:6] = self.problem.f[:6] * 0
        self.problem.f[:6] = self.constraint_13_f
        self.init = 0
        constraint_13 = self.constraint_function(x, dobj) - self.constraint_13
        if constraint_13 > 0:
            print('constraint 13')
        return constraint_13

    def constraint_function_14(
            self, x: numpy.ndarray, dobj: numpy.ndarray) -> float:
        self.problem.f[:6] = self.problem.f[:6] * 0
        self.problem.f[:6] = self.constraint_14_f
        self.init = 0
        constraint_14 = self.constraint_function(x, dobj) - self.constraint_14
        if constraint_14 > 0:
            print('constraint 14')
        return constraint_14

    def constraint_function_15(
            self, x: numpy.ndarray, dobj: numpy.ndarray) -> float:
        self.problem.f[:6] = self.problem.f[:6] * 0
        self.problem.f[:6] = self.constraint_15_f
        self.init = 0
        constraint_15 = self.constraint_function(x, dobj) - self.constraint_15
        if constraint_15 > 0:
            print('constraint 15')
        return constraint_15

    def constraint_function_16(
            self, x: numpy.ndarray, dobj: numpy.ndarray) -> float:
        self.problem.f[:6] = self.problem.f[:6] * 0
        self.problem.f[:6] = self.constraint_16_f
        self.init = 0
        constraint_16 = self.constraint_function(x, dobj) - self.constraint_16
        if constraint_16 > 0:
            print('constraint 16')
        return constraint_16

    def constraint_function_17(
            self, x: numpy.ndarray, dobj: numpy.ndarray) -> float:
        self.problem.f[:6] = self.problem.f[:6] * 0
        self.problem.f[:6] = self.constraint_17_f
        self.init = 0
        constraint_17 = self.constraint_function(x, dobj) - self.constraint_17
        if constraint_17 > 0:
            print('constraint 17')
        return constraint_17


    def constraint_function_18(
            self, x: numpy.ndarray, dobj: numpy.ndarray) -> float:
        self.problem.f[:6] = self.problem.f[:6] * 0
        self.problem.f[:6] = self.constraint_18_f
        self.init = 0
        constraint_18 = self.constraint_function(x, dobj) - self.constraint_18
        if constraint_18 > 0:
            print('constraint 18')
        return constraint_18


    def constraint_function_19(
            self, x: numpy.ndarray, dobj: numpy.ndarray) -> float:
        self.problem.f[:6] = self.problem.f[:6] * 0
        self.problem.f[:6] = self.constraint_19_f
        self.init = 0
        constraint_19 = self.constraint_function(x, dobj) - self.constraint_19
        if constraint_19 > 0:
            print('constraint 19')
        return constraint_19


    def constraint_function_20(
            self, x: numpy.ndarray, dobj: numpy.ndarray) -> float:
        self.problem.f[:6] = self.problem.f[:6] * 0
        self.problem.f[:6] = self.constraint_20_f
        self.init = 0
        constraint_20 = self.constraint_function(x, dobj) - self.constraint_20
        if constraint_20 > 0:
            print('constraint 20')
        return constraint_20

    def constraint_function_21(
            self, x: numpy.ndarray, dobj: numpy.ndarray) -> float:
        self.problem.f[:6] = self.problem.f[:6] * 0
        self.problem.f[:6] = self.constraint_21_f
        self.init = 0
        constraint_21 = self.constraint_function(x, dobj) - self.constraint_21
        if constraint_21 > 0:
            print('constraint 21')
        return constraint_21

    def constraint_function_22(
            self, x: numpy.ndarray, dobj: numpy.ndarray) -> float:
        self.problem.f[:6] = self.problem.f[:6] * 0
        self.problem.f[:6] = self.constraint_22_f
        self.init = 0
        constraint_22 = self.constraint_function(x, dobj) - self.constraint_22
        if constraint_22 > 0:
            print('constraint 22')
        return constraint_22

    def constraint_function(
            self, x: numpy.ndarray, dobj: numpy.ndarray) -> float:
        """
        Compute the objective value and gradient.

        Parameters
        ----------
        x:
            The design variables for which to compute the objective.
        dobj:
            The gradient of the objective to compute.

        Returns
        -------
        float
            The objective value.

        """
        # Filter design variables
        x = numpy.reshape(x, (self.problem.nelx, self.problem.nelz, self.problem.nely), order='C')
        x_flipped_0 = x[:, int(self.problem.nelz / 2):, :]
        x_flipped = numpy.flip(x_flipped_0, axis=1)
        x_flipped = numpy.flip(x_flipped, axis=2)
        x = numpy.hstack((x_flipped, x_flipped_0))
        x = numpy.reshape(x, x.size, order='C')

        self.filter_variables(x)


        if self.init == 1:
            self.problem.compute_displacements3_predef(self.xPhys)

        # Objective and sensitivity
        obj = self.problem.compute_compliance3_predef(self.xPhys, dobj)
        # self.passive = self.problem.passive.indices

        # Sensitivity filtering
        self.filter.filter_objective_sensitivities(self.xPhys, dobj)

        # Display physical variables
        self.dc = dobj
        return obj

    def objective_function(
            self, x: numpy.ndarray, dobj: numpy.ndarray) -> float:
        """
        Compute the objective value and gradient.

        Parameters
        ----------
        x:
            The design variables for which to compute the objective.
        dobj:
            The gradient of the objective to compute.

        Returns
        -------
        float
            The objective value.

        """
        # Filter design variables
        self.filter_variables(x)

        # Objective and sensitivity
        if self.problem.nelz > 1:
            obj = self.problem.compute_compliance3(self.xPhys, dobj)
            # self.passive = self.problem.passive.indices
        else:
            obj = self.problem.compute_compliance(self.xPhys, dobj)

        # Sensitivity filtering
        self.filter.filter_objective_sensitivities(self.xPhys, dobj)

        # Display physical variables
        if self.problem.nelz == 1:
            self.gui.update(self.xPhys)

        self.iter += 1

        self.dc = dobj
        return obj

    def objective_function_fdiff(self, x: numpy.ndarray, dobj: numpy.ndarray,
                                 epsilon=1e-6) -> float:
        """
        Compute the objective value and gradient using finite differences.

        Parameters
        ----------
        x:
            The design variables for which to compute the objective.
        dobj:
            The gradient of the objective to compute.
        epsilon:
            Change in the finite difference to compute the gradient.

        Returns
        -------
        float
            The objective value.

        """
        obj = self.objective_function(x, dobj)

        x0 = x.copy()
        dobj0 = dobj.copy()
        dobjf = numpy.zeros(dobj.shape)
        for i, v in enumerate(x):
            x = x0.copy()
            x[i] += epsilon
            o1 = self.objective_function(x, dobj)
            x[i] = x0[i] - epsilon
            o2 = self.objective_function(x, dobj)
            dobjf[i] = (o1 - o2) / (2 * epsilon)
            print("finite differences: {:g}".format(
                numpy.linalg.norm(dobjf - dobj0)))
            dobj[:] = dobj0

        return obj

    def volume_function(self, x: numpy.ndarray, dv: numpy.ndarray) -> float:
        """
        Compute the volume constraint value and gradient.

        Parameters
        ----------
        x:
            The design variables for which to compute the volume constraint.
        dobj:
            The gradient of the volume constraint to compute.

        Returns
        -------
        float
            The volume constraint value.

        """
        if self.problem.nelz > 1:
            x = numpy.reshape(x, (self.problem.nelx, self.problem.nelz, self.problem.nely), order='C')
            x_flipped_0 = x[:, int(self.problem.nelz / 2):, :]
            x_flipped = numpy.flip(x_flipped_0, axis=1)
            x_flipped = numpy.flip(x_flipped, axis=2)
            x = numpy.hstack((x_flipped, x_flipped_0))
            x = numpy.reshape(x, x.size, order='C')

        # Filter design variables
        self.filter_variables(x)

        # Volume sensitivities
        dv[:] = 1.0

        # Sensitivity filtering

        self.filter.filter_volume_sensitivities(self.xPhys, dv)

        print('Volume: ', self.xPhys.sum()/(self.problem.nelx * self.problem.nely * self.problem.nelz), '\n')
        return self.xPhys.sum() - self.volfrac * x.size


# TODO: Seperate optimizer from TopOptSolver
# class MMASolver(TopOptSolver):
#     pass
#
#
# TODO: Port over OC to TopOptSolver
class OCSolver(TopOptSolver):
    def oc(self, x, volfrac, dc):
        """ Optimality criterion """
        l1 = 0
        l2 = 1e9
        move = volfrac / 2
        # reshape to perform vector operations
        xnew = numpy.zeros((len(x)))

        while ((l2 - l1) / (l1 + l2)) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            xnew[:] = numpy.maximum(0.0, numpy.maximum(x - move, numpy.minimum(1.0,
                               numpy.minimum(x + move, x * numpy.sqrt(-dc / lmid)))))
            gt = numpy.sum(xnew)/(self.problem.nelx * self.problem.nely * self.problem.nelz) - volfrac
            if gt > 0:
                l1 = lmid
            else:
                l2 = lmid
        return xnew

    def optimize(self, x: numpy.ndarray):
        maxiter = 40
        i = 0
        j = 0

        self.dc = x.copy()
        self.xPhys = x.copy()

        while maxiter > i:
            if j == 0:
                self.problem.f[0:6] = self.problem.f[0:6] * 0
                self.problem.f[1] = 1
                j = j + 1
            elif j == 1:
                self.problem.f[0:6] = self.problem.f[0:6] * 0
                self.problem.f[2] = 1
                j = 0
            # elif j == 2:
            #     self.problem.f[0:6] = self.problem.f[0:6] * 0
            #     self.problem.f[4] = 10
            #     j = j + 1
            # elif j == 3:
            #     self.problem.f[0:6] = self.problem.f[0:6] * 0
            #     self.problem.f[5] = 10
            #     j = 0

            self.objective_function(x, self.dc)
            x_new = self.oc(self.xPhys, self.volfrac, self.dc)
            x = x_new
            i = i + 1

        self.filter = DensityBasedFilter(self.problem.nelx, self.problem.nely, self.problem.nelz, 1.5)
        i = 0

        while maxiter > i:
            if j == 0:
                self.problem.f[0:6] = self.problem.f[0:6] * 0
                self.problem.f[1] = 1
                j = j + 1
            elif j == 1:
                self.problem.f[0:6] = self.problem.f[0:6] * 0
                self.problem.f[2] = 1
                j = 0
            # elif j == 2:
            #     self.problem.f[0:6] = self.problem.f[0:6] * 0
            #     self.problem.f[4] = 10
            #     j = j + 1
            # elif j == 3:
            #     self.problem.f[0:6] = self.problem.f[0:6] * 0
            #     self.problem.f[5] = 10
            #     j = 0
            self.objective_function(x, self.dc)
            x_new = self.oc(self.xPhys, self.volfrac, self.dc)
            x = x_new
            i = i + 1

        i = 0
        while 28 > i:
            if j == 0:
                self.problem.f[0:6] = self.problem.f[0:6] * 0
                self.problem.f[1] = 1
                j = j + 1
            elif j == 1:
                self.problem.f[0:6] = self.problem.f[0:6] * 0
                self.problem.f[2] = 1
                j = 0
            # elif j == 2:
            #     self.problem.f[0:6] = self.problem.f[0:6] * 0
            #     self.problem.f[4] = 10
            #     j = j + 1
            # elif j == 3:
            #     self.problem.f[0:6] = self.problem.f[0:6] * 0
            #     self.problem.f[5] = 10
                j = 0
            self.objective_function(x, self.dc)
            x_new = self.oc(x, self.volfrac, self.dc)
            x = x_new
            i = i + 1

        return x
