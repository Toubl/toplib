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


class TopOptSolver:
    """Solver for topology optimization problems using NLopt's MMA solver."""

    def __init__(self, problem: Problem, volfrac: float, filter: Filter,
                 gui: GUI, c_critical: float, maxeval=2000, ftol_rel=1e-3):
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
        self.c_critical = c_critical

        if (c_critical == 0):
            self.opt.set_min_objective(self.objective_function)
            self.opt.add_inequality_constraint(self.volume_function, 0)
        elif (c_critical != 0):
            self.opt.set_min_objective(self.volume_function)
            self.opt.add_inequality_constraint(self.objective_function, 0)
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
            obj = self.problem.compute_compliance3(self.xPhys, dobj, self.c_critical)
        else:
            obj = self.problem.compute_compliance(self.xPhys, dobj, self.c_critical)

        # Sensitivity filtering
        self.filter.filter_objective_sensitivities(self.xPhys, dobj)

        # Display physical variables
        if self.problem.nelz == 1:
            self.gui.update(self.xPhys)
        #print(obj, '\n')
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
                j = 1
            else:
                self.problem.f[0:6] = self.problem.f[0:6] * 0
                self.problem.f[2] = 1
                j = 0

            self.objective_function(x, self.dc)
            x_new = self.oc(self.xPhys, self.volfrac, self.dc)
            x = x_new
            i = i + 1

        self.filter = DensityBasedFilter(self.problem.nelx, self.problem.nely, self.problem.nelz, 1.5)
        i = 0

        while maxiter > i:
            # if j == 0:
            #     self.problem.f[0:6] = self.problem.f[0:6] * 0
            #     self.problem.f[1] = 1
            #     j = 1
            # else:
            #     self.problem.f[0:6] = self.problem.f[0:6] * 0
            #     self.problem.f[2] = 1
            #     j = 0
            self.objective_function(x, self.dc)
            x_new = self.oc(self.xPhys, self.volfrac, self.dc)
            x = x_new
            i = i + 1

        i = 0
        while 28 > i:
            # if j == 0:
            #     self.problem.f[0:6] = self.problem.f[0:6] * 0
            #     self.problem.f[1] = 1
            #     j = 1
            # else:
            #     self.problem.f[0:6] = self.problem.f[0:6] * 0
            #     self.problem.f[2] = 1
            #     j = 0
            self.objective_function(x, self.dc)
            x_new = self.oc(x, self.volfrac, self.dc)
            x = x_new
            i = i + 1

        return x
