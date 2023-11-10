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
import matlab.engine
import keyboard
import time
import scipy.io

from topopt.problems import Problem

class TopOptSolver:
    """Solver for topology optimization problems using NLopt's MMA solver."""

    def __init__(self, problem: Problem, n_constraints, maxeval=618, ftol_rel=1e-6):
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

        n = problem.nelx * problem.nely * problem.nelz
        self.opt = nlopt.opt(nlopt.LD_MMA, n)
        self.opt.set_param('inner_maxeval', 2)
        # self.opt.set_param('verbosity', 1)
        self.xPhys = numpy.ones(n)

        # set bounds on the value of x (0 ≤ x ≤ 1)
        self.opt.set_lower_bounds(numpy.zeros(n))
        self.opt.set_upper_bounds(numpy.ones(n))

        # set stopping criteria
        self.maxeval = maxeval

        self.ftol_rel = ftol_rel
        self.xtol_rel = ftol_rel

        # setting objective and constraints function(s)
        if (n_constraints == 0):
            self.opt.set_min_objective(self.problem.objective_function)
            self.opt.add_inequality_mconstraint(self.problem.constraints_function, numpy.zeros(1))
        elif (n_constraints != 0):
            self.opt.set_min_objective(self.problem.objective_function)
            self.opt.add_inequality_mconstraint(self.problem.constraints_function, numpy.zeros(n_constraints))

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
    def xtol_rel(self):
        """:obj:`float`: Relative tolerance for convergence."""
        return self.opt.get_xtol_rel()


    @xtol_rel.setter
    def xtol_rel(self, xtol_rel):
        self.opt.set_xtol_rel(xtol_rel)

    @property
    def maxeval(self):
        """:obj:`int`: Maximum number of objective evaluations (iterations)."""
        return self.opt.get_maxeval()

    @maxeval.setter
    def maxeval(self, maxeval):
        self.opt.set_maxeval(maxeval)

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

    def optimize2(self, x: numpy.ndarray, n_constraints) -> numpy.ndarray:
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
        if not(hasattr(self, 'eng')):
            print('starting solver engine')
            self.eng = matlab.engine.start_matlab()
            print('started solver engine')
        dobj = x.copy() * 0
        xold1 = x.copy()
        xold2 = x.copy()
        low = x.copy() * 0
        upp = x.copy() * 0 + 1
        alpha = 0.5
        if (n_constraints == 0):
            grad = numpy.zeros((1, len(x)))
            result = numpy.zeros((1,))

            for i in range(self.maxeval):
                print(i)
                obj = self.problem.objective_function(x, dobj)
                self.problem.constraints_function(result, x, grad)

                data = {'m': 1, 'n': len(x), 'iter': i + 1, 'xval': x, 'xmin': x*0, 'xmax': x*0+1,
                        'xold1': xold1, 'xold2': xold2, 'f0val': obj, 'df0dx': dobj, 'fval': result[:],
                        'dfdx': grad, 'low': low, 'upp': upp, 'a0': 1, 'a': numpy.zeros((1, 1)),
                        'c': numpy.zeros((1, 1))+1000, 'd': numpy.zeros((1, 1)),
                        'nelx': self.problem.nelx, 'nely': self.problem.nely, 'nelz': self.problem.nelz}
                if self.problem.process_number == 0:
                    scipy.io.savemat('var.mat', data)
                    self.eng.mmasub(nargout=0)
                    mat_data = scipy.io.loadmat('res.mat')
                elif self.problem.process_number == 1:
                    scipy.io.savemat('var_1.mat', data)
                    self.eng.mmasub_1(nargout=0)
                    mat_data = scipy.io.loadmat('res_1.mat')
                elif self.problem.process_number == 2:
                    scipy.io.savemat('var_2.mat', data)
                    self.eng.mmasub_2(nargout=0)
                    mat_data = scipy.io.loadmat('res_2.mat')
                elif self.problem.process_number == 3:
                    scipy.io.savemat('var_3.mat', data)
                    self.eng.mmasub_3(nargout=0)
                    mat_data = scipy.io.loadmat('res_3.mat')
                elif self.problem.process_number == 4:
                    scipy.io.savemat('var_4.mat', data)
                    self.eng.mmasub_4(nargout=0)
                    mat_data = scipy.io.loadmat('res_4.mat')
                elif self.problem.process_number == 5:
                    scipy.io.savemat('var_5.mat', data)
                    self.eng.mmasub_5(nargout=0)
                    mat_data = scipy.io.loadmat('res_5.mat')
                elif self.problem.process_number == 6:
                    scipy.io.savemat('var_6.mat', data)
                    self.eng.mmasub_6(nargout=0)
                    mat_data = scipy.io.loadmat('res_6.mat')
                elif self.problem.process_number == 7:
                    scipy.io.savemat('var_7.mat', data)
                    self.eng.mmasub_7(nargout=0)
                    mat_data = scipy.io.loadmat('res_7.mat')

                res = mat_data['res']
                xold2 = xold1.copy()
                xold1 = x.copy()
                x = res[:len(x), 0] * alpha + (1 - alpha) * x
                low = res[len(x):len(x)*2, 0]
                upp = res[len(x)*2:, 0]

                if keyboard.is_pressed('5'):
                    print("Execution stopped by user.")
                    return x, 0

        elif (n_constraints != 0):
            grad = numpy.zeros((n_constraints, len(x)))
            result = numpy.zeros((n_constraints,))
            for i in range(self.maxeval):
                obj = self.problem.objective_function(x, dobj)
                self.problem.constraints_function(result, x, grad)

                data = {'m': n_constraints, 'n': len(x), 'iter': i + 1, 'xval': x, 'xmin': x*0, 'xmax': x*0+1,
                        'xold1': xold1, 'xold2': xold2, 'f0val': obj, 'df0dx': dobj, 'fval': result[:],
                        'dfdx': grad, 'low': low, 'upp': upp, 'a0': 1, 'a': numpy.zeros((n_constraints, 1)),
                        'c': numpy.zeros((n_constraints, 1))+1000, 'd': numpy.zeros((n_constraints, 1)),
                        'nelx': self.problem.nelx, 'nely': self.problem.nely, 'nelz': self.problem.nelz}

                if self.problem.process_number == 0:
                    scipy.io.savemat('var.mat', data)
                    self.eng.mmasub(nargout=0)
                    mat_data = scipy.io.loadmat('res.mat')
                elif self.problem.process_number == 1:
                    scipy.io.savemat('var_1.mat', data)
                    self.eng.mmasub_1(nargout=0)
                    mat_data = scipy.io.loadmat('res_1.mat')
                elif self.problem.process_number == 2:
                    scipy.io.savemat('var_2.mat', data)
                    self.eng.mmasub_2(nargout=0)
                    mat_data = scipy.io.loadmat('res_2.mat')
                elif self.problem.process_number == 3:
                    scipy.io.savemat('var_3.mat', data)
                    self.eng.mmasub_3(nargout=0)
                    mat_data = scipy.io.loadmat('res_3.mat')
                elif self.problem.process_number == 4:
                    scipy.io.savemat('var_4.mat', data)
                    self.eng.mmasub_4(nargout=0)
                    mat_data = scipy.io.loadmat('res_4.mat')
                elif self.problem.process_number == 5:
                    scipy.io.savemat('var_5.mat', data)
                    self.eng.mmasub_5(nargout=0)
                    mat_data = scipy.io.loadmat('res_5.mat')
                elif self.problem.process_number == 6:
                    scipy.io.savemat('var_6.mat', data)
                    self.eng.mmasub_6(nargout=0)
                    mat_data = scipy.io.loadmat('res_6.mat')
                elif self.problem.process_number == 7:
                    scipy.io.savemat('var_7.mat', data)
                    self.eng.mmasub_7(nargout=0)
                    mat_data = scipy.io.loadmat('res_7.mat')

                res = mat_data['res']
                xold2 = xold1.copy()
                xold1 = x.copy()
                x = res[:len(x), 0] * alpha + (1 - alpha) * x
                low = res[len(x):len(x) * 2, 0]
                upp = res[len(x) * 2:, 0]
                if keyboard.is_pressed('5'):
                    print("Execution stopped by user.")
                    return x, 0

            return x
        return x

class OCSolver(TopOptSolver):
    def oc(self, x, volfrac, dc):
        """ Optimality criterion """
        l1 = 0
        l2 = 1e9
        move = 0.1
        # reshape to perform vector operations
        xnew = numpy.zeros((len(x)))

        while ((l2 - l1) / numpy.maximum((l1 + l2), 1e-9)) > 1e-4:
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
        i = 0

        self.dc = x.copy()
        self.xPhys = x.copy()

        while self.maxeval > i:
            obj = self.problem.objective_function(x, self.dc)
            x_new = self.oc(x, self.problem.volfrac, self.dc)
            x = x_new.copy()
            i = i + 1
            print(i)

        return x, obj
