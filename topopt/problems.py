"""Topology optimization problem to solve."""

import abc

import numpy
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import cvxopt
import cvxopt.cholmod

from .boundary_conditions import BoundaryConditions
from .utils import deleterowcol, xy_to_id
from topopt.filters import Filter
from topopt.filters import DensityBasedFilter

import time
import os


class Problem(abc.ABC):
    """
    Abstract topology optimization problem.

    Attributes
    ----------
    bc: BoundaryConditions
        The boundary conditions for the problem.
    penalty: float
        The SIMP penalty value.
    f: numpy.ndarray
        The right-hand side of the FEM equation (forces).
    u: numpy.ndarray
        The variables of the FEM equation.
    obje: numpy.ndarray
        The per element objective values.

    """

    def __init__(self, bc: BoundaryConditions, penalty: float, volfrac: float, filter: Filter):
        """
        Create the topology optimization problem.

        Parameters
        ----------
        bc:
            The boundary conditions of the problem.
        penalty:
            The penalty value used to penalize fractional densities in SIMP.

        """
        # Problem size
        self.nelx = bc.nelx
        self.nely = bc.nely
        self.nelz = bc.nelz
        self.nel = self.nelx * self.nely * self.nelz
        
        
        self.xPhys = numpy.ones(self.nel)

        self.filter = filter

        # Count degrees of fredom
        if self.nelz > 1:
            self.ndof = 3 * (self.nelx + 1) * (self.nely + 1) * (self.nelz + 1)
        else:
            self.ndof = 2 * (self.nelx + 1) * (self.nely + 1)

        # SIMP penalty
        self.penalty = penalty

        # volume fraction
        self.volfrac = volfrac

        # BC's and support (half MBB-beam)
        self.bc = bc
        dofs = numpy.arange(self.ndof)
        self.fixed = bc.fixed_nodes
        self.free = numpy.setdiff1d(dofs, self.fixed)
        self.passive = 0
        self.passive_0 = 0
        self.reducedofs = 1

        # RHS and Solution vectors
        self.f = bc.forces
        self.u = numpy.zeros((self.ndof, self.f.shape[1]))

        # Per element objective
        self.obje = numpy.zeros(self.nely * self.nelx * self.nelz)

        self.iter = 0
        self.passive = self.bc.passive_elements

        # # setup filter
        # cylinder = 0
        # if cylinder == 1:
        #     x_coords = (numpy.arange(-(self.nelx // 2), self.nelx // 2) + 0.5) ** 2
        #     y_coords = (numpy.arange(-(self.nely // 2), self.nely // 2) + 0.5) ** 2
        #     z_coords = (numpy.arange(-(self.nelz // 2), self.nelz // 2) + 0.5) ** 2
        #     # Generate coordinate grids using meshgrid
        #     X, Y, Z = numpy.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        #     # T = (Y + Z) > ((self.nely / 2 - 0.5) ** 2 + 0.24)
        #     T = (Y) < (self.nely / 2 - 7.6) ** 2
        #     T = numpy.reshape(T, self.nelx * self.nely * self.nelz, order='C').astype(int)
        #     self.passive = numpy.array(numpy.where(T == 1))
        #     self.passive_0 = self.passive

        if self.passive.size > 0:
            self.xPhys[self.passive] = 0
        self.active = self.bc.active_elements
        if self.active.size > 0:
            self.xPhys[self.active] = 1


    def __str__(self) -> str:
        """Create a string representation of the problem."""
        return self.__class__.__name__

    def __format__(self, format_spec) -> str:
        """Create a formated representation of the problem."""
        return str(self)

    def __repr__(self) -> str:
        """Create a representation of the problem."""
        return "{}(bc={!r}, penalty={:g})".format(
            self.__class__.__name__, self.penalty, self.bc)

    def penalize_densities(self, x: numpy.ndarray, drho: numpy.ndarray = None
                           ) -> numpy.ndarray:
        """
        Compute the penalized densties (and optionally its derivative).

        Parameters
        ----------
        x:
            The density variables to penalize.
        drho:
            The derivative of the penealized densities to compute. Only set if
            drho is not None.

        Returns
        -------
        numpy.ndarray
            The penalized densities used for SIMP.

        """
        rho = x**self.penalty
        if drho is not None:
            assert(drho.shape == x.shape)
            drho[:] = rho
            valid = x != 0  # valid values for division
            drho[valid] *= self.penalty / x[valid]
        return rho

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


    @abc.abstractmethod
    def objective_function(
            self, xPhys: numpy.ndarray, dobj: numpy.ndarray) -> float:
        """
        Compute objective and its gradient.

        Parameters
        ----------
        xPhys:
            The design variables.
        dobj:
            The gradient of the objective to compute.

        Returns
        -------
        float
            The objective value.

        """
        pass

    @abc.abstractmethod
    def constraints_function(
            self, result, x: numpy.ndarray, grad: numpy.ndarray) -> float:
        """
        Compute m problem constraints and their gradients
        https://nlopt.readthedocs.io/en/latest/NLopt_Python_Reference/#vector-valued-constraints

        If the argument grad is not empty (which is the case for MMA),
        then grad is a 2d NumPy array of size m×n (m = number of constraints
        n = number of design variables) which should (upon return)
        be set in-place to the gradient of the function with
        respect to the optimization parameters at x. [from nlopt docs]

        Parameters
        ----------
        x:
            The design variables.
        grad:
            The gradient of the nonlinear constraint wrt the design variables

        Returns
        -------
        float
            the constraint value(s) and derivatives

        """
        pass


class ElasticityProblem2(Problem):
    """
    Abstract elasticity topology optimization problem.

    Attributes
    ----------
    Emin: float
        The Young's modulus use for the void regions.
    Emax: float
        The Young's modulus use for the solid regions.
    nu: float
        Poisson's ratio of the material.
    f: numpy.ndarray
        The right-hand side of the FEM equation (forces).
    u: numpy.ndarray
        The variables of the FEM equation (displacments).
    nloads: int
        The number of loads applied to the material.

    """

    def __init__(self, bc: BoundaryConditions, penalty: float, volfrac: float, filter: Filter, constraints, constraints_f, gui):
        """
        Create the topology optimization problem.

        Parameters
        ----------
        bc:
            The boundary conditions of the problem.
        penalty:
            The penalty value used to penalize fractional densities in SIMP.

        """
        super().__init__(bc, penalty, volfrac, filter)
        # Max and min stiffness
        self.Emin = 1e-9 # Don't set it to 0 because singularities
        self.Emax = 1.0 # In this line, this should remain at E=1 to avoid scaling lk twice. Can be modified from main() after creating problem class instance.
        
        # FE: Build the index vectors for the for coo matrix format.
        self.nu = 0.33 # This has to be set to the desired value already from class creation (either here or setting it as additional input for __init__)

        # BC's and support (half MBB-beam)
        self.bc = bc
        dofs = numpy.arange(self.ndof)
        self.fixed = bc.fixed_nodes
        self.free = numpy.setdiff1d(dofs, self.fixed)

        # Number of loads
        self.nloads = self.f.shape[1]
        self.gui = gui

        # build indices(assignment local to global dof)
        # and calculate element stiffness matrix
        self.build_indices()
        if (len(constraints) != 0):
            self.constraint_f = constraints_f
            self.constraints = constraints

    @staticmethod
    def lk(E: float = 1.0, nu: float = 0.3) -> numpy.ndarray:
        """
        Build the element stiffness matrix.

        Parameters
        ----------
        E:
            The Young's modulus of the material.
        nu:
            The Poisson's ratio of the material.

        Returns
        -------
        numpy.ndarray
            The element stiffness matrix for the material.

        """
        k = numpy.array([
            0.5 - nu / 6., 0.125 + nu / 8., -0.25 - nu / 12.,
            -0.125 + 0.375 * nu, -0.25 + nu / 12., -0.125 - nu / 8., nu / 6.,
            0.125 - 0.375 * nu])

        KE = E / (1 - nu ** 2) * numpy.array([
            [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
            [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
            [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
            [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
            [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
            [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
            [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
            [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])
        return KE
    
    def lk_scaled(self, E: float = 1.0, nu: float = 0.3, length_x: float = 1.0, length_y: float = 1.0) -> numpy.ndarray:
        """
        Build the element stiffness matrix with variable element length and height.
    
        Parameters
        ----------
        E:
            The Young's modulus of the material.
        nu:
            The Poisson's ratio of the material.
        length_x:
            Element length in x-direction (xi-direction)
        length_y:
            Element length in y-direction (eta-direction)
        Returns
        -------
        numpy.ndarray
            The element stiffness matrix for the material and element legnths
    
        """
        length_x = self.elen_x
        length_y = self.elen_y
        # length_x = 1.0 # Test if I get the same element stiffness matrix as with the original function
        # length_y = 1.0
        
        # Compute 3D constitutive matrix (linear isotropic continuum mechanics)
        C = E / (1 - nu**2) * numpy.array([[1,  nu,        0],
                                           [nu,  1,        0],
                                           [0,   0, (1-nu)/2]])
        
        # Gauss points coordinates on each direction        
        GaussPoint = [-1/numpy.sqrt(3), 1/numpy.sqrt(3)]
        
        # Matrix of vertices coordinates. Generic element centered at the origin.
        # Counter-clockwise numbering starting at bottom left
        coordinates = numpy.zeros((4, 2))
        coordinates[0,:] = [-length_x/2, -length_y/2]
        coordinates[1,:] = [ length_x/2, -length_y/2]
        coordinates[2,:] = [ length_x/2,  length_y/2]
        coordinates[3,:] = [-length_x/2,  length_y/2]
        
        # Preallocate memory for stiffness matrix
        KE = numpy.zeros((8, 8))
        
        # Loop over each Gauss point
        for xi1 in GaussPoint:
            for xi2 in GaussPoint:
                # Compute shape functions derivatives
                dShape = (1/4) * numpy.array([[-(1-xi2),  (1-xi2), (1+xi2), -(1+xi2)],
                                              [-(1-xi1), -(1+xi1), (1+xi1),  (1-xi1)]])
                
                # Compute Jacobian matrix
                JacobianMatrix = numpy.dot(dShape, coordinates)
                
                if numpy.linalg.det(JacobianMatrix) == 0:
                    print([length_x, length_y])
                    print(JacobianMatrix)
                
                # Compute auxiliar matrix for construction of B-Operator
                dNdx = numpy.linalg.inv(JacobianMatrix).dot(dShape)
                
                # B-Operator
                B = numpy.array([[dNdx[0,0],         0,   dNdx[0,1],         0,   dNdx[0,2],         0,   dNdx[0,3],        0],
                                 [0,         dNdx[1,0],   0,         dNdx[1,1],   0,         dNdx[1,2],   0,         dNdx[1,3]],
                                 [dNdx[1,0], dNdx[0,0],   dNdx[1,1], dNdx[0,1],   dNdx[1,2], dNdx[0,2],   dNdx[1,3], dNdx[0,3]]])
                
                # Add to stiffness matrix
                KE += numpy.dot(numpy.dot(B.T, C), B) * numpy.linalg.det(JacobianMatrix)

        return KE
    
    
    def RBE_interface(self, nelx, nely, K):
        """
        Create the Transformed stiffness matrix for a RBE2 interface.

        Parameters
        ----------
        nelx, nely:
            Element distretization in x and y.
        K:
            Assembled stiffness matrix without boundary conditions.

        Returns
        -------
        K:
            Transformed stiffness matrix with RBE2 interface.
        """

        row, col, data = K.row, K.col, K.data
        n = K.shape[0]

        # Create COO Matrix expanded by dofs of master node
        expanded_row = numpy.concatenate(([0, 1], row + 2))
        expanded_col = numpy.concatenate(([0, 1], col + 2))
        expanded_data = numpy.concatenate(([0, 0], data))
        K_ = scipy.sparse.coo_matrix((expanded_data, (expanded_row, expanded_col)), shape=(n + 2, n + 2))

        if hasattr(self, 'T_r'):
            # Code to execute when self.T_r exists
            K = self.T_r.transpose() @ K_ @ self.T_r
        else:
            # Code to execute when self.T_r does not exist
            # All dofs in original order
            alldofs0_r = numpy.arange(0, K_.shape[0])
            # Dofs that are to be removed(right hand boundary)
            sdofs_r = numpy.arange(K_.shape[0] - 2 * (nely + 1), K_.shape[0])
            # Dofs that remain
            mdofs_r = numpy.setdiff1d(alldofs0_r, sdofs_r)

            row_indices = []
            col_indices = []
            values = []

            # see pptx of Day 2 of Topology Optimization Practical Course
            # or see matlab code of Topology Optimization Practical Course for further reference
            for n in range(0, nely + 1):
                row_indices.extend([2 * n + 1])
                col_indices.extend([0])
                values.extend([1])

                C_t = numpy.cross([0, 0, 1], [0, nely / nelx * (0.5 - n / nely), 0])
                row_indices.extend([2 * n, 2 * n + 1])
                col_indices.extend([1, 1])
                values.extend([C_t[0], C_t[1]])

            Tsm = scipy.sparse.coo_matrix((values, (row_indices, col_indices)),
                                          shape=(2 * (nely + 1), K_.shape[0] - 2 * (nely + 1)))
            Ti = scipy.sparse.eye(len(mdofs_r))
            self.T_r = scipy.sparse.vstack((Ti, Tsm))
            K = self.T_r.transpose() @ K_ @ self.T_r
        return K

    def build_indices(self) -> None:
        """
        Build the index vectors for the finite element coo matrix format.
        This method is only called once (upon class instantiation).
        """
        # self.KE = self.lk(E=1.0, nu=self.nu)
        self.KE = self.lk_scaled(E=1.0, nu=self.nu)
        self.edofMat = numpy.zeros((self.nelx * self.nely, 8), dtype=int)
        for elx in range(self.nelx):
            for ely in range(self.nely):
                el = ely + elx * self.nely
                n1 = (self.nely + 1) * elx + ely
                n2 = (self.nely + 1) * (elx + 1) + ely
                self.edofMat[el, :] = numpy.array([
                    2 * n1 + 2, 2 * n1 + 3, 2 * n2 + 2, 2 * n2 + 3, 2 * n2,
                    2 * n2 + 1, 2 * n1, 2 * n1 + 1])
        # Construct the index pointers for the coo format
        self.iK = numpy.kron(self.edofMat, numpy.ones((8, 1))).flatten()
        self.jK = numpy.kron(self.edofMat, numpy.ones((1, 8))).flatten()


    def compute_young_moduli(self, x: numpy.ndarray, dE: numpy.ndarray = None
                             ) -> numpy.ndarray:
        """
        Compute the Young's modulus of each element from the densties.

        Optionally compute the derivative of the Young's modulus.

        Parameters
        ----------
        x:
            The density variable of each element.
        dE:
            The derivative of Young's moduli to compute. Only set if dE is not
            None.

        Returns
        -------
        numpy.ndarray
            The elements' Young's modulus.

        """
        drho = None if dE is None else numpy.empty(x.shape)
        rho = self.penalize_densities(x, drho)
        if drho is not None and dE is not None:
            assert(dE.shape == x.shape)
            dE[:] = (self.Emax - self.Emin) * drho
        return (self.Emax - self.Emin) * rho + self.Emin

    def build_K(self, xPhys: numpy.ndarray, remove_constrained: bool = True
                ) -> scipy.sparse.coo_matrix:
        """
        Build the stiffness matrix for the problem.

        Parameters
        ----------
        xPhys:
            The element densisities used to build the stiffness matrix.
        remove_constrained:
            Should the constrained nodes be removed?

        Returns
        -------
        scipy.sparse.coo_matrix
            The stiffness matrix for the mesh.

        """
        sK = ((self.KE.flatten()[numpy.newaxis]).T *
              self.compute_young_moduli(xPhys)).flatten(order='F')
        K = scipy.sparse.coo_matrix(
            (sK, (self.iK, self.jK)), shape=(self.ndof, self.ndof))
        
        if remove_constrained:
            K = self.RBE_interface(self.nelx, self.nely, K)
            # Remove constrained dofs from matrix and convert to coo
            # + 2 is because 2 dofs of master node have been added
            K = deleterowcol(K.tocsc(), self.fixed + 2, self.fixed + 2).tocoo()
        return K
    
    def compute_displacements(self, xPhys: numpy.ndarray) -> numpy.ndarray:
        """
        Compute the displacements given the densities.

        Compute the displacment, :math:`u`, using linear elastic finite
        element analysis (solving :math:`Ku = f` where :math:`K` is the
        stiffness matrix and :math:`f` is the force vector).

        Parameters
        ----------
        xPhys:
            The element densisities used to build the stiffness matrix.

        Returns
        -------
        numpy.ndarray
            The distplacements solve using linear elastic finite element
            analysis.

        """
        # Setup and solve FE problem

        # if deleting of dofs of passive elements leaves element disconnected matrix will become singular
        # in this case calculation is performed again with passive elements of previous iteration
        passive_backup = self.passive.copy()
        xPhys_backup = xPhys.copy()

        if self.reducedofs == 1:
            self.passive = scipy.sparse.csr_matrix(xPhys < 1e-7)
            if not (isinstance(self.passive_0, int)):
                self.passive = numpy.union1d(self.passive.indices, self.passive_0)
            else:
                self.passive = self.passive.indices
            xPhys[self.passive] = 0
        else:
            self.passive = scipy.sparse.csr_matrix(xPhys < 0)
            # self.Emin = 1e-9

        # building stiffness matrix
        K = self.build_K(xPhys)

        # removing rows and columns of zeros from stiffness matrix
        # and converting to cvxopt
        K = K.tocsr()
        s = numpy.diff(K.indptr) == 0
        s = scipy.sparse.csr_matrix(s)
        s = s.indices
        K = deleterowcol(K, s, s)
        l = K.shape[0]
        K = K.tocoo()
        K = cvxopt.spmatrix(
            K.data, K.row.astype(int), K.col.astype(int))

        # building force vector
        F = numpy.zeros((l, self.f.shape[1]))
        F[:2, :] = self.f[:2, :]
        F = cvxopt.matrix(F)

        # solving the system
        try:
            cvxopt.cholmod.linsolve(K, F)  # F stores solution after solve
            print('success')

            # inserting zeros for displacement of passive elements
            zeros_array = numpy.zeros(len(s))
            indices = numpy.arange(len(s))
            F = numpy.insert(F, s - indices, np.tile(zeros_array, (self.f.shape[1], 1)).T, axis=0)
        except:
            try:
                print('failed')
                self.passive = passive_backup
                xPhys_backup[passive_backup] = 0
                print(len(self.passive))
                K = self.build_K(xPhys_backup)
                K = K.tocsr()
                s = numpy.diff(K.indptr) == 0
                s = scipy.sparse.csr_matrix(s)
                s = s.indices
                K = deleterowcol(K, s, s)
                l = K.shape[0]
                K = K.tocoo()
                K = cvxopt.spmatrix(
                    K.data, K.row.astype(int), K.col.astype(int))
                F = numpy.zeros((l, self.f.shape[1]))
                F[:2] = self.f[:2, :]
                F = cvxopt.matrix(F)
                cvxopt.cholmod.linsolve(K, F)  # F stores solution after solve
                print('success')
                zeros_array = numpy.zeros(len(s))
                indices = numpy.arange(len(s))
                F = numpy.insert(F, s - indices, np.tile(zeros_array, (self.f.shape[1], 1)).T, axis=0)
            except:
                print('failed')
                self.Emin = 1e-9
                K = self.build_K(xPhys_backup)
                K = K.tocsr()
                s = numpy.diff(K.indptr) == 0
                s = scipy.sparse.csr_matrix(s)
                s = s.indices
                K = deleterowcol(K, s, s)
                l = K.shape[0]
                K = K.tocoo()
                K = cvxopt.spmatrix(
                    K.data, K.row.astype(int), K.col.astype(int))
                F = numpy.zeros((l, self.f.shape[1]))
                F[:2] = self.f[:2, :]
                F = cvxopt.matrix(F)
                cvxopt.cholmod.linsolve(K, F)  # F stores solution after solve
                print('success')
                zeros_array = numpy.zeros(len(s))
                indices = numpy.arange(len(s))
                F = numpy.insert(F, s - indices, np.tile(zeros_array, (self.f.shape[1], 1)).T, axis=0)
                self.Emin = 0

        # inserting zeros for displacement of fixed dofs
        zeros_to_insert = numpy.zeros(len(self.fixed))
        F = numpy.insert(F, 2, np.tile(zeros_to_insert, (self.f.shape[1], 1)).T, axis=0)  # Insert zeros of fixed dofs
        F = self.T_r @ F  # retransform to inlude slave nodes
        u_m = F[:2]  # Displacement of master node
        print(f'Master Vertical Disp: {u_m[0]}')
        F = F[2:]  # delete master node dofs
        new_u = F
        return new_u, u_m


    def update_displacements(self, xPhys: numpy.ndarray) -> None:
        """
        Update the displacements of the problem.

        Parameters
        ----------
             xPhys   :
            The element densisities used to compute the displacements.

        """
        self.u[:, :], _ = self.compute_displacements(xPhys)
    def compute_compliance(
            self, xPhys: numpy.ndarray, dc: numpy.ndarray) -> float:
        r"""
        Compute compliance and its gradient.

        The objective is :math:`\mathbf{f}^{T} \mathbf{u}`. The gradient of
        the objective is

        :math:`\begin{align}
        \mathbf{f}^T\mathbf{u} &= \mathbf{f}^T\mathbf{u} -
        \boldsymbol{\lambda}^T(\mathbf{K}\mathbf{u} - \mathbf{f})\\
        \frac{\partial}{\partial \rho_e}(\mathbf{f}^T\mathbf{u}) &=
        (\mathbf{K}\boldsymbol{\lambda} - \mathbf{f})^T
        \frac{\partial \mathbf u}{\partial \rho_e} +
        \boldsymbol{\lambda}^T\frac{\partial \mathbf K}{\partial \rho_e}
        \mathbf{u}
        = \mathbf{u}^T\frac{\partial \mathbf K}{\partial \rho_e}\mathbf{u}
        \end{align}`

        where :math:`\boldsymbol{\lambda} = \mathbf{u}`.

        Parameters
        ----------
        xPhys:
            The element densities.
        dobj:
            The gradient of compliance.

        Returns
        -------
        float
            The compliance value.

        """
        # Setup and solve FE problem
        # obtain displacements self.u
        self.update_displacements(xPhys)

        c = 0.0
        dc[:] = 0.0
        dE = numpy.empty(xPhys.shape)
        E = self.compute_young_moduli(xPhys, dE)
        for i in range(self.nloads):
            # displacement of every dof of every element
            ui = self.u[:, i][self.edofMat].reshape(-1, 8)
            # calculate strain energy of every element using displacements
            self.obje[:] = (ui @ self.KE * ui).sum(1)
            # multiplying by E and calculate sum to obtain total strain energy
            c += (E * self.obje).sum()
            # calculate derivative of strain/compliance energy with respect to density of each element
            dc[:] += -dE * self.obje
        dc /= float(self.nloads)
        return c


    def compute_volume(
            self, x: numpy.ndarray, grad: numpy.ndarray) -> float:
        """
        Compute m problem constraints and their gradients
        https://nlopt.readthedocs.io/en/latest/NLopt_Python_Reference/#vector-valued-constraints

        If the argument grad is not empty (which is the case for MMA),
        then grad is a 2d NumPy array of size m×n (m = number of constraints
        n = number of design variables) which should (upon return)
        be set in-place to the gradient of the function with
        respect to the optimization parameters at x. [from nlopt docs]

        Parameters
        ----------
        result:
            The design variables.
        grad:
            The gradient of the nonlinear constraint wrt the design variables

        Returns
        -------
        float
            the constraint value(s) and derivatives

        """
        # Filter design variables
        # self.filter_variables(x) # [pag]: This was commented for some reason
        self.xPhys = x.copy()

        # Volume sensitivities
        grad[:] = 1.0

        # Sensitivity filtering

        self.filter.filter_volume_sensitivities(self.xPhys, grad[:])

        # print('Vfrac: ', self.xPhys.sum() / (self.nelx * self.nely * self.nelz), '\n')

        return self.xPhys.sum()

    def compute_reduced_stiffness(self, x_opt: numpy.ndarray):
        """

        Parameters
        ----------
        x_opt:
            The optimized design variables.

        Returns
        -------
        K_red:
            2x2 stiffness matrix
        C_red:
            2x2 compliance matrix
        """
        C_red = numpy.zeros((2, 2))

        print("volfrac:")
        print(x_opt.sum()/(self.nelx * self.nely * self.nelz))
        print("nelx, nely, nelz:")
        print((self.nelx, self.nely, self.nelz), '\n')
        print("F:")
        print(numpy.transpose(self.f[0:2]))
        F = self.f[0:2].copy()
        self.f = self.f[:, 0]
        self.f = self.f.reshape(-1, 1)
        m = 0

        # calculate compliance matrix
        for i in [0, 1]:
            self.f[0:2] = 0 * self.f[0:2]
            self.f[i] = 1
            _, U = self.compute_displacements(x_opt)
            U = U[[0, 1]]
            C_red[:, m] = numpy.transpose(U)
            m = m + 1

        K_red = numpy.linalg.inv(C_red)

        # Calculate the threshold for cutting off entries
        threshold = numpy.max(np.abs(K_red)) * 1e-6

        # Apply the threshold and update the array
        K_red = numpy.where(np.abs(K_red) >= threshold, K_red, 0)

        # Calculate the threshold for cutting off entries
        threshold = numpy.max(np.abs(C_red)) * 1e-6

        # Apply the threshold and update the array
        C_red = numpy.where(np.abs(C_red) >= threshold, C_red, 0)

        print("Displacement Vector under load:")
        print(numpy.transpose(C_red @ F))

        numpy.set_printoptions(precision=4)

        print("C_red:")
        print(C_red, '\n')
        print("K_red:")
        print(K_red, '\n')
        return K_red, C_red
    
    def StaticCondensationInterfaces(self, K: scipy.sparse._csr.csr_matrix) -> scipy.sparse.csr_matrix:
        """
        Apply static condensation to reduce K to boundary degrees of freedom
        (Same process as Krischer 2022) - Name convention according to Guyan (1965)

        Parameters
        ----------
        K:
            Full stiffness matrix

        Returns
        -------
            Condensed stiffness matrix

        """
        
        # K = K.tocsr() # In coo format you can't acces elements. Also, I think this is getting modified in-place because by default copy=false
        
        # Build B and C
        B = K[self.keepDoFIds[:, numpy.newaxis], self.delDoFIds]
        C = K[self.delDoFIds[:, numpy.newaxis],  self.delDoFIds]
        
        # Compute Tg with sparse formats v1
        # Tg = scipy.sparse.lil_matrix((len(allDoFIds),len(keepDoFIds)))
        # ii = numpy.arange(len(keepDoFIds),dtype=int)
        # Tg[keepDoFIds,:] = scipy.sparse.csr_matrix( (numpy.ones_like(keepDoFIds), (ii,ii)) ) # The identity matrix part
        # Tg[delDoFIds,:]  = scipy.sparse.linalg.spsolve(-C,B.T.tocsc())
        # Tg = Tg.tocsr()
        
        # Compute Tg with sparse formats v2
        ii = numpy.arange(len(self.keepDoFIds)//2,dtype=int)
        # ii_up = ii[:]
        Tg_master_up   = scipy.sparse.csr_matrix( (numpy.ones(len(self.keepDoFIds)//2), (ii,ii)), shape=(len(self.keepDoFIds)//2,len(self.keepDoFIds)) ) # The upper identity matrix part
        Tg_slave       = scipy.sparse.linalg.spsolve(-C,B.T.tocsc()) # tocsc because otherwise there's a warning on performance issues. Maybe I can address this better....
        Tg_master_down = scipy.sparse.csr_matrix( (numpy.ones(len(self.keepDoFIds)//2), (ii,ii+len(self.keepDoFIds)//2)), shape=(len(self.keepDoFIds)//2,len(self.keepDoFIds)) ) # The lower identity matrix part
        Tg = scipy.sparse.vstack([Tg_master_up, Tg_slave, Tg_master_down])
        
        
        # Build Tg without sparse format
        # Tg = numpy.zeros((len(allDoFIds),len(keepDoFIds)))
        # Tg[keepDoFIds,:] = numpy.eye(len(keepDoFIds))
        # Tg[delDoFIds,:]  = scipy.sparse.linalg.spsolve(-C,B.T).toarray()
        
        # Compute Kr = T'KT -> I do this after the function call
        # Kg = Tg.T @ K @ Tg
        
        # Some test displacements to see if I get reasonable forces
        # horz = numpy.array([1,0,1,0,1,0,1,0])
        # vert = numpy.array([0,1,0,1,0,1,0,1])
        #rot  = numpy.array([1/6, -1, -1/6, -1, 1/6, 1, -1/6, 1])
        #rot  = numpy.array([-0.5, -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, 0.5]) # Planar rotation for 1-element mesh
        # rot  = numpy.array([-0.5, -1, 0.5, -1, -0.5, 1, 0.5, 1]) # Planar rotation for 2x1-element mesh
        #rot  = numpy.array([-0.5, -3, 0.5, -3, -0.5, 3, 0.5, 3]) # Planar rotation for 6x1-elements mesh
        
        # f1 = numpy.dot(Kg,horz)
        # f2 = numpy.dot(Kg,vert)
        # f3 = numpy.dot(Kg,rot)
        
        return Tg
    
    def KinematicCondensationInterfaces(self) -> scipy.sparse.csr_matrix:
        """
        Apply kinematic condensation. Master nodes located at interface midpoints
        (Same process as Krischer 2022)

        Parameters
        ----------
        Kg:
            Guyan-Reduced stiffness matrix

        Returns
        -------
            Kinematic-Condensed stiffness matrix (4x4)

        """
        
        # New nodes will be located at interface midpoints, and their DoFs will be
        # vertical displacement and planar rotation.
        
        # Get relative vertical distances from interface midpoints to interface nodes.
        nNodesOneSide = self.nely+1
        y0 = (nNodesOneSide-1)/2 # Half the height of the design domain
        dy = (y0 - numpy.arange(0, nNodesOneSide))*self.elen_y # Relative node heights (one side)
        dx = self.joint_locx # Relative horizontal distances (same as Kri2021)
        
        n_sdofs = nNodesOneSide*2*2
        n_mdofs = 4
        
        Cm = numpy.zeros((int(n_sdofs), n_mdofs))
        nNodes = nNodesOneSide*2
        for i in range(0, nNodes, 2):

            i_dy = int(i/2)
            # Upper half (left side of design domain)
            Cm[i:i+2,:] = [[0,  -dy[i_dy], 0, 0],
                           [1,       dx,   0, 0]] # Cols refer to u1, phi1, u2, phi2
            # Lower half (right side of design domain)
            Cm[i+nNodes:i+2+nNodes,:] = [[0, 0, 0, -dy[i_dy]],
                                         [0, 0, 1,      -dx]]
            
        # (!!!!!) This stuff below is how it "should" be (see Till's notes), but in practice only Cm is necessary -> Cm=Tr
        # Tr = numpy.vstack((numpy.eye(n_mdofs),Cm))
        # Kg_ = numpy.zeros((Kg.shape[0]+4,Kg.shape[0]+4))
        # Kg_[4:,4:] += Kg 
        # Kgr = Tr.T @ Kg_ @ Tr
        # Tr = Tr[4:,:]
        
        # # Rid body modes to test if Kgr*phi = 0
        # l = 20+6 # horizontal length of component (=nelx)
        # phi1 = numpy.array([1, 0, 1, 0]).reshape((4,1))
        # phi2 = numpy.array([-1, 2/l, 1, 2/l]).reshape((4,1))
        # f1 = numpy.dot(Kgr,phi1) # both modes should give zero forces
        # f2 = numpy.dot(Kgr,phi2)
        
        # # Test if equations from Kri2021 hold
        # K11 = Kgr[0,0]
        # K22 = Kgr[1,1]
        # K44 = Kgr[3,3]
        # K33d = K11 - Kgr[0,0]
        # K31d = -K11 - (-Kgr[0,0])
        # K21d = (K11*l**2+K22-K44)/(2*l) - Kgr[1,0]
        # K41d = (K11*l**2-K22+K44)/(2*l) - Kgr[3,0]
        # K32d = -(K11*l**2+K22-K44)/(2*l) - Kgr[2,1]
        # K42d = (K11*l**2-K22-K44)/(2) - Kgr[3,1]
        # K43d = -(K11*l**2-K22+K44)/(2*l) - Kgr[3,2]
        
        return Cm
    
class ElasticityProblem3(Problem):
    """
    Abstract elasticity topology optimization problem.

    Attributes
    ----------
    Emin: float
        The Young's modulus use for the void regions.
    Emax: float
        The Young's modulus use for the solid regions.
    nu: float
        Poisson's ratio of the material.
    f: numpy.ndarray
        The right-hand side of the FEM equation (forces).
    u: numpy.ndarray
        The variables of the FEM equation (displacments).
    nloads: int
        The number of loads applied to the material.

    """
    def __init__(self, bc: BoundaryConditions, penalty: float, volfrac: float, filter: Filter, constraints, constraints_f, gui):
        """
        Create the topology optimization problem.

        Parameters
        ----------
        bc:
            The boundary conditions of the problem.
        penalty:
            The penalty value used to penalize fractional densities in SIMP.

        """
        super().__init__(bc, penalty, volfrac, filter)
        # Max and min stiffness
        # self.Emin = 1e-9
        self.Emin = 0
        self.Emax = 1.0

        # FE: Build the index vectors for the for coo matrix format.
        self.nu = 0.3

        # BC's and support (half MBB-beam)
        self.bc = bc
        dofs = numpy.arange(self.ndof)
        self.fixed = bc.fixed_nodes
        self.free = numpy.setdiff1d(dofs, self.fixed)

        # Number of loads
        self.nloads = self.f.shape[1]

        # build indices(assignment local to global dof)
        # and calculate element stiffness matrix
        self.build_indices()
        if (len(constraints) != 0):
            self.constraint_f = constraints_f
            self.constraints = constraints

    @staticmethod
    def lk(E: float = 1, nu: float = 0.3, length_x: float = 1.0, length_y: float = 1.0,
           length_z: float = 1.0) -> numpy.ndarray:
        # Compute 3D constitutive matrix (linear continuum mechanics)
        C = E / ((1 + nu) * (1 - 2 * nu)) * numpy.array([[1 - nu, nu, nu, 0, 0, 0],
                                                         [nu, 1 - nu, nu, 0, 0, 0],
                                                         [nu, nu, 1 - nu, 0, 0, 0],
                                                         [0, 0, 0, (1 - 2 * nu) / 2, 0, 0],
                                                         [0, 0, 0, 0, (1 - 2 * nu) / 2, 0],
                                                         [0, 0, 0, 0, 0, (1 - 2 * nu) / 2]])

        # Gauss points coordinates on each direction
        GaussPoint = [-1 / numpy.sqrt(3), 1 / numpy.sqrt(3)]

        # Matrix of vertices coordinates. Generic element centered at the origin.
        coordinates = numpy.zeros((8, 3))
        coordinates[0, :] = [-length_x / 2, -length_y / 2, -length_z / 2]
        coordinates[1, :] = [length_x / 2, -length_y / 2, -length_z / 2]
        coordinates[2, :] = [length_x / 2, length_y / 2, -length_z / 2]
        coordinates[3, :] = [-length_x / 2, length_y / 2, -length_z / 2]
        coordinates[4, :] = [-length_x / 2, -length_y / 2, length_z / 2]
        coordinates[5, :] = [length_x / 2, -length_y / 2, length_z / 2]
        coordinates[6, :] = [length_x / 2, length_y / 2, length_z / 2]
        coordinates[7, :] = [-length_x / 2, length_y / 2, length_z / 2]

        # Preallocate memory for stiffness matrix
        KE = numpy.zeros((24, 24))

        # Loop over each Gauss point
        for xi1 in GaussPoint:
            for xi2 in GaussPoint:
                for xi3 in GaussPoint:
                    # Compute shape functions derivatives
                    dShape = (1 / 8) * numpy.array([[-(1 - xi2) * (1 - xi3), (1 - xi2) * (1 - xi3),
                                                     (1 + xi2) * (1 - xi3), -(1 + xi2) * (1 - xi3),
                                                     -(1 - xi2) * (1 + xi3), (1 - xi2) * (1 + xi3),
                                                     (1 + xi2) * (1 + xi3), -(1 + xi2) * (1 + xi3)],
                                                    [-(1 - xi1) * (1 - xi3), -(1 + xi1) * (1 - xi3),
                                                     (1 + xi1) * (1 - xi3), (1 - xi1) * (1 - xi3),
                                                     -(1 - xi1) * (1 + xi3), -(1 + xi1) * (1 + xi3),
                                                     (1 + xi1) * (1 + xi3), (1 - xi1) * (1 + xi3)],
                                                    [-(1 - xi1) * (1 - xi2), -(1 + xi1) * (1 - xi2),
                                                     -(1 + xi1) * (1 + xi2), -(1 - xi1) * (1 + xi2),
                                                     (1 - xi1) * (1 - xi2), (1 + xi1) * (1 - xi2),
                                                     (1 + xi1) * (1 + xi2), (1 - xi1) * (1 + xi2)]])

                    # Compute Jacobian matrix
                    JacobianMatrix = numpy.dot(dShape, coordinates)

                    # Compute auxiliar matrix for construction of B-Operator
                    auxiliar = numpy.linalg.inv(JacobianMatrix).dot(dShape)

                    # Preallocate memory for B-Operator
                    B = numpy.zeros((6, 24))

                    # Construct first three rows
                    for i in range(3):
                        for j in range(8):
                            B[i, 3 * j + 1 + (i - 1)] = auxiliar[i, j]

                    # Construct fourth row
                    for j in range(8):
                        B[3, 3 * j] = auxiliar[1, j]
                        B[3, 3 * j + 1] = auxiliar[0, j]

                    # Construct fifth row
                    for j in range(8):
                        B[4, 3 * j + 2] = auxiliar[1, j]
                        B[4, 3 * j + 1] = auxiliar[2, j]

                    # Construct sixth row
                    for j in range(8):
                        B[5, 3 * j] = auxiliar[2, j]
                        B[5, 3 * j + 2] = auxiliar[0, j]

                    # Add to stiffness matrix
                    KE += numpy.dot(numpy.dot(B.T, C), B) * numpy.linalg.det(JacobianMatrix)

        return KE

    def RBE_interface(self, nelx, nely, nelz, K, case):
        """
        Create the Transformed stiffness matrix for a RBE2 interface.

        Parameters
        ----------
        nelx, nely:
            Element distretization in x and y.
        K:
            Assembled stiffness matrix without boundary conditions.

        Returns
        -------
        K:
            Transformed stiffness matrix with RBE2 interface.
        """

        K = K.tocsr()
        indptr = np.insert(K.indptr, 0, np.zeros(6))
        K_ = scipy.sparse.csr_matrix((K.data, K.indices + 6, indptr), shape=(K.shape[0] + 6, K.shape[1] + 6))

        if hasattr(self, 'T_r') and case == 0:
            # Code to execute when self.T_r exists
            K_r = self.T_r.transpose() @ K_ @ self.T_r
        elif hasattr(self, 'T_ry') and case == 1:
            K_r = self.T_ry.transpose() @ K_ @ self.T_ry
        elif hasattr(self, 'T_rz') and case == 2:
            K_r = self.T_rz.transpose() @ K_ @ self.T_rz
        else:
            # Code to execute when self.T_r does not exist
            # All dofs in original order
            alldofs_r = numpy.arange(0, K_.shape[0])
            # Dofs that are to be removed(right hand boundary)
            sdofs_r = alldofs_r[-3 * (nely + 1) * (nelz + 1):K_.shape[0]]
            # Dofs that remain
            mdofs_r = numpy.setdiff1d(alldofs_r, sdofs_r)

            row_indices = []
            col_indices = []
            values = []

            # see pptx of Day 2 of Topology Optimization Practical Course
            # or see matlab code of Topology Optimization Practical Course for further reference
            m = 0
            for i in range(0, (nelz + 1)):
                for j in range(0, (nely + 1)):
                    row_indices.extend([3 * m + 0, 3 * m + 1, 3 * m + 2])
                    col_indices.extend([0, 1, 2])
                    values.extend([1, 1, 1])

                    row_indices.extend([3 * m + 1, 3 * m + 2])
                    col_indices.extend([3, 3])
                    values.extend([nelz / nelx * (0.5 - i / nelz), nely / nelx * (0.5 - j / nely)])

                    row_indices.extend([3 * m + 0, 3 * m + 2])
                    col_indices.extend([4, 4])
                    values.extend([-nelz / nelx * (0.5 - i / nelz), 0])

                    row_indices.extend([3 * m + 0, 3 * m + 1])
                    col_indices.extend([5, 5])
                    values.extend([-nely / nelx * (0.5 - j / nely), 0])

                    m = m + 1
            Tsm = scipy.sparse.coo_matrix((values, (row_indices, col_indices)), shape=(
            3 * (nely + 1) * (nelz + 1), K_.shape[0] - 3 * (nely + 1) * (nelz + 1)))
            Ti = scipy.sparse.eye(len(mdofs_r))
            if case == 0:
                self.T_r = scipy.sparse.vstack((Ti, Tsm))
                K_r = self.T_r.transpose() @ K_ @ self.T_r
            elif case == 1:
                self.T_ry = scipy.sparse.vstack((Ti, Tsm))
                K_r = self.T_ry.transpose() @ K_ @ self.T_ry
            elif case == 2:
                self.T_rz = scipy.sparse.vstack((Ti, Tsm))
                K_r = self.T_rz.transpose() @ K_ @ self.T_rz
        return K_r
    def build_indices(self) -> None:
        """Build the index vectors for the finite element coo matrix format."""
        self.KE = self.lk(E=self.Emax, nu=self.nu)
        nodenrs = numpy.reshape(numpy.arange(1, (1 + self.nelx) * (1 + self.nely) * (1 + self.nelz) + 1, dtype=numpy.int32),
                             (1 + self.nely, 1 + self.nelz, 1 + self.nelx), order='F')  # nodes numbering             #3D#
        edofVec = numpy.reshape(3 * nodenrs[0:self.nely, 0:self.nelz, 0:self.nelx] + 1, (self.nel, 1), order='F')  # #3D#

        a = 3 * (self.nely + 1) * (self.nelz + 1)
        b = 3 * (self.nely + 1)
        c = 3 * (self.nely + 1) * (self.nelz + 2)
        d = 3 * (self.nely + 1)
        self.edofMat = edofVec + numpy.array([0, 1, 2, a, a + 1, a + 2, a - 3, a - 2, a - 1, -3, -2, -1,
                                      b, b + 1, b + 2, c, c + 1, c + 2, c - 3, c - 2, c - 1,
                                      d - 3, d - 2, d - 1],
                                     dtype=numpy.int32) - 1  # connectivity matrix         #3D#

        self.iK = numpy.kron(self.edofMat, numpy.ones((24, 1))).flatten()
        self.jK = numpy.kron(self.edofMat, numpy.ones((1, 24))).flatten()


        # Perform same steps for case of pure bending in xy-plane (exploiting symmetry of problem)
        if self.nelz > 1:
            nodenrs = numpy.reshape(numpy.arange(1, (1 + self.nelx) * (1 + self.nely) * (1 + int(self.nelz/2)) + 1, dtype=numpy.int32),
                                 (1 + self.nely, 1 + int(self.nelz/2), 1 + self.nelx), order='F')  # nodes numbering             #3D#
            edofVec = numpy.reshape(3 * nodenrs[0:self.nely, 0:int(self.nelz/2), 0:self.nelx] + 1, (int(self.nel/2), 1), order='F')  # #3D#

            a = 3 * (self.nely + 1) * (int(self.nelz/2) + 1)
            b = 3 * (self.nely + 1)
            c = 3 * (self.nely + 1) * (int(self.nelz/2) + 2)
            d = 3 * (self.nely + 1)
            edofMat = edofVec + numpy.array([0, 1, 2, a, a + 1, a + 2, a - 3, a - 2, a - 1, -3, -2, -1,
                                          b, b + 1, b + 2, c, c + 1, c + 2, c - 3, c - 2, c - 1,
                                          d - 3, d - 2, d - 1],
                                         dtype=numpy.int32) - 1  # connectivity matrix         #3D#
            # Construct the index pointers for the coo format
            self.iK_y = numpy.kron(edofMat, numpy.ones((24, 1))).flatten()
            self.jK_y = numpy.kron(edofMat, numpy.ones((1, 24))).flatten()

        # Perform same steps for case of pure bending in xz-plane (exploiting symmetry of problem)
        if self.nely > 1:
            nodenrs = numpy.reshape(
                numpy.arange(1, (1 + self.nelx) * (1 + int(self.nely/2)) * (1 + self.nelz) + 1,
                             dtype=numpy.int32),
                (1 + int(self.nely/2), 1 + self.nelz, 1 + self.nelx),
                order='F')  # nodes numbering             #3D#
            edofVec = numpy.reshape(3 * nodenrs[0:int(self.nely/2), 0:self.nelz, 0:self.nelx] + 1,
                                    (int(self.nel / 2), 1), order='F')  # #3D#

            a = 3 * (int(self.nely/2) + 1) * (self.nelz + 1)
            b = 3 * (int(self.nely/2) + 1)
            c = 3 * (int(self.nely/2) + 1) * (self.nelz + 2)
            d = 3 * (int(self.nely/2) + 1)
            edofMat = edofVec + numpy.array([0, 1, 2, a, a + 1, a + 2, a - 3, a - 2, a - 1, -3, -2, -1,
                                             b, b + 1, b + 2, c, c + 1, c + 2, c - 3, c - 2, c - 1,
                                             d - 3, d - 2, d - 1],
                                            dtype=numpy.int32) - 1  # connectivity matrix         #3D#
            # Construct the index pointers for the coo format
            self.iK_z = numpy.kron(edofMat, numpy.ones((24, 1))).flatten()
            self.jK_z = numpy.kron(edofMat, numpy.ones((1, 24))).flatten()

    def compute_young_moduli(self, x: numpy.ndarray, dE: numpy.ndarray = None
                             ) -> numpy.ndarray:
        """
        Compute the Young's modulus of each element from the densties.

        Optionally compute the derivative of the Young's modulus.

        Parameters
        ----------
        x:
            The density variable of each element.
        dE:
            The derivative of Young's moduli to compute. Only set if dE is not
            None.

        Returns
        -------
        numpy.ndarray
            The elements' Young's modulus.

        """
        drho = None if dE is None else numpy.empty(x.shape)
        rho = self.penalize_densities(x, drho)
        if drho is not None and dE is not None:
            assert(dE.shape == x.shape)
            dE[:] = (self.Emax - self.Emin) * drho
        return (self.Emax - self.Emin) * rho + self.Emin

    def build_K(self, xPhys: numpy.ndarray, remove_constrained: bool = True
                ) -> scipy.sparse.coo_matrix:
        """
        Build the stiffness matrix for the problem.

        Parameters
        ----------
        xPhys:
            The element densisities used to build the stiffness matrix.
        remove_constrained:
            Should the constrained nodes be removed?

        Returns
        -------
        scipy.sparse.coo_matrix
            The stiffness matrix for the mesh.

        """
        K = 0
        K_y = 0
        K_z = 0
        fixed_y = 0
        fixed_z = 0
        a_y = 1  # to ensure same stiffness matrix is not built twice
        a_z = 1
        for i in range(self.nloads):
            # For case of pure bending in xy-plane (exploiting symmetry of problem)
            if all(self.f[[0, 2, 3, 4], i] == ([0, 0, 0, 0])) and a_y and self.nelz > 1:
                a_y = 0
                xPhys_y = numpy.reshape(xPhys, (self.nelx, self.nelz, self.nely), order='C')
                xPhys_y = xPhys_y[:, int(self.nelz / 2):, :]
                xPhys_y = numpy.reshape(xPhys_y, xPhys_y.size, order='C')
                sK = ((self.KE.flatten()[numpy.newaxis]).T *
                      self.compute_young_moduli(xPhys_y)).flatten(order='F')
                K_y = scipy.sparse.coo_matrix(
                    (sK, (self.iK_y, self.jK_y)), shape=(int(3*(self.nelx + 1)*(self.nely+1)*(self.nelz/2 + 1)), int(3*(self.nelx + 1)*(self.nely+1)*(self.nelz/2 + 1))))
                K_y = self.RBE_interface(self.nelx, self.nely, int(self.nelz/2), K_y, 1)

                if remove_constrained:
                    # Remove constrained dofs from matrix and convert to coo
                    dofs = numpy.arange(3 * (self.nelx + 1) * (self.nely + 1) * (int(self.nelz / 2) + 1))
                    fixed_y = dofs[0:3 * (int(self.nelz / 2) + 1) * (self.nely + 1)]
                    for f in range(self.nelx):
                        fixed2 = dofs[2 + 3 * f * (int(self.nelz / 2) + 1) * (self.nely + 1):2 + 3 * (
                                    self.nely + 1) + 3 * f * (int(self.nelz / 2) + 1) * (self.nely + 1):3]
                        fixed_y = numpy.union1d(fixed_y, fixed2)
                    # + 6 is because 6 dofs of master node have been added
                    K_y = deleterowcol(K_y, fixed_y + 6, fixed_y + 6)
                    K_y = deleterowcol(K_y, [3], [3])  # remove Torsion dof of master node (necessary)
            # For case of pure bending in xz-plane (exploiting symmetry of problem)
            elif all(self.f[[0, 1, 3, 5], i] == ([0, 0, 0, 0])) and a_z and self.nely > 1:
                a_z = 0
                xPhys_z = numpy.reshape(xPhys, (self.nelx, self.nelz, self.nely), order='C')
                xPhys_z = xPhys_z[:, :, int(self.nely / 2):]
                xPhys_z = numpy.reshape(xPhys_z, xPhys_z.size, order='C')
                sK = ((self.KE.flatten()[numpy.newaxis]).T *
                      self.compute_young_moduli(xPhys_z)).flatten(order='F')
                K_z = scipy.sparse.coo_matrix(
                    (sK, (self.iK_z, self.jK_z)), shape=(int(3 * (self.nelx + 1) * (self.nely/2 + 1) * (self.nelz + 1)),
                                                     int(3 * (self.nelx + 1) * (self.nely/2 + 1) * (self.nelz + 1))))
                K_z = self.RBE_interface(self.nelx, int(self.nely/2), self.nelz, K_z, 2)
                if remove_constrained:
                    # Remove constrained dofs from matrix and convert to coo
                    dofs = numpy.arange(3 * (self.nelx + 1) * (int(self.nely / 2) + 1) * (self.nelz + 1))
                    fixed_z = dofs[0:3 * (self.nelz + 1) * (int(self.nely / 2) + 1)]
                    fixed2 = dofs[1:-1 - 3 * (self.nelz + 1) * (int(self.nely / 2) + 1):3 * (int(self.nely / 2) + 1)]
                    fixed_z = numpy.union1d(fixed_z, fixed2)
                    # + 6 is because 6 dofs of master node have been added
                    K_z = deleterowcol(K_z.tocsc(), fixed_z + 6, fixed_z + 6)
                    K_z = deleterowcol(K_z.tocsc(), [3], [3])  # remove Torsion dof of master node (necessary)
            # For general case
            else:
                sK = ((self.KE.flatten()[numpy.newaxis]).T *
                      self.compute_young_moduli(xPhys)).flatten(order='F')
                K = scipy.sparse.coo_matrix(
                    (sK, (self.iK, self.jK)), shape=(self.ndof, self.ndof))

                if remove_constrained:
                    K = self.RBE_interface(self.nelx, self.nely, self.nelz, K, 0)

                    # Remove constrained dofs from matrix and convert to coo
                    K = deleterowcol(K, self.fixed + 6, self.fixed + 6)

        return K, K_y, K_z, fixed_y, fixed_z

    def compute_displacements(self, xPhys: numpy.ndarray) -> numpy.ndarray:
        """
        Compute the displacements given the densities.

        Compute the displacment, :math:`u`, using linear elastic finite
        element analysis (solving :math:`Ku = f` where :math:`K` is the
        stiffness matrix and :math:`f` is the force vector).

        Parameters
        ----------
        xPhys:
            The element densisities used to build the stiffness matrix.

        Returns
        -------
        numpy.ndarray
            The distplacements solve using linear elastic finite element
            analysis.

        """
        # Setup and solve FE problem
        passive_backup = self.passive
        xPhys_backup = xPhys.copy()

        # if deleting of dofs of passive elements leaves element disconnected matrix will become singular
        # in this case calculation is performed again with passive elements of previous iteration
        if self.reducedofs == 1:
            self.passive = scipy.sparse.csr_matrix(xPhys < 1e-7)
            if not(isinstance(self.passive_0, int)):
                self.passive = numpy.union1d(self.passive.indices, self.passive_0)
            else:
                self.passive = self.passive.indices
            xPhys[self.passive] = 0
        else:
            self.passive = scipy.sparse.csr_matrix(xPhys < 0)
            self.Emin = 1e-9

        # building stiffness matrix
        K, K_y, K_z, fixed_y, fixed_z = self.build_K(xPhys)

        # removing rows and columns of zeros from stiffness matrix
        # and converting to cvxopt
        # if K is still int, matrix has not been built and case does not have to be considered
        if type(K) is not int:
            # K = K.tocsr()
            t = numpy.diff(K.indptr) == 0
            t = scipy.sparse.csr_matrix(t)
            t = t.indices
            K = deleterowcol(K, t, t).tocoo()
            L = K.shape[0]
            K = cvxopt.spmatrix(
                K.data, K.row.astype(int), K.col.astype(int))

        if type(K_y) is not int:
            # K_y = K_y.tocsr()
            r = numpy.diff(K_y.indptr) == 0
            r = scipy.sparse.csr_matrix(r)
            r = r.indices
            K_y = deleterowcol(K_y, r, r).tocoo()
            L_y = K_y.shape[0]
            K_y = cvxopt.spmatrix(
                K_y.data, K_y.row.astype(int), K_y.col.astype(int))

        if type(K_z) is not int:
            # K_z = K_z.tocsr()
            s = numpy.diff(K_z.indptr) == 0
            s = scipy.sparse.csr_matrix(s)
            s = s.indices
            K_z = deleterowcol(K_z.tocsc(), s, s).tocoo()
            L_z = K_z.shape[0]
            K_z = cvxopt.spmatrix(
                K_z.data, K_z.row.astype(int), K_z.col.astype(int))

        new_u = self.u.copy()

        for i in range(self.nloads):
            # For case of pure bending in xy-plane (exploiting symmetry of problem)
            if all(self.f[[0, 2, 3, 4], i] == ([0, 0, 0, 0])) and self.nelz > 1:
                # building force vector
                F = numpy.zeros(L_y)
                f = self.f[0:6, i] / 2
                f = f[[0, 1, 2, 4, 5]]
                F[0:5] = f
                F = cvxopt.matrix(F)
                # solving the system
                try:
                    start = time.time()
                    cvxopt.cholmod.linsolve(K_y, F)
                    print(time.time() - start)

                    # inserting zeros for displacement of passive elements
                    zeros_array = np.zeros(len(r))
                    indices = np.arange(len(r))
                    F = numpy.insert(F, r - indices, zeros_array)
                except:
                    print('failed')
                    self.Emin = 1e-9
                    K, K_y, K_z, fixed_y, fixed_z = self.build_K(xPhys_backup)
                    self.passive = passive_backup
                    L_y = K_y.shape[0]
                    K_y = K_y.tocoo()
                    K_y = cvxopt.spmatrix(
                        K_y.data, K_y.row.astype(int), K_y.col.astype(int))
                    F = numpy.zeros(L_y)
                    f = self.f[0:6, i] / 2
                    f = f[[0, 1, 2, 4, 5]]
                    F[0:5] = f
                    F = cvxopt.matrix(F)
                    cvxopt.cholmod.linsolve(K_y, F)

                # inserting zero for torsion dof of master node
                F = numpy.insert(F, 3, 0)

                # inserting zeros for displacement of fixed dofs
                zeros_array = np.zeros(len(fixed_y))
                indices = np.arange(len(fixed_y))
                F = numpy.insert(F, fixed_y - indices + 6, zeros_array)
                # retransform to include slave nodes
                F = self.T_ry @ F
                u_m = F[0:6]  # Displacement of master node
                F = F[6:]  #delete master node dof
                # mirror displacements to the other side of symmetry plane
                F = self.reshape_F(F, 0)
                new_u[:, i] = numpy.array(F)[:]
            # For case of pure bending in xz-plane (exploiting symmetry of problem)
            elif all(self.f[[0, 1, 3, 5], i] == ([0, 0, 0, 0])) and self.nely > 1:
                # building force vector
                F = numpy.zeros(L_z)
                f = self.f[0:6, i] / 2
                f = f[[0, 1, 2, 4, 5]]
                F[0:5] = f
                F = cvxopt.matrix(F)
                # solving the system
                try:
                    cvxopt.cholmod.linsolve(K_z, F)

                    # inserting zeros for displacement of passive elements
                    zeros_array = np.zeros(len(s))
                    indices = np.arange(len(s))
                    F = numpy.insert(F, s - indices, zeros_array)
                except:
                    print('failed')
                    self.Emin = 1e-9
                    K, K_y, K_z, fixed_y, fixed_z = self.build_K(xPhys_backup)
                    self.passive = passive_backup
                    L_z = K_z.shape[0]
                    K_z = K_z.tocoo()
                    K_z = cvxopt.spmatrix(
                        K_z.data, K_z.row.astype(int), K_z.col.astype(int))
                    F = numpy.zeros(L_z)
                    f = self.f[0:6, i] / 2
                    f = f[[0, 1, 2, 4, 5]]
                    F[0:5] = f
                    F = cvxopt.matrix(F)
                    cvxopt.cholmod.linsolve(K_z, F)

                # inserting zero for torsion dof of master node
                F = numpy.insert(F, 3, 0)

                # inserting zeros for displacement of fixed dofs
                zeros_array = np.zeros(len(fixed_z))
                indices = np.arange(len(fixed_z))
                F = numpy.insert(F, fixed_z - indices + 6, zeros_array)
                # retransform to include slave nodes
                F = self.T_rz @ F
                u_m = F[0:6]  # Displacement of master node
                F = F[6:]  # delete master node dof
                # mirror displacements to the other side of symmetry plane
                F = self.reshape_F(F, 1)
                new_u[:, i] = numpy.array(F)[:]
            # For general case
            else:
                # building force vector
                F = numpy.zeros(L)
                F[0:6] = self.f[0:6, i]  # Force vector of reduced problem
                F = cvxopt.matrix(F)
                # solving the system
                try:
                    cvxopt.cholmod.linsolve(K, F)

                    # inserting zeros for displacement of passive elements
                    zeros_array = np.zeros(len(t))
                    indices = np.arange(len(t))
                    F = numpy.insert(F, t - indices, zeros_array)
                except:
                    print('failed')
                    self.Emin = 1e-9
                    K, K_y, K_z, fixed_y, fixed_z = self.build_K(xPhys_backup)
                    self.passive = passive_backup
                    L = K.shape[0]
                    K = K.tocoo()
                    K = cvxopt.spmatrix(
                        K.data, K.row.astype(int), K.col.astype(int))
                    F = numpy.zeros(L)
                    F[0:6] = self.f[0:6, i]
                    F = cvxopt.matrix(F)
                    cvxopt.cholmod.linsolve(K, F)

                # inserting zeros for displacement of fixed dofs
                zeros_array = np.zeros(len(self.fixed))
                indices = np.arange(len(self.fixed))
                F = numpy.insert(F, self.fixed - indices + 6, zeros_array)
                # retransform to include slave nodes
                F = self.T_r @ F
                u_m = F[0:6]  # Displacement of master node
                F = F[6:]  # delete master node dof
                new_u[:, i] = numpy.array(F)[:]
                self.Emin = 0
        return new_u, u_m


    def reshape_F(self, F, y):
        """
                mirror displacement vector F to the other side of symmetry plane

                Parameters
                ----------
                F:
                    displacement vector of halved problem
                y:
                    0: Symmetry plane is xy-Plane
                    1: Symmetry plane is xz-Plane

                Returns
                -------
                F:
                    full displacement vector

                """
        if y == 0:
            x = numpy.reshape(F[0:-1:3], (self.nelx + 1, int(self.nelz/2) + 1, self.nely + 1), order='C')
            y = numpy.reshape(F[1:-1:3], (self.nelx + 1, int(self.nelz/2) + 1, self.nely + 1), order='C')
            z = numpy.reshape(F[2:len(F):3], (self.nelx + 1, int(self.nelz/2) + 1, self.nely + 1), order='C')
            x_flipped = numpy.flip(x, axis=1)
            y_flipped = numpy.flip(y, axis=1)
            z_flipped = numpy.flip(z, axis=1)
            y_flipped = numpy.flip(y_flipped, axis=2)
            z_flipped = numpy.flip(z_flipped, axis=2)
            x_flipped = x_flipped[:, :-1, :]
            y_flipped = y_flipped[:, :-1, :]
            z_flipped = z_flipped[:, :-1, :]
            x_end = np.hstack((x_flipped, x))
            y_end = np.hstack((y_flipped, y))
            z_end = np.hstack((z_flipped, z))
            x = numpy.reshape(x_end, x_end.size, order='C')
            y = numpy.reshape(y_end, y_end.size, order='C')
            z = numpy.reshape(z_end, z_end.size, order='C')
            F = numpy.zeros(3*len(x))
            for k in range(len(x)):
                F[3 * k + 0] = x[k]
                F[3 * k + 1] = y[k]
                F[3 * k + 2] = z[k]
        else:
            x = numpy.reshape(F[0:-1:3], (self.nelx + 1, self.nelz + 1, int(self.nely / 2) + 1), order='C')
            y = numpy.reshape(F[1:-1:3], (self.nelx + 1, self.nelz + 1, int(self.nely / 2) + 1), order='C')
            z = numpy.reshape(F[2:len(F):3], (self.nelx + 1, self.nelz + 1, int(self.nely / 2) + 1), order='C')
            x_flipped = numpy.flip(x, axis=2)
            y_flipped = numpy.flip(y, axis=2)
            z_flipped = numpy.flip(z, axis=2)
            y_flipped = numpy.flip(y_flipped, axis=1)
            z_flipped = numpy.flip(z_flipped, axis=1)
            x_flipped = x_flipped[:, :, :-1]
            y_flipped = y_flipped[:, :, :-1]
            z_flipped = z_flipped[:, :, :-1]
            x_end = np.dstack((x_flipped, x))
            y_end = np.dstack((y_flipped, y))
            z_end = np.dstack((z_flipped, z))
            x = numpy.reshape(x_end, x_end.size, order='C')
            y = numpy.reshape(y_end, y_end.size, order='C')
            z = numpy.reshape(z_end, z_end.size, order='C')
            F = numpy.zeros(3 * len(x))
            for k in range(len(x)):
                F[3 * k + 0] = x[k]
                F[3 * k + 1] = y[k]
                F[3 * k + 2] = z[k]
        return F

    def update_displacements(self, xPhys: numpy.ndarray) -> None:
        """
        Update the displacements of the problem.

        Parameters
        ----------
        xPhys:
            The element densisities used to compute the displacements.

        """
        self.u[:, :], _ = self.compute_displacements(xPhys)

    def compute_compliance(
            self, xPhys: numpy.ndarray, dc: numpy.ndarray) -> float:
        r"""
        Compute compliance and its gradient.

        The objective is :math:`\mathbf{f}^{T} \mathbf{u}`. The gradient of
        the objective is

        :math:`\begin{align}
        \mathbf{f}^T\mathbf{u} &= \mathbf{f}^T\mathbf{u} -
        \boldsymbol{\lambda}^T(\mathbf{K}\mathbf{u} - \mathbf{f})\\
        \frac{\partial}{\partial \rho_e}(\mathbf{f}^T\mathbf{u}) &=
        (\mathbf{K}\boldsymbol{\lambda} - \mathbf{f})^T
        \frac{\partial \mathbf u}{\partial \rho_e} +
        \boldsymbol{\lambda}^T\frac{\partial \mathbf K}{\partial \rho_e}
        \mathbf{u}
        = \mathbf{u}^T\frac{\partial \mathbf K}{\partial \rho_e}\mathbf{u}
        \end{align}`

        where :math:`\boldsymbol{\lambda} = \mathbf{u}`.

        Parameters
        ----------
        xPhys:
            The element densities.
        dc:
            The gradient of compliance.

        Returns
        -------
        float
            The compliance value.

        """
        # Setup and solve FE problem
        # obtain displacements self.u
        self.update_displacements(xPhys)

        c = 0.0
        dc[:] = 0.0
        dE = numpy.empty(xPhys.shape)
        E = self.compute_young_moduli(xPhys, dE)
        for i in range(self.nloads):
            # displacement of every dof of every element
            ui = self.u[:, i][self.edofMat].reshape(-1, 24)
            # calculate strain energy of every element using displacements
            self.obje[:] = (ui @ self.KE * ui).sum(1)
            # multiplying by E and calculate sum to obtain total strain energy
            c += (E * self.obje).sum()
            # calculate derivative of strain/compliance energy with respect to density of each element
            dc[:] += -dE * self.obje
        dc /= float(self.nloads)
        return c

    def compute_volume(
            self, x: numpy.ndarray, grad: numpy.ndarray) -> float:
        """
        Compute m problem constraints and their gradients
        https://nlopt.readthedocs.io/en/latest/NLopt_Python_Reference/#vector-valued-constraints

        If the argument grad is not empty (which is the case for MMA),
        then grad is a 2d NumPy array of size m×n (m = number of constraints
        n = number of design variables) which should (upon return)
        be set in-place to the gradient of the function with
        respect to the optimization parameters at x. [from nlopt docs]

        Parameters
        ----------
        result:
            The design variables.
        grad:
            The gradient of the nonlinear constraint wrt the design variables

        Returns
        -------
        float
            the constraint value(s) and derivatives

        """
        # Filter design variables
        # self.filter_variables(x)
        self.xPhys = x.copy()

        # Volume sensitivities
        grad[:] = 1.0

        # Sensitivity filtering

        self.filter.filter_volume_sensitivities(self.xPhys, grad[:])

        print('Volume: ', self.xPhys.sum() / (self.nelx * self.nely * self.nelz), '\n')

        return self.xPhys.sum()

    def compute_reduced_stiffness(self, x_opt: numpy.ndarray):
        """

                Parameters
                ----------
                x_opt:
                    The optimized design variables.

                Returns
                -------
                K_red:
                    4x4 stiffness matrix (without normal strain, and torsion)
                C_red:
                    4x4 compliance matrix (without normal strain, and torsion)
                """
        C_red = numpy.zeros((4, 4))

        print("volfrac:")
        print(x_opt.sum()/(self.nelx * self.nely * self.nelz))
        print("nelx, nely, nelz:")
        print((self.nelx, self.nely, self.nelz), '\n')
        print("F:")
        print(numpy.transpose(self.f[0:6]))
        F = self.f[0:6].copy()
        F = F[[1, 2, 4, 5]]
        self.f = self.f[:, 0]
        self.f = self.f.reshape(-1, 1)
        self.nloads = 1
        m = 0

        # calculate compliance matrix
        for i in [1, 2, 4, 5]:
            self.f[0:6] = 0 * self.f[0:6]
            self.f[i] = 1
            _, U = self.compute_displacements(x_opt)
            U = U[[1, 2, 4, 5]]
            C_red[:, m] = numpy.transpose(U)  # * self.nelx
            m = m + 1
        K_red = numpy.linalg.inv(C_red)

        # Calculate the threshold for cutting off entries
        threshold = numpy.max(np.abs(K_red)) * 1e-6

        # Apply the threshold and update the array
        K_red = numpy.where(np.abs(K_red) >= threshold, K_red, 0)

        # Calculate the threshold for cutting off entries
        threshold = numpy.max(np.abs(C_red)) * 1e-6

        # Apply the threshold and update the array
        C_red = numpy.where(np.abs(C_red) >= threshold, C_red, 0)

        print("Displacement Vector under load:")
        print(numpy.transpose(C_red @ F))

        numpy.set_printoptions(precision=4)

        print("C_red:")
        print(C_red, '\n')
        print("K_red:")
        print(K_red, '\n')
        return K_red, C_red

    def StaticCondensationInterfaces(self, K: scipy.sparse._coo.coo_matrix, nelx: int, nely: int, nelz: int) -> numpy.ndarray:
        """
        Apply static condensation to reduce K to boundary degrees of freedom
        (Same process as Krischer 2022) - Name convention according to Guyan (1965)

        Parameters
        ----------
        K:
            Full stiffness matrix

        Returns
        -------
            Condensed stiffness matrix

        """
        K = K.toarray() # Would be nice to handle all this with sparse matrices
        
        ############ Static condensation
        # Find slave & master indexes -> master indexes are on the left (x=0) and right (x=nelx) boundaries
        n_nodes = (nelx + 1) * (nely + 1) * (nelz + 1)
        n_interfaceNodes = (nely+1)*(nelz+1)
        keepDoFs1 = numpy.arange(0, 3*n_interfaceNodes) # "Left" boundary DoFs
        keepDoFs2 = numpy.arange(3*(n_nodes-n_interfaceNodes),3*n_nodes) # "Right" boundary DoFs
        keepDoFIds = numpy.concatenate((keepDoFs1,keepDoFs2))
        allDoFIds = numpy.arange(3*n_nodes)
        delDoFIds = numpy.setdiff1d(allDoFIds, keepDoFIds) # DoFs that won't be kept
        
        # Build B and C
        B = K[keepDoFIds[:, numpy.newaxis], delDoFIds] # Ksm
        C = K[delDoFIds[:, numpy.newaxis],  delDoFIds] # Kss
        
        # Compute Tg (transformation matrix)
        Tg = numpy.zeros((len(allDoFIds),len(keepDoFIds)))
        Tg[keepDoFIds,:] = numpy.eye(len(keepDoFIds))
        Tg[delDoFIds,:]  = numpy.linalg.solve(-C,B.T)
        # Tg[delDoFIds,:]  = numpy.linalg.solve(-C,B.T)
        
        # Compute Kr = T'KT
        Kg = Tg.T @ K @ Tg
        
        # Some test displacements to see if I get reasonable forces
        # horz = numpy.array([1,0,1,0,1,0,1,0])
        # vert = numpy.array([0,1,0,1,0,1,0,1])
        # #rot  = numpy.array([1/6, -1, -1/6, -1, 1/6, 1, -1/6, 1])
        # #rot  = numpy.array([-0.5, -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, 0.5]) # Planar rotation for 1-element mesh
        # rot  = numpy.array([-0.5, -1, 0.5, -1, -0.5, 1, 0.5, 1]) # Planar rotation for 2x1-element mesh
        # #rot  = numpy.array([-0.5, -3, 0.5, -3, -0.5, 3, 0.5, 3]) # Planar rotation for 6x1-elements mesh
        
        # f1 = numpy.dot(Kg,horz)
        # f2 = numpy.dot(Kg,vert)
        # f3 = numpy.dot(Kg,rot)
        
        return Tg, Kg        

    def KinematicCondensationInterfaces(self, Kg: numpy.ndarray) -> numpy.ndarray:
        """
        Apply kinematic condensation. Master nodes located at interface midpoints
        (Same process as Krischer 2022)

        Parameters
        ----------
        Kg:
            Guyan-Reduced stiffness matrix

        Returns
        -------
            Kinematic-Condensed stiffness matrix (4x4)

        """
        
        # New nodes will be located at interface midpoints, and their DoFs will be
        # vertical displacement and XY-plane rotation.
        
        # TO DO: adapt the code because for each height of nodes there's several nodes, not just one. Maybe I need another nested for loop everything else should be the same.
        
        # Get relative vertical distances from interface midpoints to interface nodes.
        nDofsOneSide = Kg.shape[0]/2
        nNodesOneSide = nDofsOneSide/3
        y0 = (nNodesOneSide-1)/2 # Half the height of the design domain
        dy = y0 - numpy.arange(0, nNodesOneSide) # Relative node heights (one side)
        dx = 0 # Relative horizontal distances
        
        n_mdofs = 4 # Number of master DoFs
        
        Cm = numpy.zeros((int(Kg.shape[0]), n_mdofs))
        nDoFs_1side = int(Kg.shape[0]/2)
        nNodes = int(Kg.shape[0]/3)
        for i in range(0, nNodes, 2):
            i_dy = int(i/2)
            for j in range(self.nelz+1):
                # Upper half (left side of design domain)
                Cm[j:j+3,:] = [[0, -dy[i_dy], 0, 0],
                               [1,        dx, 0, 0],
                               [0,         0, 0, 0]] # Cols refer to u1, phi1, u2, phi2 (masters) and rows refer to ux, uy, uz (salves)
                
                # Lower half (right side of design domain)
                Cm[j+nDoFs_1side:j+3+nDoFs_1side,:] = [[0, 0, 0, -dy[i_dy]],
                                                       [0, 0, 1,       -dx],
                                                       [0, 0, 0,         0]]
        
        Tr = numpy.vstack((numpy.eye(n_mdofs),Cm))
        
        Kg_ = numpy.zeros((Kg.shape[0]+4,Kg.shape[0]+4))
        Kg_[4:,4:] += Kg 
        Kgr = Tr.T @ Kg_ @ Tr
        Tr = Tr[4:,:]
        # # Rid body modes to test if Kgr*phi = 0
        l = 2 # horizontal length of component (=nelx)
        phi1 = numpy.array([1, 0, 1, 0]).reshape((4,1))
        phi2 = numpy.array([-1, 2/l, 1, 2/l]).reshape((4,1))
        f1 = numpy.dot(Kgr,phi1) # both modes should give zero forces
        f2 = numpy.dot(Kgr,phi2)
        
        # Test if equations from Kri2021 hold
        K11 = Kgr[0,0]
        K22 = Kgr[1,1]
        K44 = Kgr[3,3]
        K33d = K11 - Kgr[0,0]
        K31d = -K11 - (-Kgr[0,0])
        K21d = (K11*l**2+K22-K44)/(2*l) - Kgr[1,0]
        K41d = (K11*l**2-K22+K44)/(2*l) - Kgr[3,0]
        K32d = -(K11*l**2+K22-K44)/(2*l) - Kgr[2,1]
        K42d = (K11*l**2-K22-K44)/(2) - Kgr[3,1]
        K43d = -(K11*l**2-K22+K44)/(2*l) - Kgr[3,2]
        
        return Tr, Kgr


class ComplianceProblem2(ElasticityProblem2):
    r"""
    Topology optimization problem to minimize compliance.

    :math:`\begin{aligned}
    \min_{\boldsymbol{\rho}} \quad & \mathbf{f}^T\mathbf{u}\\
    \textrm{subject to}: \quad & \mathbf{K}\mathbf{u} = \mathbf{f}\\
    & \sum_{e=1}^N v_e\rho_e \leq V_\text{frac},
    \quad 0 < \rho_\min \leq \rho_e \leq 1\\
    \end{aligned}`

    where :math:`\mathbf{f}` are the forces, :math:`\mathbf{u}` are the \
    displacements, :math:`\mathbf{K}` is the striffness matrix, and :math:`V`
    is the volume.
    """
    def __init__(self, bc: BoundaryConditions, penalty: float, volfrac: float, filter: Filter, 
                 constraints: numpy.ndarray, constraints_f, gui, domain_lens=None, joint_locs=None):
        
        if domain_lens==None:
            # Element physical lengths in x and y
            self.elen_x = 1.0
            self.elen_y = 1.0
        else:
            # Element physical lengths in x and y
            self.elen_x = domain_lens[0]/bc.nelx
            self.elen_y = domain_lens[1]/bc.nely
        
        # # Locations of the interface joints with respect to interface midpoint (Don't think I need it here)
        # self.joint_locx = joint_locs[0]
        # self.joint_locy = joint_locs[1]
        
        super().__init__(bc, penalty, volfrac, filter, constraints, constraints_f, gui)
        
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
        start = time.time()
        print(self.iter)
        self.iter += 1
        # Filter design variables
        self.filter_variables(x)
        # self.xPhys = x.copy()

        # Objective and sensitivity
        obj = self.compute_compliance(self.xPhys, dobj)

        # Sensitivity filtering
        self.filter.filter_objective_sensitivities(self.xPhys, dobj)

        # Display physical variables
        if self.gui != 0:
            self.gui.update(self.xPhys)

        print('Compliance Energy: ', obj, '\n')
        print('elapsed_time', time.time() - start)
        return obj


    def constraints_function(
            self, result, x: numpy.ndarray, grad: numpy.ndarray) -> float:
        """
        Compute m problem constraints and their gradients
        https://nlopt.readthedocs.io/en/latest/NLopt_Python_Reference/#vector-valued-constraints

        If the argument grad is not empty (which is the case for MMA),
        then grad is a 2d NumPy array of size m×n (m = number of constraints
        n = number of design variables) which should (upon return)
        be set in-place to the gradient of the function with
        respect to the optimization parameters at x. [from nlopt docs]

        Parameters
        ----------
        result:
            The design variables.
        grad:
            The gradient of the nonlinear constraint wrt the design variables

        Returns
        -------
        float
            the constraint value(s) and derivatives

        """
        result[0] = self.compute_volume(x, grad[0, :]) - self.volfrac * x.size

class ComplianceProblem3(ElasticityProblem3):
    r"""
    Topology optimization problem to minimize compliance.

    :math:`\begin{aligned}
    \min_{\boldsymbol{\rho}} \quad & \mathbf{f}^T\mathbf{u}\\
    \textrm{subject to}: \quad & \mathbf{K}\mathbf{u} = \mathbf{f}\\
    & \sum_{e=1}^N v_e\rho_e \leq V_\text{frac},
    \quad 0 < \rho_\min \leq \rho_e \leq 1\\
    \end{aligned}`

    where :math:`\mathbf{f}` are the forces, :math:`\mathbf{u}` are the \
    displacements, :math:`\mathbf{K}` is the striffness matrix, and :math:`V`
    is the volume.
    """

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
        start = time.time()

        # make sure x stays symmetrical with respect to xy and xz-plane
        if self.nelz > 1 and self.nely > 1:
            x = numpy.reshape(x, (self.nelx, self.nelz, self.nely), order='C')
            x_flipped_0 = x[:, int(self.nelz / 2):, :]
            x_flipped = numpy.flip(x_flipped_0, axis=1)
            x_flipped = numpy.flip(x_flipped, axis=2)
            x = numpy.hstack((x_flipped, x_flipped_0))

        # output_folder = "output"
        # output_file = os.path.join(output_folder, f"output_{self.iter}.mat")
        # scipy.io.savemat(output_file, {'data': x})

        x = numpy.reshape(x, x.size, order='C')




        print(self.iter)
        self.iter += 1

        # Filter design variables
        # self.filter_variables(x)
        self.xPhys = x.copy()

        # Objective and sensitivity
        obj = self.compute_compliance(self.xPhys, dobj)

        # Sensitivity filtering
        self.filter.filter_objective_sensitivities(self.xPhys, dobj)

        print('Displacement: ', obj, '\n')
        print('elapsed_time', time.time() - start)
        return obj



    def constraints_function(
            self, result, x: numpy.ndarray, grad: numpy.ndarray) -> float:
        """
        Compute m problem constraints and their gradients
        https://nlopt.readthedocs.io/en/latest/NLopt_Python_Reference/#vector-valued-constraints

        If the argument grad is not empty (which is the case for MMA),
        then grad is a 2d NumPy array of size m×n (m = number of constraints
        n = number of design variables) which should (upon return)
        be set in-place to the gradient of the function with
        respect to the optimization parameters at x. [from nlopt docs]

        Parameters
        ----------
        result:
            The design variables.
        grad:
            The gradient of the nonlinear constraint wrt the design variables

        Returns
        -------
        float
            the constraint value(s) and derivatives

        """

        result[0] = self.compute_volume(x, grad[0, :]) - self.volfrac * x.size

class MinMassProblem2(ElasticityProblem2):
    def objective_function(
            self, x: numpy.ndarray, grad: numpy.ndarray) -> float:
        """
        Compute m problem constraints and their gradients
        https://nlopt.readthedocs.io/en/latest/NLopt_Python_Reference/#vector-valued-constraints

        If the argument grad is not empty (which is the case for MMA),
        then grad is a 2d NumPy array of size m×n (m = number of constraints
        n = number of design variables) which should (upon return)
        be set in-place to the gradient of the function with
        respect to the optimization parameters at x. [from nlopt docs]

        Parameters
        ----------
        result:
            The design variables.
        grad:
            The gradient of the nonlinear constraint wrt the design variables

        Returns
        -------
        float
            the constraint value(s) and derivatives

        """
        return self.compute_volume(x, grad[:])
    def constraints_function(
            self, result, x: numpy.ndarray, grad: numpy.ndarray) -> float:

        print('Iteration:', self.iter)
        self.iter += 1
        if self.gui != 0:
            self.gui.update(self.xPhys)
        self.init = 1
        for i in range(len(self.constraints)):
            self.f[:2] = self.f[:2] * 0
            self.f[:2] = self.constraint_f[i]

            result[i] = self.compute_constraint(x, grad[i, :]) - self.constraints[i]

            self.init = 0

        print(result)


    def compute_constraint(
            self, x: numpy.ndarray, dc: numpy.ndarray) -> float:
        """
        Compute the objective value and gradient.

        Parameters
        ----------
        x:
            The design variables for which to compute the objective.
        grad:
            The gradient of the objective to compute.

        Returns
        -------
        float
            The objective value.

        """
        # Filter design variables
        # self.filter_variables(x)
        self.xPhys = x.copy()

        # calculate displacement for force/moment on each master dof -> combinations can be calculated quickly
        if self.init == 1 and len(self.constraints) > 2:
            self.compute_displacements_predef(self.xPhys)

        # Objective and sensitivity
        if len(self.constraints) > 2:
            obj = self.compute_compliance_predef(self.xPhys, dc)
        else:
            obj = self.compute_compliance(self.xPhys, dc)

        # Sensitivity filtering
        self.filter.filter_objective_sensitivities(self.xPhys, dc)
        return obj


    def compute_compliance_predef(
            self, xPhys: numpy.ndarray, dc: numpy.ndarray) -> float:
        r"""
        Compute compliance and its gradient.

        The objective is :math:`\mathbf{f}^{T} \mathbf{u}`. The gradient of
        the objective is

        :math:`\begin{align}
        \mathbf{f}^T\mathbf{u} &= \mathbf{f}^T\mathbf{u} -
        \boldsymbol{\lambda}^T(\mathbf{K}\mathbf{u} - \mathbf{f})\\
        \frac{\partial}{\partial \rho_e}(\mathbf{f}^T\mathbf{u}) &=
        (\mathbf{K}\boldsymbol{\lambda} - \mathbf{f})^T
        \frac{\partial \mathbf u}{\partial \rho_e} +
        \boldsymbol{\lambda}^T\frac{\partial \mathbf K}{\partial \rho_e}
        \mathbf{u}
        = \mathbf{u}^T\frac{\partial \mathbf K}{\partial \rho_e}\mathbf{u}
        \end{align}`

        where :math:`\boldsymbol{\lambda} = \mathbf{u}`.

        Parameters
        ----------
        xPhys:
            The element densities.
        dc:
            The gradient of compliance.

        Returns
        -------
        float
            The compliance value.

        """
        # calculate displacement for force/moment combination from precalculated displacements
        self.u = self.u_predef @ self.f[[0, 1]]

        c = 0.0
        dc[:] = 0.0
        dE = numpy.empty(xPhys.shape)
        E = self.compute_young_moduli(xPhys, dE)
        for i in range(self.nloads):
            ui = self.u[:, i][self.edofMat].reshape(-1, 8)
            self.obje[:] = (ui @ self.KE * ui).sum(1)
            c += (E * self.obje).sum()
            dc[:] += -dE * self.obje
        dc /= float(self.nloads)
        return c

    def compute_displacements_predef(self, xPhys: numpy.ndarray) -> numpy.ndarray:
        r"""
                calculate displacement for force/moment on each master dof -> combinations can be calculated quickly
                stored in self.u_predef

                Parameters
                ----------
                xPhys:
                    The element densities.

                """
        def plot_quadratic_functions(a1, b1, c1, a2, b2, c2):
            r"""
                visualize optimization of compliance matrix
                parabola represents constraints/upper limit for compliance energy for load cases

                Parameters
                ----------
                a1, b1, c1 = c_phiz_phiz, c_y_phiz, c_yy -> desired values
                a2, b2, c2 = c_phiz_phiz, c_y_phiz, c_yy -> actual values

                """
            import matplotlib.pyplot as plt
            x = np.linspace(-2, 2, 100)
            y1 = a1 * x ** 2 + b1 * x + c1
            y2 = a2 * x ** 2 + b2 * x + c2

            if not(hasattr(self, 'fig')):
                self.fig, (self.ax1) = plt.subplots(1, 1, figsize=(10, 5))
            self.ax1.cla()

            self.ax1.plot(x, y1)
            self.ax1.plot(x, y2)
            self.ax1.set_xlabel('M_z')
            self.ax1.set_ylabel('Compliance Energy')
            self.ax1.set_title('xy-Plane (F_y = 1)')
            self.ax1.grid(True)

            plt.show(block=False)
            plt.pause(0.1)

        self.u_predef = np.zeros((len(self.u), 2))
        F = self.f.copy()
        self.f = numpy.zeros((len(self.f), 2))
        self.f[0, 0] = 1
        self.f[1, 1] = 1
        U, u_m = self.compute_displacements(xPhys)
        self.u_predef = U
        C_red = u_m

        self.f = F
        if hasattr(self, 'C_desired_y'):
            plot_quadratic_functions(self.C_desired_y[1, 1], 2 * self.C_desired_y[0, 1], self.C_desired_y[0, 0],
                                 C_red[1, 1], 2 * C_red[1, 0], C_red[0, 0])

class MinMassProblem3(ElasticityProblem3):
    def objective_function(
            self, x: numpy.ndarray, grad: numpy.ndarray) -> float:
        """
        Compute m problem constraints and their gradients
        https://nlopt.readthedocs.io/en/latest/NLopt_Python_Reference/#vector-valued-constraints

        If the argument grad is not empty (which is the case for MMA),
        then grad is a 2d NumPy array of size m×n (m = number of constraints
        n = number of design variables) which should (upon return)
        be set in-place to the gradient of the function with
        respect to the optimization parameters at x. [from nlopt docs]

        Parameters
        ----------
        result:
            The design variables.
        grad:
            The gradient of the nonlinear constraint wrt the design variables

        Returns
        -------
        float
            the constraint value(s) and derivatives

        """
        return self.compute_volume(x, grad[:])
    def constraints_function(
            self, result, x: numpy.ndarray, grad: numpy.ndarray) -> float:

        print('Iteration:', self.iter)
        self.iter += 1
        self.init = 1
        for i in range(len(self.constraints)):
            self.f[:6] = self.f[:6] * 0
            self.f[:6] = self.constraint_f[i]

            result[i] = self.compute_constraint(x, grad[i, :]) - self.constraints[i]

            self.init = 0

        print(result)

    def compute_constraint(
            self, x: numpy.ndarray, dc: numpy.ndarray) -> float:
        """
        Compute the objective value and gradient.

        Parameters
        ----------
        x:
            The design variables for which to compute the objective.
        grad:
            The gradient of the objective to compute.

        Returns
        -------
        float
            The objective value.

        """
        # make sure x stays symmetrical with respect to xy and xz-plane
        if self.nelz > 1 and self.nely > 1:
            x = numpy.reshape(x, (self.nelx, self.nelz, self.nely), order='C')
            x_flipped_0 = x[:, int(self.nelz / 2):, :]
            x_flipped = numpy.flip(x_flipped_0, axis=1)
            x_flipped = numpy.flip(x_flipped, axis=2)
            x = numpy.hstack((x_flipped, x_flipped_0))
            x = numpy.reshape(x, x.size, order='C')

        # Filter design variables
        # self.filter_variables(x)
        self.xPhys = x.copy()

        # calculate displacement for force/moment on each master dof -> combinations can be calculated quickly
        if self.init == 1 and len(self.constraints) > 4:
            self.compute_displacements_predef(self.xPhys)

        # Objective and sensitivity
        if len(self.constraints) > 4:
            obj = self.compute_compliance_predef(self.xPhys, dc)
        else:
            obj = self.compute_compliance(self.xPhys, dc)

        # Sensitivity filtering
        self.filter.filter_objective_sensitivities(self.xPhys, dc)
        return obj


    def compute_compliance_predef(
            self, xPhys: numpy.ndarray, dc: numpy.ndarray) -> float:
        r"""
        Compute compliance and its gradient.

        The objective is :math:`\mathbf{f}^{T} \mathbf{u}`. The gradient of
        the objective is

        :math:`\begin{align}
        \mathbf{f}^T\mathbf{u} &= \mathbf{f}^T\mathbf{u} -
        \boldsymbol{\lambda}^T(\mathbf{K}\mathbf{u} - \mathbf{f})\\
        \frac{\partial}{\partial \rho_e}(\mathbf{f}^T\mathbf{u}) &=
        (\mathbf{K}\boldsymbol{\lambda} - \mathbf{f})^T
        \frac{\partial \mathbf u}{\partial \rho_e} +
        \boldsymbol{\lambda}^T\frac{\partial \mathbf K}{\partial \rho_e}
        \mathbf{u}
        = \mathbf{u}^T\frac{\partial \mathbf K}{\partial \rho_e}\mathbf{u}
        \end{align}`

        where :math:`\boldsymbol{\lambda} = \mathbf{u}`.

        Parameters
        ----------
        xPhys:
            The element densities.
        dc:
            The gradient of compliance.

        Returns
        -------
        float
            The compliance value.

        """
        # calculate displacement for force/moment combination from precalculated displacements
        self.u = self.u_predef @ self.f[[1, 2, 4, 5]]

        c = 0.0
        dc[:] = 0.0
        dE = numpy.empty(xPhys.shape)
        E = self.compute_young_moduli(xPhys, dE)
        for i in range(self.nloads):
            ui = self.u[:, i][self.edofMat].reshape(-1, 24)
            self.obje[:] = (ui @ self.KE * ui).sum(1)
            c += (E * self.obje).sum()
            dc[:] += -dE * self.obje
        dc /= float(self.nloads)
        return c

    def compute_displacements_predef(self, xPhys: numpy.ndarray) -> numpy.ndarray:
        r"""
                        calculate displacement for force/moment on each master dof -> combinations can be calculated quickly
                        stored in self.u_predef

                        Parameters
                        ----------
                        xPhys:
                            The element densities.

                        """

        def plot_quadratic_functions(a1, b1, c1, a2, b2, c2, a3, b3, c3, a4, b4, c4):
            r"""
                visualize optimization of compliance matrix
                parabola represents constraints/upper limit for compliance energy for load cases

                Parameters
                ----------
                a1, b1, c1 = c_phiz_phiz, c_y_phiz, c_yy -> desired values
                a2, b2, c2 = c_phiz_phiz, c_y_phiz, c_yy -> actual values
                a3, b3, c3 = c_phiy_phiy, c_z_phiy, c_zz -> desired values
                a4, b4, c4 = c_phiy_phiy, c_z_phiy, c_zz -> actual values

                """
            import matplotlib.pyplot as plt
            x = np.linspace(-2, 2, 100)
            y1 = a1 * x ** 2 + b1 * x + c1
            y2 = a2 * x ** 2 + b2 * x + c2

            y3 = a3 * x ** 2 + b3 * x + c3
            y4 = a4 * x ** 2 + b4 * x + c4

            if not(hasattr(self, 'fig')):
                self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5))
            self.ax1.cla()
            self.ax2.cla()

            self.ax1.plot(x, y1)
            self.ax1.plot(x, y2)
            self.ax1.set_xlabel('M_z')
            self.ax1.set_ylabel('Compliance Energy')
            self.ax1.set_title('xy-Plane (F_y = 1)')
            self.ax1.grid(True)

            self.ax2.plot(x, y3)
            self.ax2.plot(x, y4)
            self.ax2.set_xlabel('M_y')
            # self.ax2.set_ylabel('Compliance Energy')
            self.ax2.set_title('xz-Plane (F_z = 1)')
            self.ax2.grid(True)

            plt.show(block=False)
            plt.pause(0.1)

        C_red = numpy.zeros((6, 6))
        F = self.f[:6].copy()
        self.u_predef = np.zeros((len(self.u), 4))
        m = 0
        for i in [1, 2, 4, 5]:
            self.f[0:6] = 0 * self.f[0:6]
            self.f[i] = 1
            U, u_m = self.compute_displacements(xPhys)
            self.u_predef[:, m] = U[:, 0]
            C_red[:, i] = u_m
            m = m + 1

        if hasattr(self, 'C_desired_y') and hasattr(self, 'C_desired_z'):
            plot_quadratic_functions(self.C_desired_y[1, 1], 2 * self.C_desired_y[0, 1], self.C_desired_y[0, 0],
                                 C_red[5, 5], 2 * C_red[1, 5], C_red[1, 1],
                                 self.C_desired_z[1, 1], 2 * self.C_desired_z[0, 1], self.C_desired_z[0, 0],
                                 C_red[4, 4], 2 * C_red[2, 4], C_red[2, 2])



        self.f[:6] = F

class MinMassRedKentries2(ElasticityProblem2):
    r"""
    Topology optimization problem to minimize mass given [4x4] reduced stiffness matrix entries
    """
    def __init__(self, bc: BoundaryConditions, penalty: float, volfrac: float, filter: Filter, 
                 constraints: numpy.ndarray, constraints_f, gui, domain_lens, joint_locs):
        
        # Element physical lengths in x and y
        self.elen_x = domain_lens[0]/bc.nelx
        self.elen_y = domain_lens[1]/bc.nely
        
        # Locations of the interface joints with respect to interface midpoint
        self.joint_locx = joint_locs[0]
        self.joint_locy = joint_locs[1]
        
        super().__init__(bc, penalty, volfrac, filter, constraints, constraints_f, gui)
        
        self.Kreq = constraints[0:3] # Required stiffness entries (k00, k11, k33)
        self.keepDoFIds, self.delDoFIds = self.buildStaticCondensationIndices()
        self.Tr = self.KinematicCondensationInterfaces() # Transformation matrix for kinematic condensation (stays constant for all iterations)     
        
    def buildStaticCondensationIndices(self):
        
        ############ Static condensation
        # Find slave & master indexes -> master indexes are on the left (x=0) and right (x=nelx) boundaries
        boundCoordsX = numpy.concatenate( (numpy.tile(0,self.nely+1),   numpy.tile(self.nelx,self.nely+1)) )
        boundCoordsY = numpy.concatenate( (numpy.arange(0,self.nely+1), numpy.arange(0,self.nely+1))  )
        
        nodeIds = xy_to_id(boundCoordsX, boundCoordsY, self.nelx, self.nely)
        keepDoFIds  = numpy.concatenate( (2*nodeIds,2*nodeIds+1) )
        keepDoFIds.sort() # Since the nodeIds are sorted, but this gets lost when concatenating
        ndofs = 2 * (self.nelx + 1) * (self.nely + 1)
        allDoFIds = numpy.arange(ndofs)
        delDoFIds = allDoFIds[ ~numpy.isin(allDoFIds, keepDoFIds) ] # DoFs that won't be kept
        return keepDoFIds, delDoFIds
    
    def objective_function(
            self, x: numpy.ndarray, grad: numpy.ndarray) -> float:
        """
        
        Computes volume and sets derivatives in-place
        
        Parameters
        ----------
        result:
            The design variables.
        grad:
            The gradient of obj wrt the design variables

        Returns
        -------
        float
            Volume

        """
        return self.compute_volume(x, grad[:])

    def constraints_function(self, result, x, grad):
        """
        Compute m problem constraints and their gradients
        
        Parameters
        ----------
        result:
            The value of the nonlinear constraints (to be set in-place)
        x:
            The design variables.
        grad:
            The gradient of the nonlinear constraints (to be set in-place)
    
        """
        # Generate xPhys
        self.filter_variables(x)
        
        if self.gui != 0:
            self.gui.update(self.xPhys)
        
        K = self.build_K(self.xPhys,remove_constrained=False).tocsr()
        
        # t0_Kgr = time.perf_counter()
        
        Tg = self.StaticCondensationInterfaces(K) # Guyan (Static condensation)
        Tr = self.Tr # Kinematic condensation
        Tgr = Tg @ Tr
        Tgr_t = Tgr.transpose()
        Kgr = Tgr_t @ K @ Tgr
        
        # print(f"{numpy.round([Kgr[0,0], Kgr[1,1], Kgr[3,3]])}")

        K00, K11, K33 = self.Kreq[:]
        
        # t1_Kgr = time.perf_counter()
        # print(f'Constraint computation time: {t1_Kgr-t0_Kgr} second(s)')
        
        # Tgr = Tgr.toarray()
        # Tgr_t = Tgr_t.toarray()
        
        # t0_grad = time.perf_counter()
        
        eps = 1e-2
        result[:] = [(K00 - Kgr[0,0])/K00-eps, (K11 - Kgr[1,1])/K11-eps, (K33 - Kgr[3,3])/K33-eps, (Kgr[0,0]-K00)/K00-eps, (Kgr[1,1]-K11)/K11-eps, (Kgr[3,3]-K33)/K33-eps]
        # result[:] = [(K00 - Kgr[0,0])/K00, (K11 - Kgr[1,1])/K11, (K33 - Kgr[3,3])/K33]
        # print(result[0:3]*100)
        # print([Kgr[0,0], Kgr[1,1], Kgr[3,3]])
        # Gradients
        if grad.size > 0 :
            grad[:] = 0
            dE = numpy.empty(self.xPhys.shape)
            self.compute_young_moduli(self.xPhys, dE) # here dE gets allocated
            
            # I tried to vectorize the for loop but scipy does not allow 3D sparse arrays
            # iK = numpy.kron(self.edofMat, numpy.ones((8, 1))).flatten()
            # jK = numpy.kron(self.edofMat, numpy.ones((1, 8))).flatten()
            # kk = numpy.kron(numpy.arange(self.nel), numpy.ones((1,8*8))).flatten() # This index has 64 entries for each "slice" of the final dK matrix corresponding to each element
            # KE_resh = self.KE[:,:,numpy.newaxis]
            # dK = scipy.sparse.csr_array(((dE*KE_resh).flatten(), (iK, jK, kk)), shape=(self.ndof,self.ndof,self.nel)) # sparse format
            # dKgr = Tgr_t @ dK @ Tgr
            # grad[:,i] = [-dKgr[0,0]/K00, -dKgr[1,1]/K11, -dKgr[3,3]/K33, dKgr[0,0]/K00, dKgr[1,1]/K11, dKgr[3,3]/K33]  
            # if i%1000 == 0:
                # print(i) 
            
            for i in range(self.nel):
                
                edof_i = self.edofMat[i,:]
                
                iK = numpy.kron(edof_i, numpy.ones((8, 1))).flatten()
                jK = numpy.kron(edof_i, numpy.ones((1, 8))).flatten()
                dK = scipy.sparse.csr_array((dE[i]*self.KE.flatten(), (iK, jK)), shape=(self.ndof,self.ndof)) # sparse format
                dKgr = Tgr_t @ dK @ Tgr
                grad[:,i] = [-dKgr[0,0]/K00, -dKgr[1,1]/K11, -dKgr[3,3]/K33, dKgr[0,0]/K00, dKgr[1,1]/K11, dKgr[3,3]/K33]  
        
        # t1_grad = time.perf_counter()        
        # print(f'Gradient computation time: {t1_grad-t0_grad} second(s)')

        self.filter.filter_constraint_sensitivities(self.xPhys, grad)

class MinMassRedKentries3(ElasticityProblem3):
    r"""
    Topology optimization problem to minimize mass given [4x4] reduced stiffness matrix entries
    """
    def __init__(self, bc: BoundaryConditions, penalty: float, volfrac: float, filter: Filter, constraints: numpy.ndarray, constraints_f, gui):
        super().__init__(bc, penalty, volfrac, filter, constraints, constraints_f, gui)
        self.Kreq = constraints[0:3] # Required stiffness entries (k00, k11, k33)
        
    def objective_function(
            self, x: numpy.ndarray, grad: numpy.ndarray) -> float:
        """
        
        Computes volume and sets derivatives in-place
        
        Parameters
        ----------
        result:
            The design variables.
        grad:
            The gradient of obj wrt the design variables

        Returns
        -------
        float
            Volume

        """
        return self.compute_volume(x, grad[:])

    def constraints_function(self, result, x, grad):
        """
        Compute m problem constraints and their gradients
        
        Parameters
        ----------
        result:
            The value of the nonlinear constraints (to be set in-place)
        x:
            The design variables.
        grad:
            The gradient of the nonlinear constraints (to be set in-place)
    
        """
        # Generate xPhys
        self.filter_variables(x)
        
        # No gui in 3D ?
        # if self.gui != 0:
        #     self.gui.update(self.xPhys)
        
        K, K_y, K_z, fixed_y, fixed_z = self.build_K(self.xPhys,remove_constrained=False)
        
        Tg, Kg = self.StaticCondensationInterfaces(K, self.nelx, self.nely, self.nelz) # Guyan (Static condensation)
        
        Tr, Kgr = self.KinematicCondensationInterfaces(Kg) # Kinematic condensation

        K00, K11, K33 = self.Kreq[:]
        eps = 1e-2
        result[:] = [(K00 - Kgr[0,0])/K00-eps, (K11 - Kgr[1,1])/K11-eps, (K33 - Kgr[3,3])/K33-eps, (Kgr[0,0]-K00)/K00-eps, (Kgr[1,1]-K11)/K11-eps, (Kgr[3,3]-K33)/K33-eps]
        # result[:] = [(K00 - Kgr[0,0])/K00, (K11 - Kgr[1,1])/K11, (K33 - Kgr[3,3])/K33]
        # print(result)
        # print([Kgr[0,0], Kgr[1,1], Kgr[3,3]])
        # Gradients
        Tgr = Tg @ Tr
        if grad.size > 0 :
            grad[:] = 0
            dE = numpy.empty(self.xPhys.shape)
            self.compute_young_moduli(self.xPhys, dE) # here dE gets allocated
            for i in range(self.nel):
                
                edof_i = self.edofMat[i,:]
                
                iK = numpy.kron(edof_i, numpy.ones((8, 1))).flatten()
                jK = numpy.kron(edof_i, numpy.ones((1, 8))).flatten()
                dK = scipy.sparse.csc_array((dE[i]*self.KE.flatten(), (iK, jK)), shape=(self.ndof,self.ndof)) # sparse format
                dKgr = Tgr.transpose() @ dK @ Tgr
                grad[:,i] = [-dKgr[0,0]/K00, -dKgr[1,1]/K11, -dKgr[3,3]/K33, dKgr[0,0]/K00, dKgr[1,1]/K11, dKgr[3,3]/K33]  
                # if i%1000 == 0:
                    # print(i) 
             
        self.filter.filter_constraint_sensitivities(self.xPhys, grad)

class HarmonicLoadsProblem(ElasticityProblem2):
    r"""
    Topology optimization problem to minimize dynamic compliance.

    Replaces standard forces with undamped forced vibrations.

    :math:`\begin{aligned}
    \min_{\boldsymbol{\rho}} \quad & \mathbf{f}^T\mathbf{u}\\
    \textrm{subject to}: \quad & \mathbf{S}\mathbf{u} = \mathbf{f}\\
    & \sum_{e=1}^N v_e\rho_e \leq V_\text{frac},
    \quad 0 < \rho_\min \leq \rho_e \leq 1\\
    \end{aligned}`

    where :math:`\mathbf{f}` is the amplitude of the load, :math:`\mathbf{u}`
    is the amplitude of vibration, and :math:`\mathbf{S}` is the system matrix
    (or "dynamic striffness" matrix) defined as

    :math:`\begin{aligned}
    \mathbf{S} = \mathbf{K} - \omega^2\mathbf{M}
    \end{aligned}`

    where :math:`\omega` is the angular frequency of the load, and
    :math:`\mathbf{M}` is the global mass matrix.
    """

    @staticmethod
    def lm(nel: int) -> numpy.ndarray:
        r"""
        Build the element mass matrix.

        :math:`M = \frac{1}{9 \times 4n}\begin{bmatrix}
        4 & 0 & 2 & 0 & 1 & 0 & 2 & 0 \\
        0 & 4 & 0 & 2 & 0 & 1 & 0 & 2 \\
        2 & 0 & 4 & 0 & 2 & 0 & 1 & 0 \\
        0 & 2 & 0 & 4 & 0 & 2 & 0 & 1 \\
        1 & 0 & 2 & 0 & 4 & 0 & 2 & 0 \\
        0 & 1 & 0 & 2 & 0 & 4 & 0 & 2 \\
        2 & 0 & 1 & 0 & 2 & 0 & 4 & 0 \\
        0 & 2 & 0 & 1 & 0 & 2 & 0 & 4
        \end{bmatrix}`

        Where :math:`n` is the total number of elements. The total mass is
        equal to unity.

        Parameters
        ----------
        nel:
            The total number of elements.

        Returns
        -------
        numpy.ndarray
            The element mass matrix for the material.

        """
        return numpy.array([
            [4, 0, 2, 0, 1, 0, 2, 0],
            [0, 4, 0, 2, 0, 1, 0, 2],
            [2, 0, 4, 0, 2, 0, 1, 0],
            [0, 2, 0, 4, 0, 2, 0, 1],
            [1, 0, 2, 0, 4, 0, 2, 0],
            [0, 1, 0, 2, 0, 4, 0, 2],
            [2, 0, 1, 0, 2, 0, 4, 0],
            [0, 2, 0, 1, 0, 2, 0, 4]], dtype=float) / (36 * nel)

    def __init__(self, bc: BoundaryConditions, penalty: float):
        """
        Create the topology optimization problem.

        Parameters
        ----------
        bc:
            The boundary conditions of the problem.
        penalty:
            The penalty value used to penalize fractional densities in SIMP.

        """
        super().__init__(bc, penalty)
        self.angular_frequency = 0e-2

    def build_indices(self) -> None:
        """Build the index vectors for the finite element coo matrix format."""
        super().build_indices()
        self.ME = self.lm(self.nel)

    def build_M(self, xPhys: numpy.ndarray, remove_constrained: bool = True
                ) -> scipy.sparse.coo_matrix:
        """
        Build the stiffness matrix for the problem.

        Parameters
        ----------
        xPhys:
            The element densisities used to build the stiffness matrix.
        remove_constrained:
            Should the constrained nodes be removed?

        Returns
        -------
        scipy.sparse.coo_matrix
            The stiffness matrix for the mesh.

        """
        # vals = numpy.tile(self.ME.flatten(), xPhys.size)
        vals = (self.ME.reshape(-1, 1) *
                self.penalize_densities(xPhys)).flatten(order='F')
        M = scipy.sparse.coo_matrix((vals, (self.iK, self.jK)),
                                    shape=(self.ndof, self.ndof))
        if remove_constrained:
            # Remove constrained dofs from matrix and convert to coo
            M = deleterowcol(M.tocsc(), self.fixed, self.fixed).tocoo()
        return M

    def compute_displacements(self, xPhys: numpy.ndarray) -> numpy.ndarray:
        r"""
        Compute the amplitude of vibration given the densities.

        Compute the amplitude of vibration, :math:`\mathbf{u}`, using linear
        elastic finite element analysis (solving
        :math:`\mathbf{S}\mathbf{u} = \mathbf{f}` where :math:`\mathbf{S} =
        \mathbf{K} - \omega^2\mathbf{M}` is the system matrix and
        :math:`\mathbf{f}` is the force vector).

        Parameters
        ----------
        xPhys:
            The element densisities used to build the stiffness matrix.

        Returns
        -------
        numpy.ndarray
            The displacements solve using linear elastic finite element
            analysis.

        """
        # Setup and solve FE problem
        K = self.build_K(xPhys)
        M = self.build_M(xPhys)
        S = (K - self.angular_frequency**2 * M).tocoo()
        cvxopt_S = cvxopt.spmatrix(
            S.data, S.row.astype(int), S.col.astype(int))
        # Solve system
        F = cvxopt.matrix(self.f[self.free, :])
        try:
            # F stores solution after solve
            cvxopt.cholmod.linsolve(cvxopt_S, F)
        except Exception:
            F = scipy.sparse.linalg.spsolve(S.tocsc(), self.f[self.free, :])
            F = F.reshape(-1, self.nloads)
        new_u = self.u.copy()
        new_u[self.free, :] = numpy.array(F)[:, :]
        return new_u

    def compute_objective(
            self, xPhys: numpy.ndarray, dobj: numpy.ndarray) -> float:
        r"""
        Compute compliance and its gradient.

        The objective is :math:`\mathbf{f}^{T} \mathbf{u}`. The gradient of
        the objective is

        :math:`\begin{align}
        \mathbf{f}^T\mathbf{u} &= \mathbf{f}^T\mathbf{u} -
        \boldsymbol{\lambda}^T(\mathbf{K}\mathbf{u} - \mathbf{f})\\
        \frac{\partial}{\partial \rho_e}(\mathbf{f}^T\mathbf{u}) &=
        (\mathbf{K}\boldsymbol{\lambda} - \mathbf{f})^T
        \frac{\partial \mathbf u}{\partial \rho_e} +
        \boldsymbol{\lambda}^T\frac{\partial \mathbf K}{\partial \rho_e}
        \mathbf{u}
        = \mathbf{u}^T\frac{\partial \mathbf K}{\partial \rho_e}\mathbf{u}
        \end{align}`

        where :math:`\boldsymbol{\lambda} = \mathbf{u}`.

        Parameters
        ----------
        xPhys:
            The element densities.
        dobj:
            The gradient of compliance.

        Returns
        -------
        float
            The compliance value.

        """
        # Setup and solve FE problem
        self.update_displacements(xPhys)

        obj = 0.0
        dobj[:] = 0.0
        dE = numpy.empty(xPhys.shape)
        E = self.compute_young_moduli(xPhys, dE)
        drho = numpy.empty(xPhys.shape)
        penalty = self.penalty
        self.penalty = 2
        rho = self.penalize_densities(xPhys, drho)
        self.penalty = penalty
        for i in range(self.nloads):
            ui = self.u[:, i][self.edofMat].reshape(-1, 8)
            obje1 = (ui @ self.KE * ui).sum(1)
            obje2 = (ui @ (-self.angular_frequency**2 * self.ME) * ui).sum(1)
            self.obje[:] = obje1 + obje2
            obj += (E * obje1 + rho * obje2).sum()
            dobj[:] += -(dE * obje1 + drho * obje2)
        dobj /= float(self.nloads)
        return obj / float(self.nloads)


class VonMisesStressProblem(ElasticityProblem2):
    """
    Topology optimization problem to minimize stress.

    Todo:
        * Currently this problem minimizes compliance and computes stress on
          the side. This needs to be replaced to match the promise of
          minimizing stress.
    """

    @staticmethod
    def B(side: float) -> numpy.ndarray:
        r"""
        Construct a strain-displacement matrix for a 2D regular grid.

        :math:`B = \frac{1}{2s}\begin{bmatrix}
        1 &  0 & -1 &  0 & -1 &  0 &  1 &  0 \\
        0 &  1 &  0 &  1 &  0 & -1 &  0 & -1 \\
        1 &  1 &  1 & -1 & -1 & -1 & -1 &  1
        \end{bmatrix}`

        where :math:`s` is the side length of the square elements.

        Todo:
            * Check that this is not -B

        Parameters
        ----------
        side:
            The side length of the square elements.

        Returns
        -------
        numpy.ndarray
            The strain-displacement matrix for a 2D regular grid.

        """
        n = -0.5 / side
        p = 0.5 / side
        return numpy.array([[p, 0, n, 0, n, 0, p, 0],
                            [0, p, 0, p, 0, n, 0, n],
                            [p, p, p, n, n, n, n, p]])

    @staticmethod
    def E(nu):
        r"""
        Construct a constitutive matrix for a 2D regular grid.

        :math:`E = \frac{1}{1 - \nu^2}\begin{bmatrix}
        1 & \nu & 0 \\
        \nu & 1 & 0 \\
        0 & 0 & \frac{1 - \nu}{2}
        \end{bmatrix}`

        Parameters
        ----------
        nu:
            The Poisson's ratio of the material.

        Returns
        -------
        numpy.ndarray
            The constitutive matrix for a 2D regular grid.

        """
        return numpy.array([[1, nu, 0],
                            [nu, 1, 0],
                            [0, 0, (1 - nu) / 2.]]) / (1 - nu**2)

    def __init__(self, nelx, nely, penalty, bc, side=1):
        super().__init__(bc, penalty)
        self.EB = self.E(self.nu) @ self.B(side)
        self.du = numpy.zeros((self.ndof, self.nel * self.nloads))
        self.stress = numpy.zeros(self.nel)
        self.dstress = numpy.zeros(self.nel)

    def build_dK0(self, drho_xi, i, remove_constrained=True):
        sK = ((self.KE.flatten()[numpy.newaxis]).T * drho_xi).flatten(
            order='F')
        iK = self.iK[64 * i: 64 * i + 64]
        jK = self.jK[64 * i: 64 * i + 64]
        dK = scipy.sparse.coo_matrix(
            (sK, (iK, jK)), shape=(self.ndof, self.ndof))
        # Remove constrained dofs from matrix and convert to coo
        if remove_constrained:
            dK = deleterowcol(dK.tocsc(), self.fixed, self.fixed).tocoo()
        return dK

    def build_dK(self, xPhys, remove_constrained=True):
        drho = numpy.empty(xPhys.shape)
        self.compute_young_moduli(xPhys, drho)
        blocks = [self.build_dK0(drho[i], i, remove_constrained)
                  for i in range(drho.shape[0])]
        dK = scipy.sparse.block_diag(blocks, format="coo")
        return dK

    @staticmethod
    def sigma_pow(s11: numpy.ndarray, s22: numpy.ndarray, s12: numpy.ndarray,
                  p: float) -> numpy.ndarray:
        r"""
        Compute the von Mises stress raised to the :math:`p^{\text{th}}` power.

        :math:`\sigma^p = \left(\sqrt{\sigma_{11}^2 - \sigma_{11}\sigma_{22} +
        \sigma_{22}^2 + 3\sigma_{12}^2}\right)^p`

        Todo:
            * Properly document what the sigma variables represent.
            * Rename the sigma variables to something more readable.

        Parameters
        ----------
        s11:
            :math:`\sigma_{11}`
        s22:
            :math:`\sigma_{22}`
        s12:
            :math:`\sigma_{12}`
        p:
            The power (:math:`p`) to raise the von Mises stress.

        Returns
        -------
        numpy.ndarray
            The von Mises stress to the :math:`p^{\text{th}}` power.

        """
        return numpy.sqrt(s11**2 - s11 * s22 + s22**2 + 3 * s12**2)**p

    @staticmethod
    def dsigma_pow(s11: numpy.ndarray, s22: numpy.ndarray, s12: numpy.ndarray,
                   ds11: numpy.ndarray, ds22: numpy.ndarray,
                   ds12: numpy.ndarray, p: float) -> numpy.ndarray:
        r"""
        Compute the gradient of the stress to the :math:`p^{\text{th}}` power.

        :math:`\nabla\sigma^p = \frac{p\sigma^{p-1}}{2\sigma}\nabla(\sigma^2)`

        Todo:
            * Properly document what the sigma variables represent.
            * Rename the sigma variables to something more readable.

        Parameters
        ----------
        s11:
            :math:`\sigma_{11}`
        s22:
            :math:`\sigma_{22}`
        s12:
            :math:`\sigma_{12}`
        ds11:
            :math:`\nabla\sigma_{11}`
        ds22:
            :math:`\nabla\sigma_{22}`
        ds12:
            :math:`\nabla\sigma_{12}`
        p:
            The power (:math:`p`) to raise the von Mises stress.

        Returns
        -------
        numpy.ndarray
            The gradient of the von Mises stress to the :math:`p^{\text{th}}`
            power.

        """
        sigma = numpy.sqrt(s11**2 - s11 * s22 + s22**2 + 3 * s12**2)
        dinside = (2 * s11 * ds11 - s11 * ds22 - ds11 * s22 + 2 * s22 *
                   ds22 + 6 * s12 * ds12)
        return p * (sigma)**(p - 1) / (2.0 * sigma) * dinside

    def compute_stress_objective(self, xPhys, dobj, p=4):
        """Compute stress objective and its gradient."""
        # Setup and solve FE problem
        # self.update_displacements(xPhys)

        rho = self.compute_young_moduli(xPhys)
        EBu = sum([self.EB @ self.u[:, i][self.edofMat.T]
                   for i in range(self.nloads)])
        s11, s22, s12 = numpy.hsplit((EBu * rho / float(self.nloads)).T, 3)
        # Update the stress for plotting
        self.stress[:] = numpy.sqrt(
            s11**2 - s11 * s22 + s22**2 + 3 * s12**2).squeeze()

        obj = self.sigma_pow(s11, s22, s12, p).sum()

        # Setup and solve FE problem
        K = self.build_K(xPhys)
        K = cvxopt.spmatrix(
            K.data, K.row.astype(int), K.col.astype(int))

        # Setup dK @ u
        dK = self.build_dK(xPhys).tocsc()
        U = numpy.tile(self.u[self.free, :], (self.nel, 1))
        dKu = (dK @ U).reshape((-1, self.nel * self.nloads), order="F")

        # Solve system and solve for du: K @ du = dK @ u
        rhs = cvxopt.matrix(dKu)
        cvxopt.cholmod.linsolve(K, rhs)  # rhs stores solution after solve
        self.du[self.free, :] = -numpy.array(rhs)

        du = self.du.reshape((self.ndof * self.nel, self.nloads), order="F")
        rep_edofMat = (numpy.tile(self.edofMat.T, self.nel) + numpy.tile(
            numpy.repeat(numpy.arange(self.nel) * self.ndof, self.nel),
            (8, 1)))
        dEBu = sum([self.EB @ du[:, j][rep_edofMat]
                    for j in range(self.nloads)])
        rhodEBu = numpy.tile(rho, self.nel) * dEBu
        drho = numpy.empty(xPhys.shape)
        self.compute_young_moduli(xPhys, drho)
        drhoEBu = numpy.diag(drho).flatten() * numpy.tile(EBu, self.nel)
        ds11, ds22, ds12 = map(
            lambda x: x.reshape(self.nel, self.nel).T,
            numpy.hsplit(((drhoEBu + rhodEBu) / float(self.nloads)).T, 3))
        dobj[:] = self.dstress[:] = self.dsigma_pow(
            s11, s22, s12, ds11, ds22, ds12, p).sum(0)

        return obj

    def test_calculate_objective(
            self, xPhys: numpy.ndarray, dobj: numpy.ndarray, p: float = 4,
            dx: float = 1e-6) -> float:
        """
        Calculate the gradient of the stresses using finite differences.

        Parameters
        ----------
        xPhys:
            The element densities.
        dobj:
            The gradient of the stresses to compute.
        p:
            The exponent for computing the softmax of the stresses.
        dx:
            The difference in x values used for finite differences.

        Returns
        -------
        float
            The analytic objective value.

        """
        dobja = dobj.copy()  # Analytic gradient
        obja = self.compute_stress_objective(
            xPhys, dobja, p)  # Analytic objective
        dobjf = dobj.copy()  # Finite difference of the stress
        delta = numpy.zeros(xPhys.shape)
        for i in range(xPhys.shape[0]):
            delta[[i - 1, i]] = 0, dx
            self.update_displacements(xPhys + delta)
            s1 = self.compute_stress_objective(xPhys + delta, dobj.copy(), p)
            self.update_displacements(xPhys - delta)
            s2 = self.compute_stress_objective(xPhys - delta, dobj.copy(), p)
            dobjf[i] = ((s1 - s2) / (2. * dx))

        print("Differences: {:g}".format(numpy.linalg.norm(dobjf - dobja)))
        # print("Analytic Norm: {:g}".format(numpy.linalg.norm(ds)))
        # print("Numeric Norm:  {:g}".format(numpy.linalg.norm(dsf)))
        # print("Analytic:\n{:s}".format(ds))
        # print("Numeric:\n{:s}".format(dsf))
        return obja

    def compute_objective(self, xPhys, dobj):
        """Compute compliance and its gradient."""
        obj = ComplianceProblem.compute_objective(self, xPhys, dobj)
        self.compute_stress_objective(xPhys, numpy.zeros(dobj.shape))
        return obj
