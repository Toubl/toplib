"""Topology optimization problem to solve."""

import abc

import numpy
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import cvxopt
import cvxopt.cholmod

from .boundary_conditions import BoundaryConditions
from .utils import deleterowcol


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
        # Problem size
        self.nelx = bc.nelx
        self.nely = bc.nely
        self.nelz = bc.nelz
        self.nel = self.nelx * self.nely * self.nelz

        # Count degrees of fredom
        if self.nelz > 1:
            self.ndof = 3 * (self.nelx + 1) * (self.nely + 1) * (self.nelz + 1)
        else:
            self.ndof = 2 * (self.nelx + 1) * (self.nely + 1)

        # SIMP penalty
        self.penalty = penalty

        # BC's and support (half MBB-beam)
        self.bc = bc
        dofs = numpy.arange(self.ndof)
        self.fixed = bc.fixed_nodes
        self.free = numpy.setdiff1d(dofs, self.fixed)

        # RHS and Solution vectors
        self.f = bc.forces
        self.u = numpy.zeros((self.ndof, 1))

        # Per element objective
        self.obje = numpy.zeros(self.nely * self.nelx * self.nelz)

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

    @abc.abstractmethod
    def compute_compliance(
            self, xPhys: numpy.ndarray, dc: numpy.ndarray, c_critical) -> float:
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
    def compute_compliance3(
            self, xPhys: numpy.ndarray, dc: numpy.ndarray, c_critical) -> float:
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
        pass


class ElasticityProblem(Problem):
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
        KE = E / (1 - nu**2) * numpy.array([
            [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
            [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
            [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
            [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
            [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
            [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
            [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
            [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])
        return KE

    @staticmethod
    def lk3(E: float = 1, nu: float = 0.3, length_x: float = 1.0, length_y: float = 1.0, length_z: float = 1.0) -> numpy.ndarray:
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

    def RBE2_interface(self, nelx, nely, K):
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

        # Create an expanded COO matrix
        expanded_row = numpy.concatenate(([0, 1], row + 2))
        expanded_col = numpy.concatenate(([0, 1], col + 2))
        expanded_data = numpy.concatenate(([0, 0], data))
        K_ = scipy.sparse.coo_matrix((expanded_data, (expanded_row, expanded_col)), shape=(n + 2, n + 2))

        if hasattr(self, 'T_r'):
            # Code to execute when self.T_r exists
            K = self.T_r.transpose() @ K_ @ self.T_r
        else:
            # Code to execute when self.T_r does not exist
            alldofs0_r = numpy.arange(0, K_.shape[0])  # All dofs in original order
            sdofs_r = numpy.arange(K_.shape[0] - 2 * (nely + 1), K_.shape[0])  # Dofs that are to be removed
            mdofs_r = numpy.setdiff1d(alldofs0_r, sdofs_r)  # Dofs that remain

            C = scipy.sparse.coo_matrix((len(sdofs_r), K_.shape[1]))
            C = C.toarray()

            for n in range(0, nely + 1):
                C[2 * n, 0] = 0
                C[2 * n + 1, 0] = 1
                C_t = numpy.cross([0, 0, 1], [0, (0.1 - 0.1 * n / nely) - 0.05, 0])
                C[2 * n, 1] = C_t[0]
                C[2 * n + 1, 1] = C_t[1]
                C[2 * n, (nely + 1) * nelx * 2 + 2 * n + 2] = -1
                C[2 * n + 1, (nely + 1) * nelx * 2 + 2 * n + 3] = -1

            Tsm = -numpy.linalg.solve(C[:, sdofs_r], C[:, mdofs_r])
            Ti = numpy.eye(len(mdofs_r))
            self.T_r = scipy.sparse.coo_matrix(numpy.vstack((Ti, Tsm)))
            K = self.T_r.transpose() @ K_ @ self.T_r
        return K
    def RBE3_interface(self, nelx, nely, nelz, K):
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

        # Create an expanded COO matrix
        expanded_row = numpy.concatenate(([0, 1, 2, 3, 4, 5], row + 6))
        expanded_col = numpy.concatenate(([0, 1, 2, 3, 4, 5], col + 6))
        expanded_data = numpy.concatenate(([0, 0, 0, 0, 0, 0], data))
        K_ = scipy.sparse.coo_matrix((expanded_data, (expanded_row, expanded_col)), shape=(n + 6, n + 6))

        if hasattr(self, 'T_r'):
            # Code to execute when self.T_r exists
            K_r = self.T_r.transpose() @ K_ @ self.T_r
        else:
            # Code to execute when self.T_r does not exist
            alldofs_r = numpy.arange(0, K_.shape[0])
            sdofs_r = alldofs_r[-3*(nely+1)*(nelz+1):K_.shape[0]]
            mdofs_r = numpy.setdiff1d(alldofs_r, sdofs_r)

            C = scipy.sparse.coo_matrix((3*(nely+1)*(nelz+1), K_.shape[0]))
            C = C.toarray()

            m = 0
            column_start = -3*(nely+1)*(nelz+1)
            for i in range(0, (nelz + 1)):
                for j in range(0, (nely + 1)):
                    C[3 * m + 0, 0] = 1  # x Transl. DOF of free node --> x Transl. Dof
                    C[3 * m + 1, 1] = 1  # y Transl. DOF of free node --> y Transl. Dof
                    C[3 * m + 2, 2] = 1  # z Transl. DOF of free node --> z Transl. Dof

                    C[3 * m + 1, 3] = nelz/nelx * (0.5 - i/nelz)  # x Rotational DOF of free node --> -y Transl
                    C[3 * m + 2, 3] = nely/nelx * (0.5 - j/nely)  # x Rotational DOF of free node --> z Transl

                    C[3 * m + 0, 4] = -nelz/nelx * (0.5 - i/nelz)  # y Rotational DOF of free node --> x Transl
                    C[3 * m + 2, 4] = 0  # y Rotational DOF of free node --> z Transl

                    C[3 * m + 0, 5] = -nely/nelx * (0.5 - j/nely)  # z Rotational DOF of free node --> x Transl
                    C[3 * m + 1, 5] = 0  # z Rotational DOF of free node --> y Transl

                    C[3 * m + 0, 3 * m + 0 + column_start] = -1  # Slave nodes of 3d elements to be removed
                    C[3 * m + 1, 3 * m + 1 + column_start] = -1  # Slave nodes of 3d elements to be removed
                    C[3 * m + 2, 3 * m + 2 + column_start] = -1  # Slave nodes of 3d elements to be removed

                    m = m + 1
            Tsm = -numpy.linalg.solve(C[:, sdofs_r], C[:, mdofs_r])
            Tsm = scipy.sparse.coo_matrix(Tsm)
            Ti = scipy.sparse.eye(len(mdofs_r))
            self.T_r = scipy.sparse.vstack((Ti, Tsm))
            K_r = self.T_r.transpose() @ K_ @ self.T_r
        return K_r

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
        # Max and min stiffness
        self.Emin = 1e-9
        self.Emax = 1.0

        # FE: Build the index vectors for the for coo matrix format.
        self.nu = 0.3

        # build indices(assignment local to global dof)
        # and calculate element stiffness matrix
        if self.nelz > 1: # 3D case
            self.build_indices3()
        else: # 2D case
            self.build_indices()


        # BC's and support (half MBB-beam)
        self.bc = bc
        dofs = numpy.arange(self.ndof)
        self.fixed = bc.fixed_nodes
        self.free = numpy.setdiff1d(dofs, self.fixed)

        # Number of loads
        self.nloads = self.f.shape[1]

    def build_indices(self) -> None:
        """Build the index vectors for the finite element coo matrix format."""
        self.KE = self.lk(E=self.Emax, nu=self.nu)
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

    def build_indices3(self) -> None:
        """Build the index vectors for the finite element coo matrix format."""
        self.KE = self.lk3(E=self.Emax, nu=self.nu)
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
        # Construct the index pointers for the coo format
        self.iK = numpy.kron(self.edofMat, numpy.ones((24, 1))).flatten()
        self.jK = numpy.kron(self.edofMat, numpy.ones((1, 24))).flatten()

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
        K = self.RBE2_interface(self.nelx, self.nely, K)
        if remove_constrained:
            # Remove constrained dofs from matrix and convert to coo
            #K = deleterowcol(K.tocsc(), self.fixed, self.fixed).tocoo()
            K = deleterowcol(K.tocsc(), self.fixed + 2, self.fixed + 2).tocoo()
        return K
    def build_K3(self, xPhys: numpy.ndarray, remove_constrained: bool = True
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
        K = self.RBE3_interface(self.nelx, self.nely, self.nelz, K)
        if remove_constrained:
            # Remove constrained dofs from matrix and convert to coo
            K = deleterowcol(K.tocsc(), self.fixed + 6, self.fixed + 6).tocoo()
            #K = numpy.delete(numpy.delete(K, self.fixed, axis=0), self.fixed, axis=1)
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
        K = self.build_K(xPhys)
        K = cvxopt.spmatrix(
            K.data, K.row.astype(int), K.col.astype(int))
        # Solve system
        F = numpy.zeros(len(self.f) - 4 * (self.nely + 1) + 2) # Force vector of reduced problem
        F[0] = 1 # F_y
        F[1] = 0.1 # Moment about z-Axis

        F = cvxopt.matrix(F)
        #F = cvxopt.matrix(self.f[self.free, :])
        cvxopt.cholmod.linsolve(K, F)  # F stores solution after solve
        zeros_to_insert = numpy.zeros(len(self.fixed))  # Create an array of zeros
        F = numpy.insert(F, 2, zeros_to_insert)  # Insert zeros of fixed dofs
        F = self.T_r @ F # retransform to inlude slave nodes
        F = F[2:] #delete master node dof
        new_u = self.u.copy()
        #new_u[self.free, :] = numpy.array(F)[:, :]
        new_u[:, 0] = numpy.array(F)[:]
        return new_u

    def compute_displacements3(self, xPhys: numpy.ndarray) -> numpy.ndarray:
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
        K = self.build_K3(xPhys)
        F = self.f  # Force vector of reduced problem
        F = cvxopt.matrix(F)
        K = cvxopt.spmatrix(
            K.data, K.row.astype(int), K.col.astype(int))
        # Solve system
        cvxopt.cholmod.linsolve(K, F)  # F stores solution after solve
        # print('Displacement: ', (self.f * F).sum(), '\n')
        zeros_to_insert = numpy.zeros(len(self.fixed))  # Create an array of zeros
        F = numpy.insert(F, 6, zeros_to_insert)  # Insert zeros of fixed dofs
        F = self.T_r @ F  # retransform to inlude slave nodes
        u_m = F[0:6]
        F = F[6:]  #delete master node dof
        new_u = self.u.copy()
        new_u[:, 0] = numpy.array(F)[:]
        return new_u, u_m
    def update_displacements(self, xPhys: numpy.ndarray) -> None:
        """
        Update the displacements of the problem.

        Parameters
        ----------
        xPhys:
            The element densisities used to compute the displacements.

        """
        self.u[:, :] = self.compute_displacements(xPhys)

    def update_displacements3(self, xPhys: numpy.ndarray) -> None:
        """
        Update the displacements of the problem.

        Parameters
        ----------
        xPhys:
            The element densisities used to compute the displacements.

        """
        self.u[:, :], _ = self.compute_displacements3(xPhys)

class ComplianceProblem(ElasticityProblem):
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

    def compute_compliance(
            self, xPhys: numpy.ndarray, dc: numpy.ndarray, c_critical) -> float:
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
        return c - c_critical
    def compute_compliance3(
            self, xPhys: numpy.ndarray, dc: numpy.ndarray, c_critical) -> float:
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
        self.update_displacements3(xPhys)

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
        print('Displacement: ', c, '\n')
        return c - c_critical
    def compute_reduced_stiffness(self, x_opt: numpy.ndarray):
        return 0
    def compute_reduced_stiffness3(self, x_opt: numpy.ndarray):
        K_red = numpy.zeros((6, 6))
        C_red = numpy.zeros((6, 6))

        print("volfrac:")
        print(x_opt.sum()/(self.nelx * self.nely * self.nelz))
        print("nelx, nely, nelz:")
        print((self.nelx, self.nely, self.nelz), '\n')
        print("F:")
        print(numpy.transpose(self.f[0:6]))
        for i in range(0, 6):
            self.f[0:6] = 0 * self.f[0:6]
            self.f[i] = 1
            _, U = self.compute_displacements3(x_opt)
            C_red[:, i] = numpy.transpose(U)
        K_red = numpy.linalg.inv(C_red)

        # Calculate the threshold for cutting off entries
        threshold = numpy.max(np.abs(K_red)) * 1e-6

        # Apply the threshold and update the array
        K_red = numpy.where(np.abs(K_red) >= threshold, K_red, 0)

        # Calculate the threshold for cutting off entries
        threshold = numpy.max(np.abs(C_red)) * 1e-6

        # Apply the threshold and update the array
        C_red = numpy.where(np.abs(C_red) >= threshold, C_red, 0)

        numpy.set_printoptions(precision=4)

        print("C_red:")
        print(C_red, '\n')
        print("K_red:")
        print(K_red, '\n')
        return K_red
class HarmonicLoadsProblem(ElasticityProblem):
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


class VonMisesStressProblem(ElasticityProblem):
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
