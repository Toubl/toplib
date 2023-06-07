"""Graphics user interfaces for topology optimization."""
from __future__ import division

import matplotlib.colors as colors
import matplotlib.cm as colormaps
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy

from topopt.utils import id_to_xy
import plotly.graph_objects as go
import numpy as np

class newGUI(object):
    """
    Graphics user interface of the topology optimization.

    Draws the outputs a topology optimization problem.
    """

    def __init__(self, problem, title=""):
        """
        Create a plot and draw the initial design.

        Args:
            problem (topopt.Problem): problem to visualize
            title (str): title of the plot
        """
        self.problem = problem
        self.title = title
        plt.ion()  # Ensure that redrawing is possible
        self.init_subplots()
        plt.xlabel(title)
        self.plot_force_arrows()
        plt.show()

    def __str__(self):
        """Create a string representation of the solver."""
        return self.__class__.__name__

    def __format__(self, format_spec):
        """Create a formatted representation of the solver."""
        return str(self)
    
    def __repr__(self):
        """Create a representation of the solver."""
        return '{}(problem={!r}, title="{}")'.format(
            self.__class__.__name__, self.problem, self.title)

    def init_subplots(self):
        """Create the subplots."""
        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(
            -numpy.zeros((self.problem.nely, self.problem.nelx)), cmap='plasma',
            interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))
    
    def plot_force_arrows(self):
        """Add arrows to the plot for each force."""
        arrowprops = {"arrowstyle": "->", "connectionstyle": "arc3", "lw": 2,
                    "color": 0}
        nelx, nely, f = (self.problem.nelx, self.problem.nely, self.problem.f)
        cmap = plt.get_cmap("hsv", f.shape[1] + 1)
        for load_i in range(f.shape[1]):
            nz = numpy.nonzero(f[:, load_i])
            arrowprops["color"] = matplotlib.colors.to_hex(cmap(load_i))
            print("Arrow color:", arrowprops["color"])  # Debugging line
            for i in range(nz[0].shape[0]):
                x, y = id_to_xy(nz[0][i] // 2, nelx, nely)
                x = max(min(x, nelx - 1), 0)
                y = max(min(y, nely - 1), 0)
                z = int(nz[0][i] % 2)
                mag = -50 * f[nz[0][i], load_i]
                self.ax.annotate(
                    "", xy=(x, y), xycoords="data",
                    xytext=(0 if z else mag, mag if z else 0),
                    textcoords="offset points", arrowprops=arrowprops)

    
    def update(self, xPhys, title=None):
        """Plot the results."""
        print(xPhys)
        print(isinstance(xPhys[0], float))
        self.im.set_array(
            -xPhys.reshape((self.problem.nelx, self.problem.nely)).T)
        
        print("Type of self.im:", type(self.im))
        print("Values in self.im:", self.im.get_array())
        print(type(self.im.get_array()[0][0]))

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        if title is not None:
            plt.title(title)
        plt.pause(0.01)


class GUI(object):
    """
    Graphics user interface of the topology optimization.

    Draws the outputs a topology optimization problem.
    """

    def __init__(self, problem, title=""):
        """
        Create a plot and draw the initial design.

        Args:
            problem (topopt.Problem): problem to visualize
            title (str): title of the plot
        """
        self.problem = problem
        self.title = title
        plt.ion()  # Ensure that redrawing is possible
        self.init_subplots()
        plt.xlabel(title)
        self.plot_force_arrows()
        plt.show()

    def __str__(self):
        """Create a string representation of the solver."""
        return self.__class__.__name__

    def __format__(self, format_spec):
        """Create a formatted representation of the solver."""
        return str(self)

    def __repr__(self):
        """Create a representation of the solver."""
        return '{}(problem={!r}, title="{}")'.format(
            self.__class__.__name__, self.problem, self.title)

    def init_subplots(self):
        """Create the subplots."""
        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(
            -numpy.zeros((self.problem.nely, self.problem.nelx)), cmap='viridis',
            interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))

    def plot_force_arrows(self):
        """Add arrows to the plot for each force."""
        arrowprops = {"arrowstyle": "->", "connectionstyle": "arc3", "lw": 2,
                      "color": 0}
        nelx, nely, f = (self.problem.nelx, self.problem.nely, self.problem.f)
        cmap = plt.get_cmap("hsv", f.shape[1] + 1)
        for load_i in range(f.shape[1]):
            nz = numpy.nonzero(f[:, load_i])
            arrowprops["color"] = cmap(load_i)
            for i in range(nz[0].shape[0]):
                x, y = id_to_xy(nz[0][i] // 2, nelx, nely)
                x = max(min(x, nelx - 1), 0)
                y = max(min(y, nely - 1), 0)
                z = int(nz[0][i] % 2)
                mag = -50 * f[nz[0][i], load_i]
                self.ax.annotate(
                    "", xy=(x, y), xycoords="data",
                    xytext=(0 if z else mag, mag if z else 0),
                    textcoords="offset points", arrowprops=arrowprops)

    def update(self, xPhys, title=None):
        """Plot the results."""
        self.im.set_array(
            -xPhys.reshape((self.problem.nelx, self.problem.nely)).T)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        if title is not None:
            plt.title(title)
        plt.pause(0.01)


class StressGUI(GUI):
    """
    Graphics user interface of the topology optimization.

    Draws the output, stress, and derivative of stress of the topology
    optimization problem.
    """

    def __init__(self, problem, title=""):
        """Create a plot and draw the initial design."""
        self.mcmap = colormaps.viridis
        self.myColorMap2 = colormaps.ScalarMappable(
            norm=colors.Normalize(vmin=-1, vmax=1), cmap=self.mcmap)
        GUI.__init__(self, problem)
        plt.suptitle(title)
        title_sty = {"boxstyle": "square,pad=0.0", "fc": "white", "ec": "none"}
        self.ax.set_title("Von Mises Stress", bbox=title_sty)
        self.ax2.set_title("Derivative of Stress", bbox=title_sty)

    def init_subplots(self):
        """Create the subplots (one for the stress and one for diff stress)."""
        self.myColorMap = colormaps.ScalarMappable(
            norm=colors.Normalize(vmin=0, vmax=1), cmap=colormaps.plasma)
        self.fig, (self.ax, self.ax2) = plt.subplots(figsize=(8, 4), ncols=2)
        # Generate image for the plots
        self.stress_im = self.ax.imshow(
            numpy.zeros((self.problem.nely, self.problem.nelx, 4)),
            norm=colors.Normalize(vmin=0, vmax=1), cmap='plasma')
        self.stress_d_im = self.ax2.imshow(
            numpy.zeros((self.problem.nely, self.problem.nelx, 4)),
            norm=colors.Normalize(vmin=-1, vmax=1), cmap=self.mcmap)
        self.cbar = self.fig.colorbar(self.stress_im, ax=self.ax)
        self.cbar2 = self.fig.colorbar(self.stress_d_im, ax=self.ax2)

    def update(self, xPhys, title=None):
        """Plot the results."""
        nelx, nely = self.problem.nelx, self.problem.nely

        def values_to_rgba(values, alpha_values, cmap):
            values_rgba = cmap.to_rgba(values)
            values_rgba[:, 3] = alpha_values.astype(float)  # Set alpha by x
            return values_rgba

        # Updated von Mises subplot
        stress = self.problem.stress
        self.myColorMap.set_norm(colors.Normalize(vmin=0, vmax=max(stress)))
        stress_rgba = values_to_rgba(stress, xPhys, self.myColorMap)
        self.stress_im.set_array(
            numpy.swapaxes(stress_rgba.reshape(nelx, nely, 4), 0, 1))
        self.ax.set_xlabel(r"$\sigma_v \in$ [{:.2f}, {:.2f}]".format(
            float(min(stress)), float(max(stress))))

        # Updated derivative of von Mises subplot
        dstress = self.problem.dstress
        max_val = max(abs(dstress))
        self.myColorMap2.set_norm(
            colors.Normalize(vmin=-max_val, vmax=max_val))
        values_rgba = values_to_rgba(dstress, xPhys, self.myColorMap2)
        self.stress_d_im.set_array(
            numpy.swapaxes(values_rgba.reshape(nelx, nely, 4), 0, 1))
        self.ax2.set_xlabel(
            r"$\partial_{{x_e}} \|\sigma_v\|^2 \in $[{:.2f}, {:.2f}]".format(
                float(min(dstress)), float(max(dstress))))

        if title is not None:
            plt.suptitle(title)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

