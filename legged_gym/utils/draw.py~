import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Create a figure and an axes object for the 3D plot
class Sphere():
    def __init__(self, radius, pos):
        self.fig=plt.figure()
        self.ax=self.fig.add_subplot(111, projection='3d')

        self._x, self._y, self._z = pos[0], pos[1], pos[2]  # Coordinates of the center of the sphere
        self.radius = radius  # Radius of the sphere

    def draw(self):
        # Create the data for the sphere
        phi, theta = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j] # Define angles
        x = self.radius * np.sin(theta) * np.cos(phi) + self._x
        y = self.radius * np.sin(theta) * np.sin(phi) + self._y
        z = self.radius * np.cos(theta) + self._z

        # Plot the sphere
        self.ax.plot_surface(x, y, z, color='blue')
        
        # Set the aspect ratio to be equal so that the sphere looks like a sphere
        self.ax.set_aspect('equal')
        
        # Set the labels for the axes
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
        # Show the plot
        plt.show()

    def set(self, radius, pos):
        self._x, self._y, self._z = pos[0], pos[1], pos[2]  # Coordinates of the center of the sphere
        self.radius = radius  # Radius of the sphere


