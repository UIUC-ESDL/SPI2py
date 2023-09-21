import numpy as np
import torch


a = np.zeros((3, 3))
b = np.arange(9).reshape((3, 3))

fixed_dof = []
free_dof = []

fixed_values = []
free_values = []

design_vector = np.array([1.5, 2.5, 3.5])
