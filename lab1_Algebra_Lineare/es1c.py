import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from ase.io import write, read
import math

DataSet = read("dataset.xyz", index = ':')
masses = [molecule.get_masses() for molecule in DataSet]
positions = [molecule.get_positions() for molecule in DataSet]

# computing the center of mass of every molecule -------------------
center_of_mass = []
for index_mol in range(len(positions)):
    den = 0
    num = 0
    for index_atom in range(len(positions[index_mol])):
        den += masses[index_mol][index_atom]
        num += masses[index_mol][index_atom] * positions[index_mol][index_atom]
    center_of_mass_mol = num/den
    center_of_mass.append(center_of_mass_mol)

# computing I for every molecule -----------------------------------
    sf_count = ob_count = pr_count = as_count = 0
    relative_tolerance = 0.01
    absolute_tolerance = 1
for index_mol in range(len(positions)):
    Ixx = Iyy = Izz = Ixy = Iyz = Ixz = 0
    for index_atom in range(len(positions[index_mol])):
        pos_relative_to_cm = positions[index_mol][index_atom] - center_of_mass[index_mol]
        x_i = pos_relative_to_cm[0]
        y_i = pos_relative_to_cm[1]
        z_i = pos_relative_to_cm[2]

        Ixx += masses[index_mol][index_atom] * (y_i **2 + z_i **2)
        Iyy += masses[index_mol][index_atom] * (x_i **2 + z_i **2)
        Izz += masses[index_mol][index_atom] * (x_i **2 + y_i **2)
        Ixy += - masses[index_mol][index_atom] * x_i * y_i
        Iyz += - masses[index_mol][index_atom] * y_i * z_i
        Ixz += - masses[index_mol][index_atom] * x_i * z_i
    
    I = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])
    eigenvals, eigenvects = np.linalg.eigh(I)

    # classifying molecules based on Ia Ib, Ic ---------------------


    # CLASSIFICATION WITH RELATIVE TOLERANCE -----------------------
    if (math.isclose(eigenvals[0], eigenvals[1], rel_tol=relative_tolerance) and math.isclose(eigenvals[1], eigenvals[2], rel_tol=relative_tolerance)):
        sf_count += 1
        print(index_mol, ': sferica')
    elif (math.isclose(eigenvals[0], eigenvals[1], rel_tol=relative_tolerance) and eigenvals[1] < eigenvals[2]):
        ob_count += 1
        print(index_mol, ': oblata')
    elif (eigenvals[0] < eigenvals[1] and math.isclose(eigenvals[1], eigenvals[2], rel_tol=relative_tolerance)):
        pr_count += 1
        print(index_mol, ': prolata')
    elif (eigenvals[0] < eigenvals[1] < eigenvals[2]):
        as_count += 1
        print(index_mol, ': asimmetrica')

    # CLASSIFICATION WITH ABSOLUTE TOLERANCE
    # if (math.isclose(eigenvals[0], eigenvals[1], abs_tol=absolute_tolerance) and math.isclose(eigenvals[1], eigenvals[2], abs_tol=absolute_tolerance)):
    #     sf_count += 1
    # elif (math.isclose(eigenvals[0], eigenvals[1], abs_tol=absolute_tolerance) and eigenvals[1] < eigenvals[2]):
    #     ob_count += 1
    # elif (eigenvals[0] < eigenvals[1] and math.isclose(eigenvals[1], eigenvals[2], abs_tol=absolute_tolerance)):
    #     pr_count += 1
    # elif (eigenvals[0] < eigenvals[1] < eigenvals[2]):
    #     as_count += 1

print('Molecole sferiche: ' + str(sf_count) + '\n' + 'Molecole oblate: ' + str(ob_count) + '\n' + 'Molecole prolate: ' + str(pr_count) + '\n' + 'Molecole asimmetriche: ' + str(as_count))