import numpy as np

################## 1D models #################################
def basic_sin(x):
    return np.sin(x)

def sym_sin(x):
    # reflection symmetry about y-axis
    return (np.sin(x)**2) * np.cos(x)

################## 2D models #################################

def eggbox_2d(x,y):
    return (2+np.cos(x/2)*np.cos(y/2))**5

def simple_quadratic(x,y):
    return x**2 + y**2

################# Include new toy models below ######################