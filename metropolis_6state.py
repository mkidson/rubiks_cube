import math
import random
import numpy as np
import matplotlib.pyplot as pls
from matplotlib.animation import FuncAnimation


def hot_start():
    lattice = np.random.random_integers(0,5,(ns,ns))   
    return lattice


def bc(i):              # Periodic Boundary Conditions
    if i > ns-1:
        return 0
    if i < 0:
        return ns-1
    else:
        return i
    
def energy_of_config(centre,left,right,bottom,top):     # Energy contribution of site [j,k]
    energy = 0.
    for neighbour in [left,right,bottom,top]:
        if centre == neighbour:
            energy -=1 
        else:
            energy +=1
    return energy

def sweep(lattice, beta):   # Sweeps through the whole lattice, testing possible 'spin flips' at each site
    for j in range(0,ns):
        for k in range(0,ns):
            centre = lattice[j,k]
            left = lattice[bc(j-1), k]
            right = lattice[bc(j+1), k]
            bottom = lattice[j, bc(k-1)]
            top = lattice[j,bc(k+1)]

            # Energy of configuration before 'spin flip'
            old_energy = energy_of_config(centre,left,right,bottom,top)
            
            # Energy of the configuration after the 'spin' at [j,k] is flipped to a random state (new_spin)
            new_spin = np.random.random_integers(0,5)
            new_energy = energy_of_config(new_spin,left,right,bottom,top)

            # If the change in energy is negative, accept the spin flip.
            # If the change is positive, accept only if it satisfies the 'temperature requirement'
            # The probablity of transition is given by the ratio of the boltzmann weightings of the old and new state : exp(-dE/T).
            dE =  new_energy - old_energy
            if dE <= 0.:
                lattice[j,k] = new_spin
            elif np.exp(-beta*dE) > np.random.rand():
                lattice[j,k] = new_spin
    return lattice



def animate(lattices):
    matrices = lattices

    # Create a figure and axis for the animation
    fig, ax = pls.subplots()

    # Function to update the plot for each frame
    def update(frame):
        ax.clear()
        ax.matshow(matrices[frame], cmap='viridis')
        ax.set_title(f'Frame {frame + 1}/{num_frames}')

    # Create the animation
    animation = FuncAnimation(fig, update, frames=num_frames, repeat=False, interval=animation_speed)
    pls.show()
    


#=============================== MAIN ========================================


ns=20                       # Width of square lattice
beta =0.9                   # Inverse Temperature
updates = 150               # Number of sweeps to do
num_frames = updates        # Number of animation frames
animation_speed=90          # Time between frames in milliseconds

# Initial configuration
lattice= hot_start()


lattices=np.zeros([updates,ns,ns])
for n in range(updates):
    latt_update=sweep(lattice,beta)
    lattices[n,:,:]=latt_update

animate(lattices)

