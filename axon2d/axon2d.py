# imports
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm
from matplotlib import rc
import time

from matplotlib.ticker import AutoMinorLocator  # For minor ticks on axes
from matplotlib.ticker import MaxNLocator  # Force integers on 2axes
from scipy.signal import argrelextrema  # To find maxima/minima
from pathlib import Path
import csv
import sys


#PLOT PARAMETERS
#plt.style.use('~/.config/matplotlib/paper')
#['~/.config/matplotlib/paper'
#rc('text', usetex=True)
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['axes.titlesize'] = 18
matplotlib.rcParams['axes.labelsize'] = 24
matplotlib.rcParams['ytick.labelsize'] = 16
matplotlib.rcParams['xtick.labelsize'] = 16
matplotlib.rcParams['legend.fontsize'] = 12
#matplotlib.rcParams['errorbar.capsize'] = 3
matplotlib.rcParams['figure.figsize'] = (9,6)
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times']
matplotlib.rcParams['lines.linewidth'] = 2
# matplotlib.rcParams['figure.dpi'] = 600
matplotlib.rcParams['text.usetex']=False # needs tex

###########################
#       2D plotting       #
###########################


def plot_circle(ax, pos, rads, box,filler=False, op_alpha=.8, clr="g",
                *args, **kwargs):
    """ Function that plots circles for the 2D growth model. Circles are green
    and centers are black by default. Adds the circle elements to current axis
    element.

    Parameters
    ----------
    ax : axes object
        Axes object to add plot too.
    pos : ndarray
        Array containing the x, y and z position of each neuron.
    box : array-like, xy size
        Size of box to plot in.
    sph_rs : ndarray
        Array containing the radius for each neuron.
    op_alpha : float
        Opacity of spheres (default value is 0.6).
    """
    xs, ys = pos[:, 0], pos[:, 1]

    ax.scatter(xs, ys, c='k')

    for i in range(len(xs)):
        c = plt.Circle((xs[i], ys[i]), rads[i], clip_on=False,
                       fill=filler, alpha=op_alpha, color=clr,
                       *args, **kwargs)
        ax.add_artist(c)

    ax.set_ylabel('y position [mm]')
    ax.set_xlabel('x position [mm]')
    ax.set_xlim(-0.2, box[0] + 0.2)
    ax.set_ylim(-0.2, box[1] + 0.2)
    ax.set_aspect('equal')
    ax.margins(0)

###########################
#       axon placem       #
###########################


def new_pos(pos1, pos2, rmin, box, overlaps):
    """Function to get new position if there is overlap in all positions. If not
    all are overlapping, we assume they are in a different position.

    Parameters
    ----------
    pos1 : float
        Position of neuron 1.
    pos2 : float
        Position of neuron 2.
    rmin : float
        Smallest radius allowed for neurons.
    box : array-like, nxm size
        Size of box to place neurons in.
          
    Returns:
    --------

    """

    if (abs(pos1 - pos2) < rmin).all() == True:
        overlaps = True
        return np.random.uniform(0.0, 1.0, len(box)), overlaps
    else:
        return pos2, overlaps

###########################
#       axon growth       #
###########################

def firing_rate(x, theta=0.5, alpha=0.12):
    """ Sigmoidal firing rate function
    
    Parameters
    ----------
    x : float
        Mean membrane potential.
    theta : float
        Inflection point (mean firing activity) of sigmoidal curve (default 
        value 0.12)
    alpha : float
        Steepness of sigmoidal curve (default value 0.12)

    Returns
    -------
    f : float
        Firing rate of x.

    """

    expo = np.exp((theta - x) / alpha)
    f = 1 / (1 + expo)

    return f

def radius_growth(radius,
                  x,
                  dt,
                  rho=4e-3,
                  epsil=0.6,
                  beta=0.1,
                  ):
    """ Calculate the change in radius as a function of growth multiplied by
    the time step resolution.
    Parameters
    ----------
    radius : ndarray
        radius of neuron(i), array type
    x : ndarray
        Activity levels for each neuron.
    dt : float
        Time step resolution.
    rho : float
    epsil : 
    beta : 

    Returns
    -------
    new_radius : float
        Radius after growth.
    """

    expo = np.exp((epsil - firing_rate(x)) / beta)
    rad_chng = dt * (1 - 2 / (1 + expo)) * rho
    new_radius = radius + rad_chng

    return new_radius


def cell_overlap(pos, rmin, box, trs=1000):
    """ Check for overlapping cell bodies in initial placement, and move any
    overlapping cells to new position.

    Parameters
    ----------

    pos : float or array-like, shape (n,m)
        Position of neuron(s)
    rmin : float or array-like, shape (n,)
        Minimum radius for cell body of neuron.
    trs : int
        Number of iterations for while loop to check overlap (default value 
        1000).

    Returns
    -------
    pos : float or array-like, shape (n,m)
        Updated position with no overlapping cell bodies.
    """
    overlaps = True
    orgn = np.zeros((1, pos.shape[1]))
    tcount = 0
    npos = pos.copy()
    while overlaps == True and tcount <= trs:
        overlaps = False
        d = sp.spatial.distance.cdist(npos, orgn)
        srt = np.argsort(d, axis=0)
        tcount += 1
        for i in range(len(srt) - 1):
            if abs(d[srt[i]] - d[srt[i + 1]]) <= rmin:
                npos[srt[i + 1]], overlaps = new_pos(npos[srt[i]], npos[srt[i + 1]], rmin,
                                           box, overlaps)
        if tcount > trs:
            sys.exit("Too many tries to avoid overlap.")

    return npos


def disk_area(r):
    """Calculate area covered by axon outgrowth from neuron i with radius r.

    Parameters
    ----------
    r : float
        The radius of axon growth disc.
    
    Returns
    -------
    area : float
        Area covered by axon growth.
    """
    area = np.pi * r**2
    return area


def disk_overlap(d, r1, r2):
    """Calculate the overlapping area of two spheres of radius r1, r2, at 
    distance d.

    Parameters
    ----------
    d : float
        Distance between two neurons.
    r1 : float
        Radius of neuron 1.
    r2 : float
        Radius of neuron 2.

    Returns
    -------
    area : float
        Area overlapping for two spheres if they overlap. 
    """
    d1 = (r1**2 - r2**2 + d**2) / (2 * d)
    d2 = (r2**2 - r1**2 + d**2) / (2 * d)

    if d >= (r1 + r2):
        overlap = 0.0
    elif r1 > (d + r2):
        overlap = np.pi * r2**2
    elif r2 > (d + r1):  # 1 is entirely contained in one
        overlap = np.pi * r1**2
    else:
        overlap = sub_area(r1, d1) + sub_area(r2, d2)

    return overlap


def overlap2D(pos, rad):
    """Checks overlap of two neurons with a given position and radius, assuming
    that we have a overlap in two dimensions (overlapping disks). Uses euclidian 
    distance as default for the distance between neuron pairs. 
    
    Parameters
    ----------
    pos : ndarray
        The array containing center position of neurons
    rad : ndarray
        Array of radius for each 

    Returns
    -------
    overlap : ndarray
        Array containing amount of overlap for each neuron pair.

    """
    neurons = pos.shape[0]
    r = np.zeros((neurons, neurons, 2))
    overlap = np.zeros((neurons, neurons))
    distance = sp.spatial.distance.cdist(pos, pos)

    for i in range(neurons):
        for j in range(neurons):
            r[i, j, :] = [rad[i], rad[j]]  #????
            if i != j:
                overlap[i, j] = disk_overlap(distance[i, j], rad[i], rad[j])

    return overlap


def sub_area(r, d):
    """Calculates half of overlapping disk area (half-lens) for overlapping 
    area.
    
    Parameters
    ----------
    r : float
        Radius of the half lens.
    d : float
        Diameter of offset for lens.
    
    Returns
    -------
    area : float
        Half of overlapping area for disk (half lens)
    """
    area = r**2 * np.arccos(d / r) - d * np.sqrt(r**2 - d**2)
    return area
    

#########################
#      RUN GROWTH       #
#########################


def grow_network(n_pos=None, 
                 neurons=10, 
                 x_dim=1.0,
                 y_dim=1.0,
                 res_steps=24*60,
                 days=50,
                 min_rad=12e-3,
                 s=0.1,
                 u_min=0.0,
                 u_max=1.0,
                 *args,
                 **kwargs
                 ):

    """Run axon growth simulation in two dimentions.
    
    Parameters
    ----------
    n_pos : ndarray 
        Position of neurons (default value is None). Should be mm,if not scale
        using x_dims and y_dims
    x_dim : float
        Scaling factor for magnification or dimension of box.
    y_dim : float
        Scaling factor for magnification or dimension of box.
    
    Returns
    -------
    w : ndarray
        Weights of netwokr after growth, delimiter is ','.
    """

    tic = time.time()           # Simple timer

    start_time = time.localtime()
    current_time = time.strftime("%H:%M:%S", start_time)
    print(current_time)

    # Place neurons
    if isinstance(n_pos, np.ndarray):
        n_pos[:, 0] = n_pos[:, 0]/x_dim
        n_pos[:, 1] = n_pos[:, 1]/y_dim
        neurons = n_pos.shape[0]
        x_max, y_max = n_pos.max(axis=0)
        box_dims = np.array([x_max, y_max])
        print('Neuron placement based on input!')
    else:
        box_dims = np.array([x_dim, y_dim])
        n_pos = np.random.uniform(0.0, 1.0, (neurons, 2))
        n_pos = cell_overlap(n_pos, min_rad, box_dims, trs = 10e100)
        print('Neurons placed!')

    steps = days*res_steps
    dt = 5.0/res_steps
    u_n = np.zeros(neurons)

    print(kwargs)

    if 'r_path' not in kwargs:
        r_path = 'results_' + str(neurons)
    else:
        r_path = kwargs['r_path'] 

    savepath  = Path.cwd() / (r_path)
    if not savepath.exists():
        savepath.mkdir()        # same effect as exist_ok = True

    # initiate random neuron growth size
    n_rad = np.ones(neurons) * min_rad  #np.random.rand(neurons)*max_init_rad
    # n_rad[n_rad < min_rad] = min_rad

    fig_init = plt.figure(figsize=(9, 8))
    ax_init = fig_init.add_subplot(111)

    ax_init.set_ylabel('y')
    ax_init.set_xlabel('x')

    save_fig_init = savepath / 'fig_init.png'
    plot_circle(ax_init, n_pos, n_rad, box_dims)
    fig_init.savefig(save_fig_init)  #, dpi =600)

    # initiate neurons membrane potential

    ov_ar = overlap2D(n_pos, n_rad)
    w = s * ov_ar

    area = np.zeros(steps)
    k = np.zeros(steps)
    r = np.zeros(steps)
    u_av = np.zeros(steps)

    for i in range(steps):
        if (i+1)%res_steps == 0:
            print('Arrived at day:', (i+1)/res_steps)

        # grow neurons to stable size
        ov_ar = overlap2D(n_pos, n_rad)
        w = s * ov_ar

        activity = w * firing_rate(u_n)

        du_n = (- u_n/np.exp(1) + (1- u_n) * activity.sum(axis=1))
        # if activity.any() > 0.6 * s:
        #     print(du_n)

        u_n = u_n + du_n

        n_rad  = radius_growth(n_rad, u_n, dt)
        n_rad[n_rad < min_rad] = min_rad
        u_n[u_n < u_min] = u_min
        u_n[u_n > u_max] = u_max

        area[i] = np.mean(disk_area(n_rad))

        k[i] = np.count_nonzero(w) / neurons #w.size
        r[i] = np.mean(n_rad)
        u_av[i] = np.mean(u_n)
    #     #spiking


    #########################
    #       PLOTTING        #
    #########################

    fig_1 = plt.figure(figsize=(9, 8))
    ax_1 = fig_1.add_subplot(111)

    ax_1.set_ylabel('y')
    ax_1.set_xlabel('x')

    save_fig_1 = savepath / 'fig_1.png'
    plot_circle(ax_1, n_pos, n_rad, box_dims)
    fig_1.savefig(save_fig_1)  #, dpi =600)

    # plt.show()

    x_val = np.linspace(1, steps, steps)

    fig_area = plt.figure(figsize=(10, 8))
    ax_area = fig_area.add_subplot(111)

    ax_area.plot(x_val, area)

    ax_area.set_ylabel('Average area')
    ax_area.set_xlabel('Step')

    save_fig_area = savepath / 'fig_area.png'
    fig_area.savefig(save_fig_area)

    fig_k = plt.figure(figsize=(10, 8))
    ax_k = fig_k.add_subplot(111)

    ax_k.plot(x_val, k)

    ax_k.set_ylabel('Average number of connections per neuron')
    ax_k.set_xlabel('Step')

    save_fig_k = savepath / 'fig_k.png'
    fig_k.savefig(save_fig_k)

    fig_r = plt.figure(figsize=(10, 8))
    ax_r = fig_r.add_subplot(111)

    ax_r.plot(x_val, r)

    ax_r.set_ylabel('Average radius')
    ax_r.set_xlabel('Step')

    save_fig_r = savepath / 'fig_r.png'
    fig_r.savefig(save_fig_r)

    fig_saturation = plt.figure(figsize=(10, 8))
    ax_saturation = fig_saturation.add_subplot(111)

    ax_saturation.plot(x_val, u_av)

    ax_saturation.set_ylabel('Average saturation')
    ax_saturation.set_xlabel('Step')

    save_fig_saturation = savepath / 'fig_saturation.png'
    fig_saturation.savefig(save_fig_saturation)

    save_var = savepath / 'weight.txt'
    #outdated. Use numpy write txt
    np.savetxt(save_var, w, delimiter=',')

    end_time = time.localtime()
    current_time = time.strftime("%H:%M:%S", end_time)
    print(current_time)
    toc = time.time() - tic
    print("Elapsed time is: ", toc)
    print("Elapsed time is ", np.floor(toc/60), ' minutes and ', toc % 60, 'seconds.')

    return w
