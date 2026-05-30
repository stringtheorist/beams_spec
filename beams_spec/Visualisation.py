
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from .InitCondtions import *

def visual_check_init_conditions(v1_st, e1_st, o2_st, k2_st, s, params):
    """Plot initial conditions coming from the solution vs their
    actual values. Note that this function will block until
    the plot window is closed."""

    v1_s0_ = np.vectorize(v1_s0)
    e1_s0_ = np.vectorize(e1_s0)
    w2_s0_ = np.vectorize(w2_s0)
    k2_s0_ = np.vectorize(k2_s0)

    v1_s0_d = v1_s0_(s, params)
    e1_s0_d = e1_s0_(s, params)
    w2_s0_d = w2_s0_(s, params)
    k2_s0_d = k2_s0_(s, params)

    v1_st_d = v1_st(s, 0)
    e1_st_d = e1_st(s, 0)
    w2_st_d = o2_st(s, 0)
    k2_st_d = k2_st(s, 0)

    #check that the sizes match.
    assert np.shape(v1_s0_d) == np.shape(v1_st_d)
    assert np.shape(e1_s0_d) == np.shape(e1_st_d)
    assert np.shape(w2_s0_d) == np.shape(w2_st_d)
    assert np.shape(k2_s0_d) == np.shape(k2_st_d)

    fig, axes = plt.subplots(4, 1)
    #fig.clf()

    ax1 = axes[0]
    ax1.plot(s, np.real(v1_s0_d))
    ax1.plot(s, np.real(v1_st_d), 'x-')

    ax2 = axes[1]
    ax2.plot(s, np.real(e1_s0_d))
    ax2.plot(s, np.real(e1_st_d), 'x-')

    ax3 = axes[2]
    ax3.plot(s, np.real(w2_s0_d))
    ax3.plot(s, np.real(w2_st_d), 'x-')

    ax4 = axes[3]
    ax4.plot(s, np.real(k2_s0_d))
    ax4.plot(s, np.real(k2_st_d), 'x-')

    # ax5 = axes[4]
    # ax5.plot(s, ph_s0(s))
    # ax5.plot(s, ph_st(s, 0), 'x-')
    
    # ax6 = axes[5]
    # ax6.plot(s, th_s0(s))
    # ax6.plot(s, th_st(s, 0), 'x-')

    fig.tight_layout()
    plt.show()

    
def visual_check_eigenfunctions(v1, e1, o2, k2, wn, wc, nmax, s):
    """Plot the first nmax eigenfunctions to check if they look okay"""
    #DEBUG NOTE: Date 27/05/2026 there is something wrong with the
    #lf eigenfunctions probably due to the fact that your dot product
    #removes imaginary stuff unlike the one from the matlab implementation
    #which is based on the integrate function which takes complex arguments
    #
    nc = np.asarray(wn > wc).nonzero()[0][0]

    fig, axes = plt.subplots(4, 2)
    for i in range(nmax):  
        axes[0, 0].plot(s, np.imag(v1[i, 0](s)), lw=1)
        axes[1, 0].plot(s, np.imag(e1[i, 0](s)), lw=1)
        axes[2, 0].plot(s, np.imag(o2[i, 0](s)), lw=1)
        axes[3, 0].plot(s, np.imag(k2[i, 0](s)), lw=1)

        axes[0, 1].plot(s, v1[nc + i, 0](s), lw=3)
        axes[1, 1].plot(s, e1[nc + i, 0](s), lw=3)
        axes[2, 1].plot(s, o2[nc + i, 0](s), lw=3)
        axes[3, 1].plot(s, k2[nc + i, 0](s), lw=3)

    fig.tight_layout()
    plt.show()
    
