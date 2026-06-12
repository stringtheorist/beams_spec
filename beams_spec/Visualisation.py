
import numpy as np
import scipy as sp
from time import sleep
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from .InitCondtions import *
from .UtilityFunctions import *
from .BeamProblem import TimoshenkoAdimParams




def visually_check_initial_conditions(s, sparams:Sol_params):

    params = sparams.efparams.Tparams
    v1_s0_ = np.vectorize(v1_s0)
    e1_s0_ = np.vectorize(e1_s0)
    w2_s0_ = np.vectorize(w2_s0)
    k2_s0_ = np.vectorize(k2_s0)

    v1_s0_d = v1_s0_(s, params)
    e1_s0_d = e1_s0_(s, params)
    o2_s0_d = w2_s0_(s, params)
    k2_s0_d = k2_s0_(s, params)

    v1_st_d = v1_st(s, 0.0, sparams)
    e1_st_d = e1_st(s, 0.0, sparams)
    o2_st_d = o2_st(s, 0.0, sparams)
    k2_st_d = k2_st(s, 0.0, sparams)

    fig, axes = plt.subplots(4, 1)

    axes[0].plot(s, np.real(v1_s0_d))
    axes[0].plot(s, np.real(v1_st_d))

    axes[1].plot(s, np.real(e1_s0_d))
    axes[1].plot(s, np.real(e1_st_d))

    axes[2].plot(s, np.real(o2_s0_d))
    axes[2].plot(s, np.real(o2_st_d))

    axes[3].plot(s, np.real(k2_s0_d))
    axes[3].plot(s, np.real(k2_st_d))

    fig.tight_layout()
    plt.show()
    



def visual_check_f_functions(f_test, f_test_matlab, w):
    fig, axs = plt.subplots(2)
    axs[0].plot(w, np.real(f_test), 'x')
    axs[0].plot(w, np.real(f_test_matlab))
    axs[1].plot(w, np.imag(f_test), 'x')
    axs[1].plot(w, np.imag(f_test_matlab))
    fig.tight_layout()
    plt.show()

def visually_inspect_efuncs(efparams:Ef_params, norms_ef, s, wc):
    """  """
    wn = efparams.w
    nc = np.asarray(wn > wc).nonzero()[0][0]

    fig, axes = plt.subplots(4, 2)

    for i in range(5):
        axes[0, 0].plot(s, np.imag(v1_ef(s, i, efparams, norms_ef)), lw=1)
        axes[1, 0].plot(s, np.imag(e1_ef(s, i, efparams, norms_ef)), lw=1)
        axes[2, 0].plot(s, np.imag(o2_ef(s, i, efparams, norms_ef)), lw=1)
        axes[3, 0].plot(s, np.imag(k2_ef(s, i, efparams, norms_ef)), lw=1)
        
        # axes[0, 0].plot(s, (v1[i, 0](s)), lw=1)
        # axes[1, 0].plot(s, (e1[i, 0](s)), lw=1)
        # axes[2, 0].plot(s, (o2[i, 0](s)), lw=1)
        # axes[3, 0].plot(s, (k2[i, 0](s)), lw=1)

        axes[0, 1].plot(s, v1_ef(s, nc + i, efparams, norms_ef), lw=3)
        axes[1, 1].plot(s, e1_ef(s, nc + i, efparams, norms_ef), lw=3)
        axes[2, 1].plot(s, o2_ef(s, nc + i, efparams, norms_ef), lw=3)
        axes[3, 1].plot(s, k2_ef(s, nc + i, efparams, norms_ef), lw=3)

    fig.tight_layout()
    plt.show()


def simulate_beam(phi1, phi3, s, t, sparams:Sol_params):
    """animate the beam motion"""
    
    ns = np.shape(phi1)[0]
    nt = np.shape(phi1)[1]
    s = s.reshape((ns,))

    fig, ax = plt.subplots()

    sines = np.zeros((ns, nt))
    cosines = np.zeros((ns, nt))
    phx_s = np.zeros((ns,))
    phy_s = np.zeros((ns,))

    for ii in range(ns):
        for jj in range(nt):
            sines[ii, jj] = np.sin(theta_st(s[ii], t[jj], sparams))
            cosines[ii, jj] = np.cos(theta_st(s[ii], t[jj], sparams))

    def update(it):

        ax.clear()
        ax.set(xlim=[0.0, 60.0], ylim=[-80.0, +80.0])
        for i in range(ns):
            phx_s[i] = -phi1[i, it]*sines[i, it] + phi3[i, it]*cosines[i, it]
            phy_s[i] = +phi1[i, it]*cosines[i, it] + phi3[i, it]*sines[i, it]

        medial = ax.plot(phx_s, phy_s)

        q1 = ax.quiver(phx_s, phy_s, -sines[:, it], cosines[:, it])
        q2 = ax.quiver(phx_s, phy_s, sines[:, it], -cosines[:, it])
        
        sleep(0.1)
        fig.canvas.draw()
        return medial

    anim = FuncAnimation(fig=fig, func=update, frames=nt)
    plt.show()
    
        
    
    

    
    
    
    
