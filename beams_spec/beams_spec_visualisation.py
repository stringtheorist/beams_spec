
import numpy as np
import scipy as sp
from time import sleep
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from .beams_spec_core import *





def visually_check_initial_conditions(s, sparams:SolutionParameters):

    params = sparams.basis_params.adim_params
    v1_s0_ = np.vectorize(v1_init)
    e1_s0_ = np.vectorize(e1_init)
    w2_s0_ = np.vectorize(o2_init)
    k2_s0_ = np.vectorize(k2_init)

    v1_s0_d = v1_s0_(s, params)
    e1_s0_d = e1_s0_(s, params)
    o2_s0_d = w2_s0_(s, params)
    k2_s0_d = k2_s0_(s, params)

    v1_st_d = v1(s, 0.0, sparams)
    e1_st_d = e1(s, 0.0, sparams)
    o2_st_d = o2(s, 0.0, sparams)
    k2_st_d = k2(s, 0.0, sparams)

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

def visually_inspect_efuncs(efparams:BasisParameters, norms_ef, s, wc):
    """  """
    wn = efparams.w
    nc = np.asarray(wn > wc).nonzero()[0][0]

    fig, axes = plt.subplots(4, 2)

    for i in range(5):
        axes[0, 0].plot(s, np.imag(v1_mode_i(s, i, efparams, norms_ef)), lw=1)
        axes[1, 0].plot(s, np.imag(e1_mode_i(s, i, efparams, norms_ef)), lw=1)
        axes[2, 0].plot(s, np.imag(o2_mode_i(s, i, efparams, norms_ef)), lw=1)
        axes[3, 0].plot(s, np.imag(k2_mode_i(s, i, efparams, norms_ef)), lw=1)
        
        # axes[0, 0].plot(s, (v1[i, 0](s)), lw=1)
        # axes[1, 0].plot(s, (e1[i, 0](s)), lw=1)
        # axes[2, 0].plot(s, (o2[i, 0](s)), lw=1)
        # axes[3, 0].plot(s, (k2[i, 0](s)), lw=1)

        axes[0, 1].plot(s, v1_mode_i(s, nc + i, efparams, norms_ef), lw=3)
        axes[1, 1].plot(s, e1_mode_i(s, nc + i, efparams, norms_ef), lw=3)
        axes[2, 1].plot(s, o2_mode_i(s, nc + i, efparams, norms_ef), lw=3)
        axes[3, 1].plot(s, k2_mode_i(s, nc + i, efparams, norms_ef), lw=3)

    fig.tight_layout()
    plt.show()


def simulate_beam_exact_theta(phi1, phi3,
                              s_grid, t_grid,
                              sparams:SolutionParameters,
                              save=False, filename='beam_sim_exact_theta.mp4'):
    """animate the beam motion"""
    
    ns = np.shape(phi1)[0]
    nt = np.shape(phi1)[1]
    s_grid = s_grid.reshape((ns,))

    fig, ax = plt.subplots()

    sines = np.zeros((ns, nt))
    cosines = np.zeros((ns, nt))
    phx_s = np.zeros((ns,))
    phy_s = np.zeros((ns,))
    x_lim_fig = s_grid[-1] + 10.0
    y_lim_fig = s_grid[-1] + 10.0

    for ii in range(ns):
        for jj in range(nt):
            sines[ii, jj] = np.sin(theta(s_grid[ii], t_grid[jj], sparams))
            cosines[ii, jj] = np.cos(theta(s_grid[ii], t_grid[jj], sparams))

    def update(it):

        ax.clear()
        text_box_props = dict(boxstyle='round', alpha=0.5)
        textstr = f"t = {t_grid[it]:0.2f}" 
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=text_box_props)
       
        for i in range(ns):
            phx_s[i] = -phi1[i, it]*sines[i, it] + phi3[i, it]*cosines[i, it]
            phy_s[i] = +phi1[i, it]*cosines[i, it] + phi3[i, it]*sines[i, it]

        medial = ax.plot(phx_s, phy_s)

        q1 = ax.quiver(phx_s, phy_s, -sines[:, it], cosines[:, it], headaxislength=0, headlength=0)
        q2 = ax.quiver(phx_s, phy_s, sines[:, it], -cosines[:, it], headaxislength=0, headlength=0)
        ax.set(xlim=[0.0, x_lim_fig], ylim=[-y_lim_fig, +y_lim_fig])
        ax.set_aspect('equal', adjustable='box')
        
        sleep(0.1)
        fig.canvas.draw()
        return medial

    anim = FuncAnimation(fig=fig, func=update, frames=nt)
    if save==True:
        anim.save(filename=filename, writer="ffmpeg")
    else:
        plt.show()
    
      
def simulate_beam_approx_theta(phi1, phi3,
                               cos_theta, sin_theta,
                               s_grid, t_grid,
                               save=False, filename='beam_sim_approx_theta.mp4'):
    """Uses externally computed theta"""
    ns = np.shape(phi1)[0]
    nt = np.shape(phi1)[1]
    s_grid = s_grid.reshape((ns,))

    fig, ax = plt.subplots()

    phx_s = np.zeros((ns,))
    phy_s = np.zeros((ns,))
    x_lim_fig = s_grid[-1] + 10.0
    y_lim_fig = s_grid[-1] + 10.0

   
    def update(it):

        ax.clear()
        text_box_props = dict(boxstyle='round', alpha=0.5)
        textstr = f"t = {t_grid[it]:0.2f}" 
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=text_box_props)

       
        for i in range(ns):
            phx_s[i] = -phi1[i, it]*sin_theta[i, it] + phi3[i, it]*cos_theta[i, it]
            phy_s[i] = +phi1[i, it]*cos_theta[i, it] + phi3[i, it]*sin_theta[i, it]

        medial = ax.plot(phx_s, phy_s)

        q1 = ax.quiver(phx_s, phy_s, -sin_theta[:, it], cos_theta[:, it], headaxislength=0, headlength=0)
        q2 = ax.quiver(phx_s, phy_s, sin_theta[:, it], -cos_theta[:, it], headaxislength=0, headlength=0)
        ax.set(xlim=[0.0, x_lim_fig], ylim=[-y_lim_fig, +y_lim_fig])
        ax.set_aspect('equal', adjustable='box')
        
        sleep(0.1)
        fig.canvas.draw()
        return medial
    
    anim = FuncAnimation(fig=fig, func=update, frames=nt)
    if save == True: 
        anim.save(filename=filename, writer="ffmpeg")
    else:
        plt.show()



  
    
    
    
