import matplotlib.pyplot as plt
import numpy as np
import itertools

import cellconstructor as CC
import sscha.Ensemble

import sscha.Classify as Classify

from scipy.special import eval_hermite
from scipy.special import factorial
from scipy.linalg import eigh

hbar = 1 # Atomic units
kb = 3.166811e-6/2 # Ry/K

angstroms2bohr = 1.8897259886
Ry2eV = 13.6056980659
eV2Ry = 0.0734985857
eV2cm = 8065.544
Ry2cm = Ry2eV*eV2cm
cm2Thz = 0.0299792458

class model():

    def __init__(self, a, b, c, d, a0):
        self.Natoms = 2
        self.Nspecies = 1

        self.atom_species = np.array([
            ["C", "1"],
            ["C", "1"]
        ])

        # I used the mass of Carbon
        self.masses = np.array([10947.15,10947.15])
        
        self.unit_cell = np.array([
            [5,0,0],
            [0,5,0],
            [0,0,4]
        ])

        self.atom_coords_frac = np.array([
            [0,0,0],
            [0.5,0.5,0.5]
        ])

        self.atom_coords = CC.Methods.cryst_to_cart(self.unit_cell, self.atom_coords_frac)

        self.pols = 1.0 / np.sqrt(2.0) * np.array([
            [1,0,0,1,0,0],
            [0,1,0,0,1,0],
            [0,0,1,0,0,1],
            [1,0,0,-1,0,0],
            [0,1,0,0,-1,0],
            [0,0,1,0,0,-1], # this is the unstable mode
    ])
        
        self.a = a
        self.b = b
        self.c = c
        self.d = d

        self.a0 = a0

        self.frequencies = [0,0,0,np.sqrt(octic_second_deriv(0,self.a0,0,0,0)+0j), np.sqrt(octic_second_deriv(0,self.a0,0,0,0)+0j), np.sqrt(octic_second_deriv(0,self.a,self.b,self.c,self.d)+0j)]
        self.eigenvalues = [0,0,0,octic_second_deriv(0,self.a0,0,0,0),octic_second_deriv(0,self.a0,0,0,0),octic_second_deriv(0,self.a,self.b,self.c,self.d)]

        # # Test units with full harmonic.
        # self.frequencies = [0,0,0,self.omega,self.omega,self.omega]
        # self.eigenvalues = [0,0,0,self.omega**2,self.omega**2,self.omega**2]

        self.dyns =  self.pols @ np.diag(self.eigenvalues) @ self.pols.T

        self.phis = np.empty(self.dyns.shape)
        for i in range(self.Natoms):
            for j in range(self.Natoms):
                self.phis[i*3:(i+1)*3,j*3:(j+1)*3] = self.dyns[i*3:(i+1)*3,j*3:(j+1)*3] * np.sqrt(self.masses[i] * self.masses[j])

    def get_energies(self, disps):
        Qns = disps @ self.pols.T * np.sqrt(self.masses[0])
        energies = octic_potential(Qns[:,5],self.a,self.b,self.c,self.d) + octic_potential(Qns[:,3],self.a0,0,0,0) + octic_potential(Qns[:,4],self.a0,0,0,0)
        # energies = harmonic_potential(Qns[:,5],self.omega) + harmonic_potential(Qns[:,3],self.omega) + harmonic_potential(Qns[:,4],self.omega)
        return energies
    
    def get_forces(self, disps):
        Qns = disps @ self.pols.T * np.sqrt(self.masses[0])
        forces = (-1)*(np.outer(octic_first_deriv(Qns[:,5],self.a,self.b,self.c,self.d),self.pols[:,5]) + np.outer(octic_first_deriv(Qns[:,3],self.a0,0,0,0),self.pols[3]) + np.outer(octic_first_deriv(Qns[:,4],self.a0,0,0,0),self.pols[4]))
        # forces = (-1)*(np.outer(harmonic_first_deriv(Qns[:,5],self.omega),self.pols[5]) + np.outer(harmonic_first_deriv(Qns[:,3],self.omega),self.pols[3]) + np.outer(harmonic_first_deriv(Qns[:,4],self.omega),self.pols[4]))
        return forces*np.sqrt(self.masses[0]) # Because we are working with the normal mode basis!!
    
    def write(self, file : str) -> None:
        """
        This function writes the DynQ object in QuantumESPRESSO format.
        
        Parameters
        ----------
            file: str.
                Name of the file to be written.
            alat: bool.
                If we want to renormalize the dynamical matrix in alat units as ESPRESSO does. Usefull
                is the dyn matrix will be used in a QunatumESPRESSO calcuation in the later.
        """
        file = open(file, "w")
        file.write(f"Dynamical matrix file\n\n")
        fmt_general_info = "{0:>3d}{1:>5d}{2:>4d}{3:14.7f}{4:14.7f}{5:14.7f}{6:14.7f}{7:14.7f}{8:14.7f}\n"
        fmt_cell = "  {0:15.9f}{1:15.9f}{2:15.9f}\n"
        fmt_masses = " {0:>12d}  '{1:<4}'{2:20.9f}\n"
        fmt_atomic_position = "{0:>5d}{1:>5d} {2:16.10f} {3:16.10f} {4:16.10f}\n"
        fmt_dyn_matrix="{0:12.8f} {1:12.8f}   {2:12.8f} {3:12.8f}   {4:12.8f} {5:12.8f}\n"
        
        norm = np.linalg.norm(self.unit_cell[0,:])
        file.write(fmt_general_info.format(
            self.Nspecies,
            self.Natoms, 
            int(0),
            norm*angstroms2bohr,
            0,0,0,0,0)
            )
        file.write("Basis vectors\n")
        for i in range(3):
            file.write(fmt_cell.format(
                self.unit_cell[i, 0]/norm,
                self.unit_cell[i, 1]/norm,
                self.unit_cell[i, 2]/norm,
            ))
        name_list = []
        mass_list = []
        for atom in range(self.Natoms):
            if self.atom_species[atom, 0] not in name_list:
                name_list.append(self.atom_species[atom,0])
                mass_list.append(self.masses[atom])
        for s in range(self.Nspecies):
            file.write(fmt_masses.format(
                int(s+1),
                name_list[s],
                mass_list[s]
            ))
        for atom in range(self.Natoms):
            file.write(fmt_atomic_position.format(
                int(atom+1),
                int(self.atom_species[atom, 1]),
                self.atom_coords[atom, 0]/norm,
                self.atom_coords[atom, 1]/norm,
                self.atom_coords[atom, 2]/norm
            ))
        file.write("\n     Dynamical  Matrix in cartesian axes\n\n")
        file.write("     q = ( {0:14.9f}{1:14.9f}{2:14.9f} )\n\n".format(
            0,0,0
        ))
        for i in range(self.Natoms):
            for j in range(self.Natoms):
                file.write("{0:>5d}{1:>5d}\n".format(i+1, j+1))
                for k in range(3):
                    file.write(fmt_dyn_matrix.format(
                        self.phis[3*i+k,3*j],
                        0,
                        self.phis[3*i+k,3*j+1],
                        0,
                        self.phis[3*i+k,3*j+2],
                        0,

                    ))
        file.write("\n     Diagonalizing the dynamical matrix\n\n")
        file.write("     q = ( {0:14.9f}{1:14.9f}{2:14.9f} )\n\n".format(
                0,0,0
            ))
        file.write(" **************************************************************************\n")
        for mode in range(self.Natoms*3):
            if np.square(self.frequencies[mode]) < 0: ## Maybe this is the problem.
                file.write("     freq ({0:>4d}) = {1:14.6f} [THz] = {2:14.6f} [cm-1]\n".format(
                int(mode+1),
                np.abs(self.frequencies[mode].imag) * Ry2cm * cm2Thz*(-1),
                np.abs(self.frequencies[mode].imag)* Ry2cm * (-1)
            ))
            else:
                file.write("     freq ({0:>4d}) = {1:14.6f} [THz] = {2:14.6f} [cm-1]\n".format(
                    int(mode+1),
                    self.frequencies[mode].real * Ry2cm * cm2Thz,
                    self.frequencies[mode].real * Ry2cm
                ))
            
            disp = self.pols[mode, :]/np.linalg.norm(self.pols[mode, :]/np.sqrt(self.masses[0]))
            for atom in range(self.Natoms):
                file.write("({0:10.6f}{1:10.6f}{2:10.6f}{3:10.6f}{4:10.6f}{5:10.6f} )\n".format(
                    disp[atom*3].real,
                    0,
                    disp[atom*3+1].real,
                    0,
                    disp[atom*3+2].real,
                    0,
                ))
        file.write(" **************************************************************************\n")

    def plot_quartic_phi(self, eigenvalues, eigenvectors, basis, omegas, pols, weighted_samples, weights, T):
        disps3 = np.outer(np.linspace(-5,5,400),pols[:,3])
        disps4 = np.outer(np.linspace(-5,5,400),pols[:,4])
        disps5 = np.outer(np.linspace(-5,5,400),pols[:,5])

        Q3 = disps3 @ pols * np.sqrt(self.masses[0])
        Q4 = disps4 @ pols * np.sqrt(self.masses[0])
        Q5 = disps5 @ pols * np.sqrt(self.masses[0])

        rho3 = get_rho(basis,Q3[:,3:],omegas[3:],eigenvalues,eigenvectors,T=T)
        rho4 = get_rho(basis,Q4[:,3:],omegas[3:],eigenvalues,eigenvectors,T=T)
        rho5 = get_rho(basis,Q5[:,3:],omegas[3:],eigenvalues,eigenvectors,T=T)

        q0, rho3_0, E0 = get_reference_rho(self.a,self.b,self.c,self.d,T,x_max=500,N=1000)
        q0, rho4_0, E0 = get_reference_rho(self.a0,0,0,0,T,x_max=500,N=1000)
        q0, rho5_0, E0 = get_reference_rho(self.a0,0,0,0,T,x_max=500,N=1000)

        print(E0, eigenvalues[0])
        f = plt.figure()
        f1 = f.add_subplot(2,3,1)
        f3 = f.add_subplot(2,3,2)
        f5 = f.add_subplot(2,3,3)
        f2 = f1.twinx()
        f4 = f3.twinx()
        f6 = f5.twinx()

        f1b = f.add_subplot(2,3,4)
        f3b = f.add_subplot(2,3,5)
        f5b = f.add_subplot(2,3,6)
        f2b = f1b.twinx()
        f4b = f3b.twinx()
        f6b = f5b.twinx()

        
        energies3 = self.get_energies(disps3)
        energies4 = self.get_energies(disps4)
        energies5 = self.get_energies(disps5)
        
        f1.plot(Q3[:,3], energies3)
        f3.plot(Q4[:,4], energies4)
        f5.plot(Q5[:,5], energies5)

        f1b.plot(Q3[:,3], energies3)
        f3b.plot(Q4[:,4], energies4)
        f5b.plot(Q5[:,5], energies5)

        f2.fill_between(Q3[:,3], 0, rho3, color="red", alpha=0.5)
        f4.fill_between(Q4[:,4], 0, rho4, color="red", alpha=0.5)
        f6.fill_between(Q5[:,5], 0, rho5, color="red", alpha=0.5)
        f2.plot(q0, rho3_0*(rho3.max()/rho3_0.max()), color="black", linestyle="dashed")
        f4.plot(q0, rho4_0*(rho4.max()/rho4_0.max()), color="black", linestyle="dashed")
        f6.plot(q0, rho5_0*(rho5.max()/rho5_0.max()), color="black", linestyle="dashed")

        f2b.hist(weighted_samples[:,0],50,weights=weights)
        #f4b.hist(weighted_samples[:,1],50,weights=weights)
        #f6b.hist(weighted_samples[:,0],50,weights=weights)

        #f2.set_ylim(0,(phi0**2).max()*1.2)
        #f1.set_ylim(-10,20)
        f1.set_ylabel("Energy (Ry)", fontsize=12)
        f1.set_xlabel(r"Q (Bohr$\cdot\sqrt{m_e}$)", fontsize=12)
        f3.set_ylabel("Energy (Ry)", fontsize=12)
        f3.set_xlabel(r"Q (Bohr$\cdot\sqrt{m_e}$)", fontsize=12)
        f5.set_ylabel("Energy (Ry)", fontsize=12)
        f5.set_xlabel(r"Q (Bohr$\cdot\sqrt{m_e}$)", fontsize=12)
        plt.show()
    
    def plot_harmonic_phi0(self):
        disps = np.outer(np.linspace(-0.5,0.5,200),self.pols[3,:])
        Q = disps @ self.pols.T * np.sqrt(self.masses[0])
        f = plt.figure()
        f1 = f.add_subplot()
        f2 = f1.twinx()
        
        tmp_freqs = np.zeros(len(self.frequencies))
        for mode in range(len(self.frequencies)):
            tmp_freqs[mode] = np.abs(self.frequencies[mode])
        
        phi0 = get_basis_function([0,0,0],Q[:,3:],tmp_freqs[3:])
        energies = self.get_energies(disps)
        f1.plot(Q[:,3], energies)
        f2.plot(Q[:,3], phi0**2)
        f2.set_ylim(0,(phi0**2).max()*1.2)
        plt.show()


def independent_harm_osc(n, q, omega):
    qi = np.sqrt(omega/hbar)*q
    Hn = np.polynomial.hermite.hermval(qi, [0]*n + [1])
    prefac = (omega / (np.pi * hbar))**(1/4) / np.sqrt(2.0**n * factorial(n))
    return prefac * Hn * np.exp(-0.5 * qi**2)

def get_basis_function(ns,qs,omegas):

    phi = 1
    for ni, n in enumerate(ns):
        phi *= independent_harm_osc(n, qs[:,ni], omegas[ni])
    return phi

def get_rho(basis, qs, omegas, eigenvalues, eigenvectors, T):
    """
    Computes the exact finite-temperature spatial probability density distribution rho(q, T)
    for a multi-mode coupled harmonic system.
    
    Parameters:
    -----------
    basis : np.ndarray, shape (n_basis, n_modes)
        Uncoupled harmonic basis quantum numbers (e.g., [[0,0,0], [1,0,0], ...]).
    qs : np.ndarray, shape (N_grid_points, n_modes)
        Spatial coordinate grid points where each column corresponds to mode q_i.
    omegas : np.ndarray, shape (n_modes,)
        Fundamental mode frequencies.
    eigenvalues : np.ndarray, shape (n_basis,)
        System energy eigenvalues E_k.
    eigenvectors : np.ndarray, shape (n_basis, n_basis)
        Unitary transformation matrix C_{n, k} mapping basis state n to eigenstate k.
        (Assumed: eigenvectors[:, k] is the k-th eigenstate vector).
    T : float
        Temperature.
    kb : float, optional
        Boltzmann constant (defaults to 1.0).
        
    Returns:
    --------
    rho : np.ndarray, shape (N_grid_points,)
        Thermal probability density distribution sum_k P_k |Psi_k(q)|^2.
    """
    n_basis = basis.shape[0]
    n_grid = qs.shape[0]
    
    # 1. Strictly sort eigenvalues and eigenvectors so k=0 is guaranteed to be E_0
    sort_idx = np.argsort(eigenvalues)
    evals_sorted = eigenvalues[sort_idx]
    
    # Ensure eigenvectors are indexed as columns evecs[:, k] for state k
    evecs_sorted = eigenvectors[:, sort_idx]
    
    # 2. Compute Boltzmann populations P_k
    if T == 0 or T is None:
        weights = np.zeros(n_basis, dtype=np.float64)
        weights[0] = 1.0  # k=0 is the TRUE system ground state Psi_0(q)
    else:
        beta = 1.0 / (kb * T)
        E_shift = evals_sorted - evals_sorted[0]
        weights = np.exp(-beta * E_shift)
        weights /= np.sum(weights)
        
    # 3. Evaluate all uncoupled product basis functions phi_n(q) on the grid
    # phi_basis shape: (n_basis, N_grid_points)
    phi_basis = np.empty((n_basis, n_grid), dtype=np.float64)
    for n_idx, ns in enumerate(basis):
        phi_basis[n_idx, :] = get_basis_function(ns, qs, omegas)
        
    # 4. Compute thermal density rho(q, T) = sum_k P_k * |sum_n C_{n,k} * phi_n(q)|^2
    rho = np.zeros(n_grid, dtype=np.float64)
    
    for k in range(n_basis):
        if weights[k] < 1e-15:
            continue
            
        # Extract the expansion coefficients C_{n, k} for physical eigenstate k
        C_k = evecs_sorted[:, k]
        
        # Physical eigenstate wavefunction Psi_k(q) = sum_n C_{n,k} * phi_n(q)
        Psi_k = C_k @ phi_basis  # Shape: (N_grid_points,)
        
        # Add contribution P_k * |Psi_k(q)|^2
        rho += weights[k] * (Psi_k**2)
            
    return rho

def get_Qnm(basis, eigenvectors, omegas):
    """
    Computes position matrix elements Q_nm in the eigenbasis.
    In Rydberg units (m_e = 1/2), Q = sqrt(hbar / omega) * (a + a^\dagger).
    """
    n_modes = basis.shape[1]
    n_basis = basis.shape[0]
    Qnm_harm = np.zeros([n_modes, n_basis, n_basis], dtype=np.float64)
    Qnm = np.zeros([n_modes, n_basis, n_basis], dtype=np.float64)
    
    for mu in range(n_modes):
        omega_mu = omegas[mu]
        for ni, ns in enumerate(basis):
            for mi, ms in enumerate(basis):
                # Check if all other modes remain unchanged
                diffs = ns - ms
                active_mode_diff = diffs[mu]
                other_modes_same = np.all(diffs[:mu] == 0) and np.all(diffs[mu+1:] == 0)
                
                if other_modes_same:
                    # <n| Q | n-1> = sqrt(hbar * n / omega)
                    if active_mode_diff == 1:
                        Qnm_harm[mu, ni, mi] = np.sqrt(hbar * ns[mu] / omega_mu)
                    # <n| Q | n+1> = sqrt(hbar * (n + 1) / omega)
                    elif active_mode_diff == -1:
                        Qnm_harm[mu, ni, mi] = np.sqrt(hbar * (ns[mu] + 1) / omega_mu)

        # Transform to exact eigenbasis via eigenvectors
        Qnm[mu] = eigenvectors.T @ Qnm_harm[mu] @ eigenvectors
        
    return Qnm


def get_correlation_QQ(t, mu, nu, T, eigenvalues, Qnm):
    """
    Computes time correlation function C_mu_nu(t) = <Q_mu(t) Q_nu(0)>.
    Uses numerically stable Boltzmann weights relative to ground state.
    """
    n_basis = len(eigenvalues)
    
    # Calculate populations relative to E_0 to prevent exponent overflow
    if T == 0 or T is None:
        weights = np.zeros(n_basis, dtype=np.float64)
        weights[0] = 1.0
    else:
        beta = 1.0 / (kb * T)
        E_shift = eigenvalues - eigenvalues[0]
        weights = np.exp(-beta * E_shift)
        weights /= np.sum(weights)

    qqt = 0.0 + 0j
    for n in range(n_basis):
        if weights[n] < 1e-15:
            continue
        for m in range(n_basis):
            omega_mn = (eigenvalues[m] - eigenvalues[n]) / hbar
            # C(t) = sum_nm P_n * <n|Q_mu|m> <m|Q_nu|n> * e^(i * w_mn * t)
            qqt += weights[n] * Qnm[mu, n, m] * Qnm[nu, m, n] * np.exp(1j * omega_mn * t)
            
    return qqt


def get_spectrum_J(mu, nu, T, eigenvalues, eigenvectors, basis, omegas, 
                   w_min=0.0, w_max=0.004, n_points=2000, eta=1e-5):
    """
    Computes the spectral density J(w) analytically in the frequency domain.
    Eliminates all FFT artifacts, time-grid truncation ripples, and negative values.
    """
    n_basis = len(eigenvalues)
    
    # 1. Calculate Qnm in eigenbasis once
    Qnm = get_Qnm(basis, eigenvectors, omegas)
    
    # 2. Calculate Boltzmann populations
    if T == 0 or T is None:
        weights = np.zeros(n_basis, dtype=np.float64)
        weights[0] = 1.0
    else:
        beta = 1.0 / (kb * T)
        E_shift = eigenvalues - eigenvalues[0]
        weights = np.exp(-beta * E_shift)
        weights /= np.sum(weights)

    # 3. Frequency axis setup
    w_plot = np.linspace(w_min, w_max, n_points)
    J_plot = np.zeros_like(w_plot, dtype=np.float64)

    # 4. Direct analytical summation over lorentzian-broadened transitions
    for n in range(n_basis):
        if weights[n] < 1e-15:
            continue
        for m in range(n_basis):
            w_mn = (eigenvalues[m] - eigenvalues[n]) / hbar
            
            # Transition dipole strength
            strength = weights[n] * Qnm[mu, n, m] * Qnm[nu, m, n]
            
            # Analytical delta-function representation (Lorentzian line profile)
            lorentzian = (1.0 / np.pi) * (eta / ((w_plot - w_mn)**2 + eta**2))
            
            J_plot += np.real(strength) * lorentzian

    # Guarantee non-negative physical output
    J_plot = np.maximum(J_plot, 0.0)

    return w_plot, J_plot

def get_spectrum_A(T, eigenvalues, eigenvectors, basis, omegas, 
                   w_min=0.0, w_max=0.004, n_points=2000, eta=1e-5):
    """
    Computes the response spectral function A(w) analytically in the frequency domain 
    for all modes using the Lehmann representation with Lorentzian broadening.
    """
    n_basis = len(eigenvalues)
    n_modes = basis.shape[1]
    
    # ------------------------------------------------------------------
    # 1. Compute Boltzmann thermal populations
    # ------------------------------------------------------------------
    if T == 0 or T is None:
        weights = np.zeros(n_basis, dtype=np.float64)
        weights[0] = 1.0
    else:
        beta = 1.0 / (kb * T)
        E_shift = eigenvalues - eigenvalues[0]
        weights = np.exp(-beta * E_shift)
        weights /= np.sum(weights)

    # Population differences (P_n - P_m) for absorption/emission balance
    pop_diff = weights[np.newaxis, :] - weights[:, np.newaxis] # P_n - P_m

    # ------------------------------------------------------------------
    # 2. Setup Frequency Axis and Transition Energies
    # ------------------------------------------------------------------
    w_plot = np.linspace(w_min, w_max, n_points)
    w_2d = w_plot[np.newaxis, :]  # shape (1, n_points)
    
    # Transition energy matrix w_mn = (E_m - E_n) / hbar
    w_mn_matrix = (eigenvalues[:, np.newaxis] - eigenvalues[np.newaxis, :]) / hbar
    
    A_modes = np.zeros((n_modes, n_points), dtype=np.float64)

    # Pre-calculate difference tensor across basis states once: shape (N, N, n_modes)
    diffs = basis[:, np.newaxis, :] - basis[np.newaxis, :, :]

    # ------------------------------------------------------------------
    # 3. Loop over all optical modes mu
    # ------------------------------------------------------------------
    for mu in range(n_modes):
        omega_m = omegas[mu]
        Q_harm_mu = np.zeros((n_basis, n_basis), dtype=np.float64)
        
        # Build mask where all OTHER modes remain unchanged
        other_modes_mask = np.ones((n_basis, n_basis), dtype=bool)
        for m_idx in range(n_modes):
            if m_idx != mu:
                other_modes_mask &= (diffs[:, :, m_idx] == 0)
                
        # Matrix elements: <n| Q | n-1> and <n| Q | n+1>
        creation_mask = other_modes_mask & (diffs[:, :, mu] == 1)
        annihilation_mask = other_modes_mask & (diffs[:, :, mu] == -1)
        
        # Broadcast quanta to full (N, N) matrix shape matching masks
        n_quanta_matrix = np.tile(basis[:, mu][:, np.newaxis], (1, n_basis))
        
        Q_harm_mu[creation_mask] = np.sqrt(hbar * n_quanta_matrix[creation_mask] / omega_m)
        Q_harm_mu[annihilation_mask] = np.sqrt(hbar * (n_quanta_matrix[annihilation_mask] + 1.0) / omega_m)

        # Transform to exact eigenbasis: Q_eigen = V^T @ Q_harm @ V
        Qnm_mu = eigenvectors.T @ Q_harm_mu @ eigenvectors
        
        # Transition probability weights: |<m| Q_mu |n>|^2 * (P_n - P_m)
        transition_weights = (np.abs(Qnm_mu.T)**2) * pop_diff

        # Direct analytical Lorentzian summation over transitions
        for n in range(n_basis):
            for m in range(n_basis):
                w_transition = w_mn_matrix[m, n]
                S = transition_weights[m, n]
                
                # Filter positive absorptive transitions
                if S <= 1e-15 or w_transition <= 0:
                    continue
                    
                lorentzian = (1.0 / np.pi) * (eta / ((w_2d - w_transition)**2 + eta**2))
                A_modes[mu] += S * lorentzian[0]

    # Total response is the sum across all modes
    A_total = np.sum(A_modes, axis=0)

    return w_plot, A_total, A_modes

def K_nm(ns,ms,omegas):
    n_modes = ns.shape[0]
    its_nzero = np.ones(n_modes, dtype=bool)
    for mu in range(n_modes):
        for h in range(n_modes):
            if ns[h] != ms[h]:
                if mu != h:
                    its_nzero[mu] = False
                    break
    
    kinetic = 0
    for mu in range(n_modes):
        if its_nzero[mu]:
            if ms[mu] == ns[mu]:
                kinetic += (hbar * (omegas[mu]) / 4) * (2*ms[mu]+1)
            elif ms[mu] == ns[mu]+2:
                kinetic += -(hbar * (omegas[mu]) / 4) * np.sqrt((ns[mu]+2)*(ns[mu]+1))
            elif ms[mu] == ns[mu]-2:
                kinetic += -(hbar * (omegas[mu]) / 4) * np.sqrt((ns[mu])*(ns[mu]-1))
    return kinetic

def V_nm(ns,ms,omegas,Qns,Vs,weights):

    phi_n = get_basis_function(ns,Qns,omegas)
    phi_m = get_basis_function(ms,Qns,omegas)
    
    # In principle, phis are real.
    integrand = np.conjugate(phi_n)*phi_m*Vs

    return np.sum(integrand*weights)

def construct_basis(nmax, decoupled, n_modes):
    """
    Constructs the vibrational basis ONLY for the coupled modes (where decoupled == 0).
    The returned array has shape (N_coupled_states, n_coupled), matching only the 
    number of coupled dimensions (zeros in the decoupled array).
    
    Parameters:
    -----------
    nmax : list or np.ndarray
        Maximum quantum number per mode (excluding acoustic modes).
    decoupled : list or np.ndarray
        Array where 1 = decoupled (pure harmonic), 0 = coupled.
    n_modes : int
        Total number of non-acoustic optical modes.
        
    Returns:
    --------
    coupled_basis : np.ndarray, shape (N_coupled_states, n_coupled)
        Reduced basis vectors containing quantum numbers only for coupled modes.
    """
    nmax_opt = np.array(nmax[:n_modes], dtype=np.int64)
    decoupled_opt = np.array(decoupled[:n_modes], dtype=np.int64)
    
    # Identify indices of coupled modes (where decoupled == 0)
    coupled_indices = np.where(decoupled_opt == 0)[0]
    
    nmax_coupled = nmax_opt[coupled_indices]
    n_coupled = len(coupled_indices)
    
    # If all modes are decoupled, return a single ground state for 0 coupled modes
    if n_coupled == 0:
        return np.zeros((1, 0), dtype=np.int64)

    coupled_basis = []

    # Distribute quanta among coupled modes only
    for k in range(np.sum(nmax_coupled) + 1):
        for comb in itertools.combinations_with_replacement(range(n_coupled), k):
            ns_coupled = np.zeros(n_coupled, dtype=np.int64)
            for mode_idx in comb:
                ns_coupled[mode_idx] += 1

            # Enforce mode-specific nmax for coupled modes
            if np.all(ns_coupled <= nmax_coupled):
                coupled_basis.append(ns_coupled)

    return np.array(coupled_basis, dtype=np.int64)


def diagonalize_hamiltonian(basis, ensemble, Qns, weights, model, no_sym):

    mapping, rot_cart, map_uc, map_tr, T_list, T_list_frac = Classify.map_singlet(ensemble.current_dyn)

    nsym = rot_cart.shape[0]
    nconf = Qns.shape[0]
    nat = ensemble.current_dyn.structure.N_atoms

    n_states = basis.shape[0]

    super_dyn = ensemble.current_dyn.GenerateSupercellDyn(ensemble.supercell)
    
    masses_sc = super_dyn.structure.get_masses_array()
    masses_sc_flat = np.repeat(masses_sc, 3)

    ws, pols = ensemble.current_dyn.DiagonalizeSupercell()

    u_disps = Qns @ pols.T[3:4,:] / np.sqrt(masses_sc_flat)
    energies = model.get_energies(u_disps)

    # if not no_sym:
    #     sym_u_disps = np.empty([nconf*nsym,u_disps.shape[1]], dtype=np.float64)
    #     sym_energies = np.empty([nconf*nsym], dtype=np.float64)
    #     sym_weights = np.empty([nconf*nsym], dtype=np.int64)

    #     for ri,rot in enumerate(rot_cart):
    #         for si,dR in enumerate(u_disps):
    #             for ai in range(nat):
    #                 sym_u_disps[ri*nconf+si,mapping[ai,ri]*3:(mapping[ai,ri]+1)*3] = rot @ dR[ai*3:(ai+1)*3]
    #         sym_energies[ri*nconf:(ri+1)*nconf] = energies
    #         sym_weights[ri*nconf:(ri+1)*nconf] = weights
    #     sym_Qns = (sym_u_disps @ pols * np.sqrt(masses_sc_flat))[:,5:]

    H = np.zeros((n_states, n_states), dtype=np.float64)

    for ni in range(n_states):
        for mj in range(n_states):
            ns = basis[ni]
            ms = basis[mj]
            if no_sym:
                H_ij = K_nm(ns,ms,ws[3:4]) + V_nm(ns,ms,ws[3:4],Qns,energies,weights)
            # else:
            #     H_ij = K_nm(ns,ms,ws[5:]) + V_nm(ns,ms,ws[5:],sym_Qns,sym_energies,sym_weights)
            
            H[ni,mj] = H_ij
            if ni != mj:
                H[mj,ni] = H_ij
    eigenvalues, eigenvectors = eigh(H)

    return eigenvalues, eigenvectors, H

def get_eigenvalues(basis, omega):
    """
    Calculates the total energy (eigenvalues) for a set of quantum states.
    Uses the global variable `hbar` to scale the frequencies correctly.
    
    Parameters:
    -----------
    basis : array-like of shape (N, D)
        The quantum numbers for N basis states across D independent oscillators.
    omega : array-like of shape (D,)
        The angular frequencies of the D independent harmonic oscillators.
        
    Returns:
    --------
    eigenvalues : np.ndarray of shape (N,)
        The total energy for each of the N basis states.
    """
    
    # Convert to numpy arrays to ensure vectorized math works perfectly
    basis_array = np.array(basis, dtype=np.float64)
    omega_array = np.array(omega, dtype=np.float64)
    
    # Calculate E = sum( hbar * omega_i * (n_i + 0.5) ) across the D dimension (axis=1)
    eigenvalues = np.sum(hbar * omega_array * (basis_array + 0.5), axis=1)
    
    return eigenvalues

def get_new_correction(model, ensemble, T, n_max, decoupled, N, step_size, alpha_mix=0.01, no_sym=False):

    decoupled = np.asarray(decoupled)

    super_dyn = ensemble.current_dyn.GenerateSupercellDyn(ensemble.supercell)
    nat_sc = super_dyn.structure.N_atoms
    n_modes = nat_sc*3

    w, pols = ensemble.current_dyn.DiagonalizeSupercell()

    w_coupled = w[decoupled==0]
    basis_coupled = construct_basis(n_max, decoupled, n_modes)
    print("Iteration 0")

    weighted_samples, weights = build_general_quadrature_grid(basis_coupled,w_coupled)

    eigenvalues1, eigenvectors1, H = diagonalize_hamiltonian(basis_coupled, ensemble, weighted_samples, weights, model, no_sym)
    breakpoint()
    eigenvalues1, eigenvectors1, basis = expand_decoupled_system(eigenvalues1,eigenvectors1,basis_coupled,n_max[3:],decoupled[3:],w[3:])

    mapping, rot_cart, map_uc, map_tr, T_list, T_list_frac = Classify.map_singlet(ensemble.current_dyn)

    nsym = rot_cart.shape[0]
    nconf = weighted_samples.shape[0]
    nat = ensemble.current_dyn.structure.N_atoms

    super_dyn = ensemble.current_dyn.GenerateSupercellDyn(ensemble.supercell)
    
    masses_sc = super_dyn.structure.get_masses_array()
    masses_sc_flat = np.repeat(masses_sc, 3)

    u_disps = weighted_samples @ pols.T[5:,:] / np.sqrt(masses_sc_flat)
    if not no_sym:
        sym_u_disps = np.empty([nconf*nsym,u_disps.shape[1]], dtype=np.float64)
        sym_rphi2 = np.empty([nconf*nsym], dtype=np.float64)
        sym_weights = np.empty([nconf*nsym], dtype=np.int64)

        for ri,rot in enumerate(rot_cart):
            for si,dR in enumerate(u_disps):
                for ai in range(nat):
                    sym_u_disps[ri*nconf+si,mapping[ai,ri]*3:(mapping[ai,ri]+1)*3] = rot @ dR[ai*3:(ai+1)*3]
            #sym_rphi2[ri*nconf:(ri+1)*nconf] = rphi2
            sym_weights[ri*nconf:(ri+1)*nconf] = weights
        sym_weighted_samples = (sym_u_disps @ pols * np.sqrt(masses_sc_flat))[:,3:]

    model.plot_quartic_phi(eigenvalues1,eigenvectors1,basis,w,pols,weighted_samples,weights,T)
    print("")


    return eigenvalues1, eigenvectors1, basis, H


def razavy_potential(x, xi=1.0, j=2, lambda_=1.0, scale=1.0):
    """
    Computes the Razavy potential with coordinate scaling (lambda_) 
    and energy scaling (scale).
    """
    u = lambda_ * x
    V = (xi**2 / 8.0) * np.cosh(4*u) - (xi * (2*j + 1) / 2.0) * np.cosh(2*u)
    return scale * V

def razavy_first_deriv(x, xi=1.0, j=2, lambda_=1.0, scale=1.0):
    """
    Computes the first derivative (Force) of the Razavy potential.
    """
    u = lambda_ * x
    deriv = lambda_ * ((xi**2 / 2.0) * np.sinh(4*u) - (xi * (2*j + 1)) * np.sinh(2*u))
    return scale * deriv

def razavy_second_deriv(x, xi=1.0, j=2, lambda_=1.0, scale=1.0):
    """
    Computes the second derivative (Hessian/Curvature) of the Razavy potential.
    """
    u = lambda_ * x
    deriv2 = (lambda_**2) * ((2 * xi**2) * np.cosh(4*u) - (2 * xi * (2*j + 1)) * np.cosh(2*u))
    return scale * deriv2

def octic_potential(x, a, b, c, d):
    """
    V(x) = a*x^2 + b*x^4 + c*x^6 + d*x^8
    """
    return a*x**2 + b*x**4 + c*x**6 + d*x**8

def octic_first_deriv(x, a, b, c, d):
    """
    Force related: dV/dx = 2*a*x + 4*b*x^3 + 6*c*x^5 + 8*d*x^7
    """
    return 2*a*x + 4*b*x**3 + 6*c*x**5 + 8*d*x**7

def octic_second_deriv(x, a, b, c, d):
    """
    Hessian (Curvature): d^2V/dx^2 = 2*a + 12*b*x^2 + 30*c*x^4 + 56*d*x^6
    """
    return 2*a + 12*b*x**2 + 30*c*x**4 + 56*d*x**6

def harmonic_potential(x, omega=1.0):
    return 0.5 * omega**2 * x**2
                                   
def harmonic_first_deriv(x, omega=1.0):
    return omega**2 * x

def ensemble_from_model(model, dyn, N=100, Temperature=0, pop_id=1):
    dyn_sscha_final=CC.Phonons.Phonons(dyn, nqirr=1)
    dyn_sscha_final.ForcePositiveDefinite()

    ensemble = sscha.Ensemble.Ensemble(dyn_sscha_final, T0=Temperature, supercell = dyn_sscha_final.GetSupercell())
    ensemble.generate(N)
    ensemble.save_bin("./data/", pop_id)

    ensemble.xats = np.load("%s/xats_pop%d.npy" % ("./data/", pop_id))
    ensemble.N = len(ensemble.xats)

    super_structure = ensemble.dyn_0.structure.generate_supercell(ensemble.supercell)
    Nat_sc = super_structure.N_atoms

    ensemble.u_disps = np.zeros( (ensemble.N, 3 * Nat_sc), order = "F", dtype = np.float64)
    # Build the structures
    ensemble.structures = [None] * ensemble.N
    for i in range(ensemble.N):
        ensemble.structures[i] = super_structure.copy()
        ensemble.structures[i].coords = ensemble.xats[i,:,:]
        ensemble.u_disps[i, :] = (ensemble.xats[i, :, :] - super_structure.coords).reshape( 3*Nat_sc )
    
    ensemble.energies = model.get_energies(ensemble.u_disps*angstroms2bohr) # Ry
    ensemble.forces = np.reshape(model.get_forces(ensemble.u_disps*angstroms2bohr)*angstroms2bohr,[N,Nat_sc,3]) # Ry/A

    ensemble.sscha_energies = np.zeros(ensemble.N, dtype = np.float64)
    ensemble.sscha_forces = np.zeros( (ensemble.N, Nat_sc, 3), order = "F", dtype = np.float64)

    # Initialize everything for running the minimization faster.
    ensemble.init()

    # Setup the initial weights
    ensemble.rho = np.ones(ensemble.N, dtype = np.float64)

    # Setup that both forces and stresses are not computed
    ensemble.stress_computed = np.ones(ensemble.N, dtype = bool)
    ensemble.force_computed = np.ones(ensemble.N, dtype = bool)

    return ensemble


def build_general_quadrature_grid(basis, omegas):
    """
    Builds a D-dimensional Gauss-Hermite quadrature grid directly in 
    physical mass-weighted normal coordinates Q (Bohr * sqrt(mass)) 
    for Rydberg atomic units.
    
    Parameters:
    -----------
    basis : np.ndarray
        Array of shape (N_states, n_modes) containing quantum number vectors.
    omegas : np.ndarray or list
        The fundamental frequencies for each mode in Rydberg energy units.
    """
    basis = np.asarray(basis)
    omegas = np.asarray(omegas)
    
    n_modes = basis.shape[1]      
    n_max = np.max(basis, axis=0)          
    
    
    grids_list = []
    weights_list = []
    
    for mu in range(n_modes):
        # Required points per mode (safe up to quartic potentials, d_V = 4)
        n_points = int(np.ceil(n_max[mu] + (4 + 1) / 2))
        
        # Get standard mathematical roots (x) and weights (w) for e^(-x^2)
        x_std, w_std = np.polynomial.hermite.hermgauss(n_points)

        omega = omegas[mu]
        
        # 1. Physical mass-weighted normal coordinate mapping (Rydberg)
        q_physical = x_std / np.sqrt(omega)
        
        # 2. Correct weight stripping with the transformation Jacobian dQ = dx/sqrt(2*omega)
        jacobian = 1.0 / np.sqrt(omega)
        w_physical = w_std * jacobian * np.exp(x_std**2)
        
        grids_list.append(q_physical)
        weights_list.append(w_physical)
    
    # Multi-dimensional grid assembly via meshgrid
    mesh_grids = np.meshgrid(*grids_list, indexing='ij')
    weight_grids = np.meshgrid(*weights_list, indexing='ij')
    
    # Flatten and stack into your final coordinate matrix
    grid_nd = np.stack([g.ravel() for g in mesh_grids], axis=1)
    
    # Total volume weight element (dQ1 * dQ2 * ...)
    weights_nd = weight_grids[0].ravel()
    for w_grid in weight_grids[1:]:
        weights_nd *= w_grid.ravel()
        
    return grid_nd, weights_nd

def get_reference_rho(a, b, c, d, T=0.0, x_max=200, N=1000):
    """
    Solves the 1D Schrödinger equation for an octic potential and returns
    strictly the spatial coordinate grid and thermal probability density rho(x, T).
    
    Inputs:
        a, b, c, d: Coefficients for octic_potential(x, a, b, c, d)
        T: Temperature in matching units (defaults to 0.0)
        kb: Boltzmann constant
        hbar: Reduced Planck's constant
        x_max: Grid boundary
        N: Number of grid points
    
    Returns:
        x: 1D spatial coordinate grid
        rho_thermal: Spatial probability density distribution rho(x, T)
    """
    # 1. Spatial grid setup
    x = np.linspace(-x_max, x_max, N)
    dx = x[1] - x[0]
    
    # 2. Hamiltonian construction
    main_diag = np.ones(N) * (-2.0)
    off_diag = np.ones(N - 1) * 1.0
    D2 = (np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)) / (dx**2)
    T_mat = -0.5 * (hbar**2) * D2
    V_mat = np.diag(octic_potential(x, a, b, c, d))
    
    H = T_mat + V_mat
    
    # 3. Solve for all eigenvalues and eigenvectors
    energies, eigenvectors = eigh(H)
    
    # 4. Compute continuous wavefunctions psi_k(x) = eigenvector_k / sqrt(dx)
    psi_all = eigenvectors / np.sqrt(dx)
    
    # 5. Compute thermal probability density rho(x, T)
    if T == 0 or T is None:
        # At T=0, return pure ground state density |psi_0(x)|^2
        rho_thermal = psi_all[:, 0]**2
    else:
        # Boltzmann weights P_k = exp(-(E_k - E_0) / k_B T) / Z
        beta = 1.0 / (kb * T)
        E_shift = energies - energies[0]
        weights = np.exp(-beta * E_shift)
        weights /= np.sum(weights)
        
        # Matrix-vector product: sum_k P_k * |psi_k(x)|^2
        rho_thermal = (psi_all**2) @ weights
        
    return x, rho_thermal, energies[0]

def expand_decoupled_system(evals_coupled, evecs_coupled, basis_coupled, nmax, decoupled, omegas, hbar=1.0):
    """
    Expands the eigenvalues, eigenvectors, and basis vectors from the reduced coupled subspace 
    to the full system space using analytical harmonic spectra for decoupled modes.
    Includes zero-point energy (ZPE) for decoupled modes.
    
    Parameters:
    -----------
    evals_coupled : np.ndarray, shape (N_coupled,)
        Eigenvalues from diagonalizing the coupled block.
    evecs_coupled : np.ndarray, shape (N_coupled, N_coupled)
        Eigenvectors from diagonalizing the coupled block.
    basis_coupled : np.ndarray, shape (N_coupled, n_coupled)
        Reduced coupled basis containing quantum numbers ONLY for coupled modes.
    nmax : list or np.ndarray
        Maximum quantum number per mode [nmax_0, nmax_1, ...].
    decoupled : list or np.ndarray
        Mask where 1 = decoupled (pure harmonic), 0 = coupled.
    omegas : np.ndarray
        Frequencies for all optical modes.
    hbar : float, optional
        Planck's reduced constant (default 1.0).
        
    Returns:
    --------
    evals_full : np.ndarray, shape (N_full,)
        Expanded energy eigenvalues (with decoupled ZPE and excitations).
    evecs_full : np.ndarray, shape (N_full, N_full)
        Expanded block-diagonal eigenvector matrix.
    basis_full : np.ndarray, shape (N_full, n_modes)
        Complete basis quantum number array for all states.
    """
    decoupled_arr = np.array(decoupled, dtype=int)
    decoupled_indices = np.where(decoupled_arr == 1)[0]
    coupled_indices = np.where(decoupled_arr == 0)[0]
    
    n_modes = len(decoupled)
    
    # 1. Slice frequencies using boolean/index selection
    omegas_decoupled = omegas[decoupled_indices]
    
    # 2. Constant Zero-Point Energy (ZPE) of all decoupled modes
    E_zpe = np.sum(0.5 * hbar * omegas_decoupled)
    
    # 3. Generate all quantum number combinations for decoupled modes
    dec_ranges = [range(nmax[m] + 1) for m in decoupled_indices]
    dec_combinations = list(itertools.product(*dec_ranges))
    
    n_blocks = len(dec_combinations)
    n_coupled = len(evals_coupled)
    
    evals_full = []
    basis_full = []
    
    # Block-diagonal eigenvector allocation
    n_full = n_blocks * n_coupled
    evecs_full = np.zeros((n_full, n_full), dtype=np.float64)
    
    # 4. Expand system state by state
    for b_idx, dec_state in enumerate(dec_combinations):
        # Excitation energy + ZPE: sum( n_i * hbar * w_i ) + E_zpe
        E_offset = E_zpe + np.sum(np.array(dec_state) * hbar * omegas_decoupled)
        
        # Expand Eigenvalues
        evals_full.append(evals_coupled + E_offset)
        
        # Expand Basis Quantum Numbers into full n_modes dimension
        for coupled_row in basis_coupled:
            full_state = np.zeros(n_modes, dtype=np.int64)
            full_state[decoupled_indices] = dec_state
            full_state[coupled_indices] = coupled_row
            basis_full.append(full_state)
            
        # Fill block-diagonal eigenvector matrix
        start = b_idx * n_coupled
        end = (b_idx + 1) * n_coupled
        evecs_full[start:end, start:end] = evecs_coupled

    evals_full = np.concatenate(evals_full)
    basis_full = np.array(basis_full, dtype=np.int64)
    
    return evals_full, evecs_full, basis_full