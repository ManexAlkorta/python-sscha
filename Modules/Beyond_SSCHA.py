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

    def __init__(self, a, b, c, d, omega=9.1127e-4/2):
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

        self.omega = omega

        self.frequencies = [0,0,0,self.omega,self.omega,np.sqrt(octic_second_deriv(0,self.a,self.b,self.c,self.d)+0j)]
        self.eigenvalues = [0,0,0,self.omega**2,self.omega**2,octic_second_deriv(0,self.a,self.b,self.c,self.d)]

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
        energies = octic_potential(Qns[:,5],self.a,self.b,self.c,self.d) + harmonic_potential(Qns[:,3],self.omega) + harmonic_potential(Qns[:,4],self.omega)
        # energies = harmonic_potential(Qns[:,5],self.omega) + harmonic_potential(Qns[:,3],self.omega) + harmonic_potential(Qns[:,4],self.omega)
        return energies
    
    def get_forces(self, disps):
        Qns = disps @ self.pols.T * np.sqrt(self.masses[0])
        forces = (-1)*(np.outer(octic_first_deriv(Qns[:,5],self.a,self.b,self.c,self.d),self.pols[:,5]) + np.outer(harmonic_first_deriv(Qns[:,3],self.omega),self.pols[3]) + np.outer(harmonic_first_deriv(Qns[:,4],self.omega),self.pols[4]))
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

    def plot_quartic_phi(self, eigenvalues, eigenvectors, basis, omegas, pols, weighted_samples, weights):
        disps3 = np.outer(np.linspace(-2,2,200),pols[:,3])
        disps4 = np.outer(np.linspace(-2,2,200),pols[:,4])
        disps5 = np.outer(np.linspace(-2,2,200),pols[:,5])

        Q3 = disps3 @ pols * np.sqrt(self.masses[0])
        Q4 = disps4 @ pols * np.sqrt(self.masses[0])
        Q5 = disps5 @ pols * np.sqrt(self.masses[0])

        phi3 = np.zeros(Q3.shape[0])
        phi4 = np.zeros(Q4.shape[0])
        phi5 = np.zeros(Q5.shape[0])

        for n in range(basis.shape[0]):
            phi3 += eigenvectors[n,0] * get_basis_function(basis[n],Q3[:,3:],omegas[3:])
            phi4 += eigenvectors[n,0] * get_basis_function(basis[n],Q4[:,3:],omegas[3:])
            phi5 += eigenvectors[n,0] * get_basis_function(basis[n],Q5[:,3:],omegas[3:])

        phi3_0 = get_basis_function([0,0,0],Q3[:,3:],omegas[3:])
        phi4_0 = get_basis_function([0,0,0],Q4[:,3:],omegas[3:])
        phi5_0 = get_basis_function([0,0,0],Q5[:,3:],omegas[3:])

        x, energy_g, phi_g = self.get_reference_ground_state()
        
        print(f"E(n_max)={eigenvalues[0]}")
        print(f"E(inf)={energy_g}")

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
        
        f2.fill_between(Q3[:,3], 0, phi3**2, color="red", alpha=0.5)
        f4.fill_between(Q4[:,4], 0, phi4**2, color="red", alpha=0.5)
        f6.fill_between(Q5[:,5], 0, phi5**2, color="red", alpha=0.5)
        f2.plot(Q3[:,3], phi3_0**2*(phi3.max()**2/phi3_0.max()**2), color="black", linestyle="dashed")
        f4.plot(Q4[:,4], phi4_0**2*(phi4.max()**2/phi4_0.max()**2), color="black", linestyle="dashed")
        f6.plot(Q5[:,5], phi5_0**2, color="black", linestyle="dashed")
        f6.plot(x, phi_g**2/phi_g[500]**2*phi5[100]**2, color="red", linestyle="dashed")

        breakpoint()
        f2b.hist(weighted_samples[:,0],50,weights=weights)
        f4b.hist(weighted_samples[:,1],50,weights=weights)
        f6b.hist(weighted_samples[:,2],50,weights=weights)

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

    def get_reference_ground_state(self, x_max=200, N=1000):
        """
        Solves the 1D Schrödinger equation in the normal mode basis.
        Mass does not appear in the equation, but hbar is explicitly maintained.
        
        Inputs:
            a, b, c, d: Coefficients for the octic potential
            hbar: The explicit value of Planck's constant / 2pi being used in your method
            x_max: Grid boundary
            N: Number of grid points
        Returns:
            x: 1D coordinate grid
            ground_energy: Exact ground state energy
            psi_ground: Normalized continuous wavefunction array
        """
        # 1. Setup the spatial grid in normal coordinates
        x = np.linspace(-x_max, x_max, N)
        dx = x[1] - x[0]
        
        # 2. Construct Kinetic Energy Matrix (T)
        # Mass is gone, but hbar^2 remains explicitly in the numerator
        main_diag = np.ones(N) * (-2.0)
        off_diag = np.ones(N - 1) * 1.0
        D2 = (np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)) / (dx**2)
        
        T = -0.5 * (hbar**2) * D2
        
        # 3. Construct Potential Energy Matrix (V)
        V = np.diag(octic_potential(x, self.a, self.b, self.c, self.d))
        
        # 4. Total Hamiltonian
        H = T + V
        
        # 5. Solve the Eigenproblem
        energies, eigenvectors = eigh(H)
        
        # 6. Extract and properly normalize the ground state
        ground_energy = energies[0]
        psi_ground = eigenvectors[:, 0] / np.sqrt(dx)
        
        # Enforce a consistent positive phase at the origin
        if psi_ground[N // 2] < 0:
            psi_ground = -psi_ground
            
        return x, ground_energy, psi_ground


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

def get_Qnm(basis, eigenvectors, omegas):
    n_modes = basis.shape[1]
    n_basis = basis.shape[0]
    Qnm_harm = np.empty([n_modes,n_basis,n_basis], dtype=np.float64)
    Qnm = np.empty([n_modes,n_basis,n_basis], dtype=np.float64)
    for mu in range(n_modes):
        for ni, ns in enumerate(basis):
            for mi, ms in enumerate(basis):
                its_active = True
                for nu in range(n_modes):
                    if its_active:
                        if nu != mu:
                            if ns[nu]!=ms[nu]:
                                Qnm_harm[mu,ni,mi] = 0
                                its_active = False
                        else:
                            if ns[mu] == ms[mu]+1: 
                                Qnm_harm[mu,ni,mi] = np.sqrt(hbar*(ms[mu]+1)/(omegas[mu]))
                            if ns[mu] == ms[mu]-1: 
                                Qnm_harm[mu,ni,mi] = np.sqrt(hbar*ms[mu]/(omegas[mu]))
                            else:
                                Qnm_harm[mu,ni,mi] = 0
                                its_active = False

        Qnm[mu] = eigenvectors.T @ Qnm_harm[mu] @ eigenvectors
    return Qnm

def get_correlation_QQ(t, mu, nu, T, eigenvalues, Qnm):
    """Computes C(t) quickly by reusing precalculated Qnm matrix."""
    # Handle T=0 safely to avoid division by zero or large exponentials
    if T == 0:
        # At T=0, only the ground state (n=0) is populated
        # Z = 1, and exp(-E_0 / kT) is the only active starting state
        qqt = 0.0 + 0j
        n = 0 
        for m in range(len(eigenvalues)):
            omega_mn = (eigenvalues[m] - eigenvalues[n]) / hbar
            qqt += np.exp(1j * omega_mn * t) * Qnm[mu, n, m] * Qnm[nu, m, n]
        return qqt
    
    # At T > 0, use full Boltzmann weights
    weights = np.exp(-eigenvalues / (kb * T))
    Z = np.sum(weights)
    
    n_basis = len(eigenvalues)
    qqt = 0.0 + 0j
    for n in range(n_basis):
        for m in range(n_basis):
            omega_mn = (eigenvalues[m] - eigenvalues[n]) / hbar
            qqt += (weights[n] / Z) * np.exp(1j * omega_mn * t) * Qnm[mu, n, m] * Qnm[nu, m, n]
            
    return qqt

def get_spectrum_J(mu, nu, T, eigenvalues, eigenvectors, basis, omegas, precision_factor=4.0):
    """
    Physically extends t_max to narrow the peaks and increase the true frequency 
    resolution. 
    
    Set precision_factor=4.0 or 8.0 for extremely high precision.
    """
    n_basis = len(eigenvalues)
    
    # 1. PRECALCULATE Qnm ONCE
    Qnm = get_Qnm(basis, eigenvectors, omegas)
    
    # 2. CALCULATE ENERGY GAPS
    gaps = []
    for i in range(n_basis):
        for j in range(i + 1, n_basis):
            gaps.append(abs(eigenvalues[j] - eigenvalues[i]))
    gaps = np.array(gaps)
    
    max_gap = np.max(gaps) / hbar
    min_gap = np.min(gaps[gaps > 1e-6]) / hbar 
    
    # 3. DEFINE HIGH-PRECISION GRID LIMITS
    # dt stays sharp to catch the fastest oscillations
    dt = np.pi / (2 * max_gap) 
    
    # t_max is physically expanded to squeeze peaks narrower and density higher
    t_max = ((2 * np.pi / min_gap) * 2.0) * precision_factor        
    
    t_array = np.arange(0, t_max, dt)
    
    # Damping naturally scales down so your signal physically lives longer!
    eta = 3.0 / t_max 
    
    print(f"Calculating high-precision spectrum...")
    print(f"Physical t_max extended to: {t_max:.2f}")
    print(f"Total time steps to compute: {len(t_array)}")
    
    # 4. COMPUTE CORRELATION FUNCTION USING THE FAST PATHWAY
    C_t = np.array([
        get_correlation_QQ(t, mu, nu, T, eigenvalues, Qnm) 
        for t in t_array
    ])
    
    # 5. FOURIER TRANSFORM PIPELINE
    damping = np.exp(-(eta * t_array)**2)
    C_t_damped = C_t * damping
    
    J_raw = np.fft.fft(C_t_damped) * dt
    frequencies = np.fft.fftfreq(len(t_array), d=dt) * 2 * np.pi
    
    freq_shifted = np.fft.fftshift(frequencies)
    J_shifted = np.fft.fftshift(J_raw)
    
    # 6. FOCUS PLOT STRICTLY ON YOUR WINDOW (0 to 0.004)
    zoom_mask = (freq_shifted >= 0.0) & (freq_shifted <= 0.004)
    
    w_plot = freq_shifted[zoom_mask]
    J_plot = np.real(J_shifted[zoom_mask])
    
    return w_plot, J_plot

            
def sample_wavefunction_unique(
    basis, 
    eigenvectors1, eigenvalues1, 
    eigenvectors0, eigenvalues0, 
    alpha_mix, 
    temperature, 
    omegas, 
    N_unique_targets, 
    step_size=0.5, # Your input step size (static for production)
    target_rate=0.44, 
):

    # 1. Temperature & Boltzmann Setup
    if temperature <= 1e-6:
        w1 = np.zeros_like(eigenvalues1)
        w1[np.argmin(eigenvalues1)] = 1.0
        w0 = np.zeros_like(eigenvalues0)
        w0[np.argmin(eigenvalues0)] = 1.0
    else:
        beta = 1.0 / (kb * temperature)
        E1_shifted = eigenvalues1 - np.min(eigenvalues1)
        w1 = np.exp(-beta * E1_shifted) / np.sum(np.exp(-beta * E1_shifted))
        E0_shifted = eigenvalues0 - np.min(eigenvalues0)
        w0 = np.exp(-beta * E0_shifted) / np.sum(np.exp(-beta * E0_shifted))

    # 2. Probability Evaluator
    def get_explicit_thermal_probability(q_coord):
        phi_vals = np.array([get_basis_function(ns, q_coord, omegas)[0] for ns in basis])
        psi1 = np.dot(phi_vals, eigenvectors1)
        psi0 = np.dot(phi_vals, eigenvectors0)
        rho1 = np.sum(w1 * np.real(psi1 * np.conj(psi1)))
        rho0 = np.sum(w0 * np.real(psi0 * np.conj(psi0)))
        return alpha_mix * rho1 + (1.0 - alpha_mix) * rho0

    # Parse inputs
    if isinstance(step_size, (int, float)):
        input_step_sizes = np.array([step_size, step_size, step_size], dtype=np.float64)
    else:
        input_step_sizes = np.array(step_size, dtype=np.float64)

    # --- PHASE 1: ACTIVE TUNING (To find the real optimal step size) ---
    burn_in_steps = 1000
    print(f"Running {burn_in_steps} steps of adaptive burn-in to find optimal step sizes...")
    
    q_current = np.zeros((1, 3), dtype=np.float64)
    p_current = get_explicit_thermal_probability(q_current)
    
    # During burn-in, we actively scale this copy so it can converge properly
    active_sigmas = np.copy(input_step_sizes)
    log_sigmas = np.log(active_sigmas)
    
    for t in range(1, burn_in_steps + 1):
        learning_rate = 1.0 / (t ** 0.6)
        
        for d in range(3):
            q_proposal = np.copy(q_current)
            # Use the shifting active_sigmas here so the feedback loop works!
            q_proposal[0, d] += np.random.normal(0, active_sigmas[d])
            p_proposal = get_explicit_thermal_probability(q_proposal)
            
            if np.asarray(p_current).item() == 0.0:
                ratio = 1.0
            else:
                ratio = np.asarray(p_proposal / p_current).item()
                
            accept_prob = min(1.0, ratio)
            accepted = False
            if np.random.uniform(0, 1) < accept_prob:
                q_current = np.real(q_proposal) 
                p_current = p_proposal
                accepted = True
            
            # Adjust the active step size based on success
            log_sigmas[d] += learning_rate * (int(accepted) - target_rate)
            active_sigmas[d] = np.clip(np.exp(log_sigmas[d]), 1e-5, 100.0)

    # --- PHASE 2: PRINT THE COMPARISON ---
    print("\n" + "="*55)
    print("           MCMC STEP SIZE DIAGNOSTICS")
    print("="*55)
    print(f"Target Acceptance Rate: {target_rate:.1%}")
    print(f"Dimension | Input Step Size | Suggested Step Size (for 44%)")
    print(f"   X     |    {input_step_sizes[0]:.6f}     |     {active_sigmas[0]:.6f}")
    print(f"   Y     |    {input_step_sizes[1]:.6f}     |     {active_sigmas[1]:.6f}")
    print(f"   Z     |    {input_step_sizes[2]:.6f}     |     {active_sigmas[2]:.6f}")
    print("="*55 + "\n")
    
    # --- PHASE 3: PRODUCTION RUN (Forced to use your raw Input Step Sizes) ---
    print(f"Running production with STATIC step sizes {input_step_sizes}...")
    
    unique_list = []
    weights_list = []
    probs_list = []  
    
    current_unique_q = np.copy(q_current[0])
    current_prob = p_current
    current_weight = 1
    
    total_accepted = 0
    total_steps = 0
    
    while len(unique_list) < N_unique_targets:
        for d in range(3):
            total_steps += 1
            q_proposal = np.copy(q_current)
            
            # CRITICAL: We use input_step_sizes here, completely ignoring the suggestions!
            q_proposal[0, d] += np.random.normal(0, input_step_sizes[d])
            p_proposal = get_explicit_thermal_probability(q_proposal)
            
            if np.asarray(p_current).item() == 0.0:
                ratio = 1.0
            else:
                ratio = np.asarray(p_proposal / p_current).item()
                
            if np.random.uniform(0, 1) < ratio:
                unique_list.append(current_unique_q)
                weights_list.append(current_weight)
                probs_list.append(current_prob)
                
                q_current = np.real(q_proposal)
                p_current = p_proposal
                current_unique_q = np.copy(q_current[0])
                current_prob = p_current
                current_weight = 1
                
                total_accepted += 1
            else:
                current_weight += 1
                
            if len(unique_list) >= N_unique_targets:
                break

    unique_qs = np.array(unique_list, dtype=np.float64)
    weights = np.array(weights_list)
    prob_densities = np.array(probs_list)
    
    avg_acceptance = total_accepted / total_steps
    print(f"MCMC Complete. Collected {N_unique_targets} unique points.")
    print(f"Final Production Acceptance Rate (with static input steps): {avg_acceptance:.1%}")
    
    return unique_qs, weights, prob_densities

def do_check(ns, omegas):
    phi = 1.0
    pts_per_axis = 100
    
    # 1. Dynamically compute the width (sigma) for each individual mode
    sigma0 = np.sqrt(hbar / omegas[0])
    sigma1 = np.sqrt(hbar / omegas[1])
    sigma2 = np.sqrt(hbar / omegas[2])
    
    # 2. Build custom axes that extend exactly to 6 * sigma for each mode
    q1_axis = np.linspace(-6 * sigma0, 6 * sigma0, pts_per_axis)
    q2_axis = np.linspace(-6 * sigma1, 6 * sigma1, pts_per_axis)
    q3_axis = np.linspace(-6 * sigma2, 6 * sigma2, pts_per_axis)
    
    # 3. Mesh and stack as normal
    Q1, Q2, Q3 = np.meshgrid(q1_axis, q2_axis, q3_axis)
    qs = np.column_stack([Q1.ravel(), Q2.ravel(), Q3.ravel()])
    
    # 4. Evaluate the wavefunctions
    for ni, n in enumerate(ns):
        phi *= independent_harm_osc(n, qs[:, ni], omegas[ni])
        
    # 5. Run the check
    check_normalization(phi, qs)


def check_normalization(phi,qs):
    """
    Computes the total integral of |phi|^2 over the space defined by qs.
    Inputs:
        phi: 1D array of wavefunction values, shape (N_snapshots,)
        qs:  2D array of coordinates, shape (N_snapshots, N_modes)
    """
    num_modes = qs.shape[1]
    dV = 1.0
    
    # Automatically calculate the step size (dq) for each individual mode
    for i in range(num_modes):
        unique_coords = np.unique(qs[:, i])
        if len(unique_coords) < 2:
            raise ValueError(f"Mode {i} does not have enough grid points to integrate.")
        
        # Distance between two adjacent grid points
        dq = unique_coords[1] - unique_coords[0]
        dV *= dq
    
    # Total Integral = sum( |phi|^2 ) * dV
    total_integral = np.sum(np.abs(phi)**2) * dV
    
    print("--- Standalone Integration Test ---")
    print(f"Number of modes detected: {num_modes}")
    print(f"Calculated Integral:       {total_integral:.6f}")
    return total_integral

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

def V_nm_st(ns,ms,omegas,Qns,Vs,weights,rphi2):

    phi_n = get_basis_function(ns,Qns,omegas)
    phi_m = get_basis_function(ms,Qns,omegas)

    # In principle, phis are real.
    integrand = np.conjugate(phi_n)*phi_m*Vs/(rphi2)*weights

    return np.sum(integrand)/np.sum(weights)

def V_nm(ns,ms,omegas,Qns,Vs,weights):

    phi_n = get_basis_function(ns,Qns,omegas)
    phi_m = get_basis_function(ms,Qns,omegas)
    
    # In principle, phis are real.
    integrand = np.conjugate(phi_n)*phi_m*Vs

    return np.sum(integrand*weights)

def construct_basis(nmax, n_modes):

    n_modes_opt = n_modes - 3

    # Maximum quantum number for each non-acoustic mode
    nmax_opt = nmax[3:]

    basis = []

    # Total number of vibrational quanta
    for k in range(np.sum(nmax_opt) + 1):

        # All ways of distributing k quanta among the modes
        for comb in itertools.combinations_with_replacement(
            range(n_modes_opt), k
        ):

            ns = np.zeros(n_modes_opt, dtype=np.int64)

            for mode in comb:
                ns[mode] += 1

            # Keep only states satisfying the mode-specific nmax
            if np.all(ns <= nmax_opt):
                basis.append(ns)

    return np.array(basis)

def construct_nmm_basis(nmax, n_modes):

    n_modes_opt = n_modes-3
    basis = [np.array([0,0,0], dtype=np.int64)]
    for mode in range(n_modes_opt):
        for k in range(1,nmax[mode+3]+1):
            ns = np.zeros(n_modes_opt, dtype=np.int64)
            ns[mode] = k
            basis.append(ns)
    
    return np.array(basis)

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

    u_disps = Qns @ pols.T[3:,:] / np.sqrt(masses_sc_flat)
    energies = model.get_energies(u_disps)

    sym_u_disps = np.empty([nconf*nsym,u_disps.shape[1]], dtype=np.float64)
    sym_energies = np.empty([nconf*nsym], dtype=np.float64)
    #sym_rphi2 = np.empty([nconf*nsym], dtype=np.float64)
    sym_weights = np.empty([nconf*nsym], dtype=np.int64)

    for ri,rot in enumerate(rot_cart):
        for si,dR in enumerate(u_disps):
            for ai in range(nat):
                sym_u_disps[ri*nconf+si,mapping[ai,ri]*3:(mapping[ai,ri]+1)*3] = rot @ dR[ai*3:(ai+1)*3]
        sym_energies[ri*nconf:(ri+1)*nconf] = energies
        #sym_rphi2[ri*nconf:(ri+1)*nconf] = rphi2
        sym_weights[ri*nconf:(ri+1)*nconf] = weights
    sym_Qns = (sym_u_disps @ pols * np.sqrt(masses_sc_flat))[:,3:]

    H = np.zeros((n_states, n_states), dtype=np.float64)

    for ni in range(n_states):
        for mj in range(n_states):
            ns = basis[ni]
            ms = basis[mj]
            if no_sym:
                #H_ij = K_nm(ns,ms,ws[3:]) + V_nm_st(ns,ms,ws[3:],Qns,energies,weights,rphi2)
                H_ij = K_nm(ns,ms,ws[3:]) + V_nm(ns,ms,ws[3:],Qns,energies,weights)
            else:
                #H_ij = K_nm(ns,ms,ws[3:]) + V_nm_st(ns,ms,ws[3:],sym_Qns,sym_energies,sym_weights,sym_rphi2)
                H_ij = K_nm(ns,ms,ws[3:]) + V_nm(ns,ms,ws[3:],sym_Qns,sym_energies,sym_weights)
            
            print(f"K({ni},{mj}):",K_nm(ns,ms,ws[3:]))
            print(f"V({ni},{mj}):",V_nm(ns,ms,ws[3:],Qns,energies,weights))
            
            H[ni,mj] = H_ij
            if ni != mj:
                H[mj,ni] = H_ij
    eigenvalues, eigenvectors = eigh(H)

    return eigenvalues, eigenvectors, H

def diagonalize_hamiltonian_st(basis, ensemble, Qns, weights, model, rphi2, no_sym):

    mapping, rot_cart, map_uc, map_tr, T_list, T_list_frac = Classify.map_singlet(ensemble.current_dyn)

    nsym = rot_cart.shape[0]
    nconf = Qns.shape[0]
    nat = ensemble.current_dyn.structure.N_atoms

    n_states = basis.shape[0]

    super_dyn = ensemble.current_dyn.GenerateSupercellDyn(ensemble.supercell)
    
    masses_sc = super_dyn.structure.get_masses_array()
    masses_sc_flat = np.repeat(masses_sc, 3)


    ws, pols = ensemble.current_dyn.DiagonalizeSupercell()

    u_disps = Qns @ pols.T[3:,:] / np.sqrt(masses_sc_flat)
    energies = model.get_energies(u_disps)

    sym_u_disps = np.empty([nconf*nsym,u_disps.shape[1]], dtype=np.float64)
    sym_energies = np.empty([nconf*nsym], dtype=np.float64)
    sym_rphi2 = np.empty([nconf*nsym], dtype=np.float64)
    sym_weights = np.empty([nconf*nsym], dtype=np.int64)

    for ri,rot in enumerate(rot_cart):
        for si,dR in enumerate(u_disps):
            for ai in range(nat):
                sym_u_disps[ri*nconf+si,mapping[ai,ri]*3:(mapping[ai,ri]+1)*3] = rot @ dR[ai*3:(ai+1)*3]
        sym_energies[ri*nconf:(ri+1)*nconf] = energies
        sym_rphi2[ri*nconf:(ri+1)*nconf] = rphi2
        sym_weights[ri*nconf:(ri+1)*nconf] = weights
    sym_Qns = (sym_u_disps @ pols * np.sqrt(masses_sc_flat))[:,3:]

    H = np.zeros((n_states, n_states), dtype=np.float64)

    for ni in range(n_states):
        for mj in range(n_states):
            ns = basis[ni]
            ms = basis[mj]
            
            if no_sym:
                H_ij = K_nm(ns,ms,ws[3:]) + V_nm_st(ns,ms,ws[3:],Qns,energies,weights,rphi2)
                # H_ij = K_nm(ns,ms,ws[3:]) + V_nm(ns,ms,ws[3:],Qns,energies,weights)
            else:
                H_ij = K_nm(ns,ms,ws[3:]) + V_nm_st(ns,ms,ws[3:],sym_Qns,sym_energies,sym_weights,sym_rphi2)
                # H_ij = K_nm(ns,ms,ws[3:]) + V_nm(ns,ms,ws[3:],sym_Qns,sym_energies,sym_weights,sym_rphi2)
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

def get_new_correction(model, ensemble, T, n_max, N, step_size, alpha_mix=0.01, no_sym=False):

    super_dyn = ensemble.current_dyn.GenerateSupercellDyn(ensemble.supercell)
    nat_sc = super_dyn.structure.N_atoms
    n_modes = nat_sc*3

    w, pols = ensemble.current_dyn.DiagonalizeSupercell()

    basis = construct_basis(n_max, n_modes)
    eigenvectors0 = np.diag(np.ones(basis.shape[0])) # Hasteko. Gero hau hobetu beharko da.
    eigenvalues0 = get_eigenvalues(basis, w[3:]) # Hasteko. Gero hau hobetu beharko da.

    print("Iteration 0")

    #weighted_samples, weights, rphi2 = sample_wavefunction_unique(basis,eigenvectors0,eigenvalues0,eigenvectors0,eigenvalues0,alpha_mix,T,omegas=w[3:],N_unique_targets=N,step_size=step_size)
    weighted_samples, weights = build_general_quadrature_grid(basis,w[3:])

    #eigenvalues1, eigenvectors1, H = diagonalize_hamiltonian(basis, ensemble, weighted_samples, weights, model, rphi2, no_sym)
    breakpoint()
    eigenvalues1, eigenvectors1, H = diagonalize_hamiltonian(basis, ensemble, weighted_samples, weights, model, no_sym)
    print("E(0)=", eigenvalues1[0])
    print(eigenvectors1[:,0])

    mapping, rot_cart, map_uc, map_tr, T_list, T_list_frac = Classify.map_singlet(ensemble.current_dyn)

    nsym = rot_cart.shape[0]
    nconf = weighted_samples.shape[0]
    nat = ensemble.current_dyn.structure.N_atoms

    n_states = basis.shape[0]

    super_dyn = ensemble.current_dyn.GenerateSupercellDyn(ensemble.supercell)
    
    masses_sc = super_dyn.structure.get_masses_array()
    masses_sc_flat = np.repeat(masses_sc, 3)

    u_disps = weighted_samples @ pols.T[3:,:] / np.sqrt(masses_sc_flat)

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

    for ns in basis:
        print(f"\tKong-liu ratio for [{ns[0]},{ns[1]},{ns[2]}]", effective_sample_size_ratio(get_basis_function([ns[0],ns[1],ns[2]],sym_weighted_samples,w[3:]), np.sqrt(sym_rphi2), sym_weights))
    model.plot_quartic_phi(eigenvalues1,eigenvectors1,basis,w,pols,sym_weighted_samples,sym_weights)
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

def quartic_potential(x, a, b):
    """
    Computes V(x) = a*x^4 - b*x^2
    """
    return a * x**4 - b * x**2

def quartic_first_deriv(x, a, b):
    """
    Computes the force: F = -dV/dx = 2*b*x - 4*a*x^3
    """
    # Note: Returning the derivative (dV/dx)
    # If you want the Force (F = -dV/dx), use -1 * this result
    return 4 * a * x**3 - 2 * b * x

def quartic_second_deriv(x, a, b):
    """
    Computes the curvature: d^2V/dx^2 = 12*a*x^2 - 2*b
    """
    return 12 * a * x**2 - 2 * b

def sextic_potential(x, a, b, c):
    """
    V(x) = c*x^6 + a*x^4 - b*x^2
    """
    return c*x**6 + a*x**4 - b*x**2

def sextic_first_deriv(x, a, b, c):
    """
    Force related: dV/dx = 6*c*x^5 + 4*a*x^3 - 2*b*x
    """
    return 6*c*x**5 + 4*a*x**3 - 2*b*x

def sextic_second_deriv(x, a, b, c):
    """
    Hessian: d^2V/dx^2 = 30*c*x^4 + 12*a*x^2 - 2*b
    """
    return 30*c*x**4 + 12*a*x**2 - 2*b

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

def effective_sample_size_ratio(phi_n, phi_proposal, weights=None):
    """
    Returns the ESS ratio (ESS / N) as a fraction between 0 and 1,
    safely handling both real and complex wavefunctions.
    """
    if weights is None:
        weights = np.ones(phi_n.shape[0])
        
    # Calculate the physical probability densities (squared magnitudes)
    # np.abs() squared correctly handles complex conjugates: |a + ib|^2 = a^2 + b^2
    target_density = np.abs(phi_n)**2
    proposal_density = np.abs(phi_proposal)**2
    
    # Calculate the importance weights directly
    # (Using a small epsilon to prevent any division-by-zero errors)
    imp_weights = target_density / (proposal_density + 1e-15)
    
    # Compute the weighted means of the importance weights
    w_mean = np.sum(imp_weights * weights) / np.sum(weights)
    w2_mean = np.sum((imp_weights**2) * weights) / np.sum(weights)
    
    return w_mean**2 / w2_mean

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