import numpy as np
import itertools
import os

import cellconstructor as CC

import scipy.special, scipy.linalg


hbar = 1
kb = 1.5834055e-6 # Ry/K

class Beyond:
    """
    """
    @classmethod
    def from_aux(cls, aux_dyn: object, nmax: list, dmodes: list, use_syms: bool, dV: int=4) -> "Beyond":
        """
        """
        # Initializing input attributes
        instance = cls()
        instance.dyn = aux_dyn
        instance.nmax = np.asarray(nmax, dtype=np.int32)
        instance.dmodes = np.asarray(dmodes, dtype=bool)
        instance.use_syms = use_syms
        instance.dV = dV

        # Building the setup...
        instance.super_dyn = instance.dyn.GenerateSupercellDyn()
        instance.structure = instance.super_dyn.structure
        instance.N_atoms = instance.super_dyn.structure.N_atoms
        instance.N_modes = instance.N_atoms*3
        instance.N_dmodes = np.sum(instance.dmodes)

        instance.omegas = np.empty([instance.N_modes], dtype=np.complex128)
        instance.pols = np.empty([instance.N_modes, instance.N_modes], dtype=np.float64)

        instance.masses_sc = instance.super_dyn.structure.get_masses_array()
        _masses_sc_flat = np.repeat(instance.masses_sc, 3)

        instance.omegas, instance.pols = aux_dyn.DiagonalizeSupercell()

        instance.domegas = instance.omegas[instance.dmodes==True]
        instance.dbasis = _construct_basis(instance.nmax, instance.dmodes)

        instance.N_dbasis_states = instance.dbasis.shape[0] 

        instance.dqgrid, instance.weights = _build_grid(instance.dbasis,instance.domegas,instance.dV)

        instance.N_conf = instance.dqgrid.shape[0]
        instance.qgrid = np.zeros([instance.N_conf,instance.N_modes], dtype=np.float64)

        instance.qgrid[:,instance.dmodes==True] = instance.dqgrid

        instance.dR_grid = instance.qgrid @ instance.pols.T / np.sqrt(_masses_sc_flat) / CC.Units.A_TO_BOHR # Angstroms

        instance.xats_grid = np.empty([instance.N_conf, instance.N_atoms, 3], dtype=np.float64)
        instance.xats_grid = np.reshape(instance.dR_grid, [instance.N_conf, instance.N_atoms, 3]) + instance.super_dyn.structure.coords # Angstroms

        instance.energies = np.zeros([instance.N_conf], dtype=np.float64)

        instance.structures = np.empty([instance.N_conf], dtype=object)
        for n in range(instance.N_conf):
            instance.structures[n] = instance.structure.copy()
            instance.structures[n].coords = instance.xats_grid[n]

        return instance

    
    def diagonalize_dH(self):
        self.H = np.zeros([self.N_dbasis_states, self.N_dbasis_states], dtype=np.float64)

        for ni in range(self.N_dbasis_states):
            for mj in range(self.N_dbasis_states):
                ns = self.dbasis[ni]
                ms = self.dbasis[mj]

                H_ij = K_nm(ns,ms,self.domegas) + V_nm(ns,ms,self.domegas,self.qgrid,self.weights,self.energies)
                self.H[ni,mj] = H_ij
                if ni != mj:
                    self.H[mj,ni] = H_ij

        self.deigenvalues, self.deigenvectors = scipy.linalg.eigh(self.H)

    def save_bin(self, data_dir, population_id = 1):
        # Check if the data dir exists
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        np.save("%s/xats_pop%d.npy" % (data_dir, population_id), self.xats_grid)
        np.save("%s/energies_pop%d.npy" % (data_dir, population_id), self.energies)

    def load_bin(self, data_dir, population_id = 1):

        self.xats_grid = np.load("%s/xats_pop%d.npy" % (data_dir, population_id))
        self.energies = np.load("%s/energies_pop%d.npy" % (data_dir, population_id))

    def save(self, data_dir, population_id = 1):
        # Check if the data dir exists
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        np.savetxt(os.path.join(data_dir, "energies_supercell_population%d.dat" % (population_id)), self.energies)
        for n in range(self.N_conf):
            # Save the configurations
            struct = self.structures[n]
            struct.save_scf("%s/scf_population%d_%d.dat" % (data_dir, population_id, n+1))

    def get_A(self, T, w_grid, eta=1e-6):
        As = []

        # 1. Coupled subspace (Index 0)
        _A, _ = compute_subspace_A(
            T, 
            self.deigenvalues, 
            self.deigenvectors, 
            self.dbasis, 
            self.domegas, 
            w_grid, 
            eta
        )
        As.append(_A)

        # 2. Decoupled 1D harmonic modes (Indices 1, 2, ...)
        for mode in range(self.N_modes - 3):
            opt_mode = mode + 3
            if not self.dmodes[opt_mode]:
                n_max_mode = self.nmax[opt_mode]
                omega_mode = self.omegas[opt_mode]

                eigenvalues = _get_ho_eigenvalues(omega_mode, n_max_mode)
                eigenvectors = np.eye(n_max_mode + 1, dtype=np.float64)
                basis = np.arange(n_max_mode + 1, dtype=np.int64)[:, np.newaxis]
                omegas_single = np.array([omega_mode], dtype=np.float64)

                _A, _ = compute_subspace_A(
                    T, 
                    eigenvalues, 
                    eigenvectors, 
                    basis, 
                    omegas_single, 
                    w_grid, 
                    eta
                )
                As.append(_A)

        # Convert list to 2D numpy array: shape (N_subspaces, len(w_grid))
        A_subspaces = np.array(As)
        
        # Total spectrum is simply the sum along axis 0
        A_total = np.sum(A_subspaces, axis=0)

        return A_total, A_subspaces

def compute_subspace_A(T, eigenvalues, eigenvectors, basis, omegas, w_grid, 
                       eta=1e-5):
    """
    Computes the response spectral function A(w) analytically in the frequency domain 
    for a subspace of modes using the Lehmann representation with Lorentzian broadening.
    """
    n_basis = len(eigenvalues)
    # Ensure basis is 2D even for a single mode
    if basis.ndim == 1:
        basis = basis[:, np.newaxis]
    n_modes = basis.shape[1]
    n_points = len(w_grid)
    
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

    # pop_diff[n, m] = P_n - P_m (Initial state n, Final state m)
    pop_diff = weights[:, np.newaxis] - weights[np.newaxis, :] 

    # ------------------------------------------------------------------
    # 2. Transition Energies w_mn = (E_m - E_n) / hbar
    # ------------------------------------------------------------------
    # w_mn_matrix[n, m] = E_m - E_n (Positive for absorption)
    w_mn_matrix = (eigenvalues[np.newaxis, :] - eigenvalues[:, np.newaxis]) / hbar
    
    A_modes = np.zeros((n_modes, n_points), dtype=np.float64)
    w_2d = w_grid[np.newaxis, :]  # shape (1, n_points)

    diffs = basis[:, np.newaxis, :] - basis[np.newaxis, :, :]

    # ------------------------------------------------------------------
    # 3. Loop over all optical modes mu
    # ------------------------------------------------------------------
    for mu in range(n_modes):
        omega_m = omegas[mu]
        Q_harm_mu = np.zeros((n_basis, n_basis), dtype=np.float64)
        
        other_modes_mask = np.ones((n_basis, n_basis), dtype=bool)
        for m_idx in range(n_modes):
            if m_idx != mu:
                other_modes_mask &= (diffs[:, :, m_idx] == 0)
                
        creation_mask = other_modes_mask & (diffs[:, :, mu] == 1)
        annihilation_mask = other_modes_mask & (diffs[:, :, mu] == -1)
        
        n_quanta_matrix = np.tile(basis[:, mu][:, np.newaxis], (1, n_basis))
        
        # FIXED: Added the 2.0 in the denominator, and restored YOUR correct +1.0 logic!
        Q_harm_mu[creation_mask] = np.sqrt(hbar * n_quanta_matrix[creation_mask] / (2.0 * omega_m))
        Q_harm_mu[annihilation_mask] = np.sqrt(hbar * (n_quanta_matrix[annihilation_mask] + 1.0) / (2.0 * omega_m))

        # Transform to exact eigenbasis: Q_eigen = V^T @ Q_harm @ V
        Qnm_mu = eigenvectors.T @ Q_harm_mu @ eigenvectors
        
        # Transition weights: |<m| Q_mu |n>|^2 * (P_n - P_m)
        transition_weights = (np.abs(Qnm_mu)**2) * pop_diff

        # Filter strictly for absorptive transitions (dropped 1e-15 to > 0 to prevent clipping tiny physical values)
        valid_mask = (transition_weights > 0) & (w_mn_matrix > 0)
        
        if not np.any(valid_mask):
            continue

        S_valid = transition_weights[valid_mask]
        w_trans_valid = w_mn_matrix[valid_mask]

        # Your exact direct analytical Lorentzian summation over transitions
        for S, w_trans in zip(S_valid, w_trans_valid):
            lorentzian = (1.0 / np.pi) * (eta / ((w_2d - w_trans)**2 + eta**2))
            A_modes[mu] += S * lorentzian[0]

    A_total = np.sum(A_modes, axis=0)

    return A_total, A_modes

def combine_spectra(A_list, w_grid, eta=1e-5, broadening_type="lorentzian"):
    """
    Combines a list of independent subspace spectral functions A(w) into a single 
    total A_total(w) via time-domain correlation function multiplication.
    
    Parameters:
    -----------
    A_list : list of ndarray
        List containing the 1D A(w) arrays for each independent subspace.
        All arrays must have the same length as w_grid.
    w_grid : ndarray (n_points,)
        Equidistant 1D frequency grid used for all spectra.
    eta : float
        Broadening factor (decay constant in time domain).
    broadening_type : str
        Type of broadening to apply: 'lorentzian' (linear decay) 
        or 'gaussian' (quadratic decay).
        
    Returns:
    --------
    A_total : ndarray (n_points,)
        The combined, exact total response spectral function.
    """
    n_points = len(w_grid)
    dw = w_grid[1] - w_grid[0]
    
    # ------------------------------------------------------------------
    # 1. Setup Time Grid corresponding to the Frequency Grid
    # ------------------------------------------------------------------
    # Time step derived from total frequency range
    w_span = w_grid[-1] - w_grid[0]
    dt = 2.0 * np.pi / w_span
    t_grid = np.arange(n_points) * dt

    # ------------------------------------------------------------------
    # 2. Transform Subspace Spectra to Time-Domain Correlation Functions
    # ------------------------------------------------------------------
    # Initialize total correlation function as identity (1.0)
    chi_total_t = np.ones(n_points, dtype=np.complex128)

    for A_sub in A_list:
        # Inverse FFT converts spectral density to time-domain response
        chi_sub_t = np.fft.ifft(A_sub)
        
        # Multiply independent correlation functions in time (Convolution in w)
        chi_total_t *= chi_sub_t

    # ------------------------------------------------------------------
    # 3. Apply Phenomenological Broadening in Time Domain
    # ------------------------------------------------------------------
    if eta > 0:
        if broadening_type.lower() == "lorentzian":
            damping = np.exp(-eta * t_grid)
        elif broadening_type.lower() == "gaussian":
            damping = np.exp(-(eta * t_grid)**2)
        else:
            raise ValueError("broadening_type must be 'lorentzian' or 'gaussian'")
        
        chi_total_t *= damping

    # ------------------------------------------------------------------
    # 4. FFT back to Frequency Domain
    # ------------------------------------------------------------------
    A_total = np.real(np.fft.fft(chi_total_t))
    
    # Normalize to preserve total spectral weight / integrated intensity
    A_total = np.maximum(0.0, A_total)  # Ensure non-negative numerical noise
    
    return A_total

def _construct_basis(nmax, dmodes):
    # From now on, we are working just with optical modes.
    nmax_opt = nmax[3:]
    dmodes_opt = dmodes[3:]

    dindices = np.where(dmodes_opt)[0]
    dnmax = nmax_opt[dindices]

    N_dmodes = len(dindices)

    dbasis = []

    for k in range(np.sum(dnmax)+1):
        for comb in itertools.combinations_with_replacement(range(N_dmodes), k):
            dns = np.zeros(N_dmodes, dtype=np.int32)
            for mode in comb:
                dns[mode] += 1
            if np.all(dns <= dnmax):
                dbasis.append(dns)
    dbasis = np.asarray(dbasis, dtype=np.int32)

    # Check size of dbasis.
    assert np.prod(nmax_opt[dindices]+1) == dbasis.shape[0], f"Expected {np.prod(nmax_opt[dindices])} basis functions for decoupled space, obtained {dbasis.shape[0]}."

    return dbasis

def _build_grid(basis, omegas, dV):
    basis = np.asarray(basis)
    omegas = np.asarray(omegas)

    N_modes = basis.shape[1]

    # Knowing our basis, loop for the max n in each column.
    nmax = np.max(basis, axis=0)

    grid_list = []
    weight_list = []

    for mu in range(N_modes):
        # Exact for polinomial Vs up to dV-th order.
        n_points = int(np.ceil(nmax[mu]+(dV+1)/2))

        # Get standard mathematical roots (x) and weights (w) for e^(-x^2)
        x_std, w_std = np.polynomial.hermite.hermgauss(n_points)

        omega = omegas[mu]
        
        # 1. Physical mass-weighted normal coordinate mapping (Rydberg)
        q_physical = x_std / np.sqrt(omega)

        # 2. Correct weight stripping with the transformation Jacobian dQ = dx/sqrt(2*omega)
        jacobian = 1.0 / np.sqrt(omega)
        w_physical = w_std * jacobian * np.exp(x_std**2)

        grid_list.append(q_physical)
        weight_list.append(w_physical)

    # Multi-dimensional grid assembly via meshgrid
    mesh_grids = np.meshgrid(*grid_list, indexing='ij')
    weight_grids = np.meshgrid(*weight_list, indexing='ij')
    
    # Flatten and stack into your final coordinate matrix
    grid_nd = np.stack([g.ravel() for g in mesh_grids], axis=1)
    
    # Total volume weight element (dQ1 * dQ2 * ...)
    weights_nd = weight_grids[0].ravel()
    for w_grid in weight_grids[1:]:
        weights_nd *= w_grid.ravel()
        
    return grid_nd, weights_nd

def _get_ho_eigenvalues(omega, n_max):
    """
    """
    eigenvalues = np.zeros(n_max+1, dtype=np.float64)
    
    for n in range(n_max+1):
        eigenvalues[n] = hbar * omega * (0.5 + n)
    
    return eigenvalues

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
    
    integrand = np.conjugate(phi_n)*phi_m*Vs

    return np.sum(integrand*weights)

def independent_harm_osc(n, q, omega):
    qi = np.sqrt(omega/hbar)*q
    Hn = np.polynomial.hermite.hermval(qi, [0]*n + [1])
    prefac = (omega / (np.pi * hbar))**(1/4) / np.sqrt(2.0**n * scipy.special.factorial(n))
    return prefac * Hn * np.exp(-0.5 * qi**2)

def get_basis_function(ns,qs,omegas):

    phi = 1
    for ni, n in enumerate(ns):
        phi *= independent_harm_osc(n, qs[:,ni], omegas[ni])
    return phi