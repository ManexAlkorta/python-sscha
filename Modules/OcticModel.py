import numpy as np

import cellconstructor as CC

class OcticModel:
    """
    """
    @classmethod
    def from_aux(cls, aux_dyn):
        """
        """
        # Initializing attributes
        instance = cls()
        instance.dyn0 = aux_dyn
        instance.structure = aux_dyn.structure
        instance.N = 0

        # Generate supercell
        instance.super_dyn0 = instance.dyn0.GenerateSupercellDyn()
        instance.super_structure = instance.super_dyn0.structure
        instance.N_atoms = instance.super_structure.N_atoms
        instance.N_modes = instance.N_atoms*3

        instance.masses_sc = instance.super_dyn0.structure.get_masses_array()
        instance.masses_sc_flat = np.repeat(instance.masses_sc, 3)

        # Diagonalize dynamical matrix
        instance.omegas = np.empty([instance.N_modes], dtype=np.complex128)
        instance.omegas0, instance.pols = instance.super_dyn0.DiagonalizeSupercell()
        instance.omegas = instance.omegas0.astype(np.complex128)
        instance.eigenvalues = np.real(np.square(instance.omegas))

        instance.dyns =  instance.pols @ np.diag(instance.eigenvalues) @ instance.pols.T
        instance.phis = np.empty(instance.dyns.shape)
        for i in range(instance.N_atoms):
            for j in range(instance.N_atoms):
                instance.phis[i*3:(i+1)*3,j*3:(j+1)*3] = instance.dyns[i*3:(i+1)*3,j*3:(j+1)*3] * np.sqrt(instance.masses_sc[i] * instance.masses_sc[j])
    
        # Define the multidimensional octic potential
        instance.params = np.zeros([instance.N_modes,4], dtype=np.float64)
        instance.params[:,0] = np.real(np.square(instance.omegas)) * 0.5

        return instance

    def run_bin(self, data_dir, population_id):
        self.load_bin(data_dir, population_id)
        self.get_energies()
        self.get_forces()
        self.save_bin(data_dir, population_id)

    def get_energies(self):
        Qns = (self.dR * np.sqrt(self.masses_sc_flat)) @ self.pols  # u_disps are in Angstroms
        self.energies = np.zeros([self.N], dtype=np.float64)
        for mode in range(self.N_modes):
            self.energies += octic_potential(Qns[:,mode],self.params[mode,0],self.params[mode,1],self.params[mode,2],self.params[mode,3])

    def get_forces(self):
        Qns = (self.dR * np.sqrt(self.masses_sc_flat)) @ self.pols  # u_disps are in Angstroms
        self.forces = np.zeros([self.N, self.N_atoms, 3], dtype=np.float64)
        for mode in range(self.N_modes):
            self.forces += np.reshape((-1) * np.sqrt(self.masses_sc_flat) * (np.outer(octic_first_deriv(Qns[:,mode],self.params[mode,0],self.params[mode,1],self.params[mode,2],self.params[mode,3]), self.pols[:,mode])), [self.N,self.N_atoms,3])

    def save_bin(self, data_dir, population_id=1):
        self.xats = np.save(f"{data_dir}/xats_pop{population_id}.npy", self.xats / CC.Units.A_TO_BOHR) # SSCHA wants A
        self.energies = np.save(f"{data_dir}/energies_pop{population_id}.npy", self.energies) # Ry
        self.forces = np.save(f"{data_dir}/forces_pop{population_id}.npy", self.forces * CC.Units.A_TO_BOHR) # SSCHA wants Ry/A

    def load_bin(self, data_dir, population_id=1):
        self.xats = np.load(f"{data_dir}/xats_pop{population_id}.npy") * CC.Units.A_TO_BOHR
        self.N = self.xats.shape[0]
        self.energies = np.zeros([self.N], dtype=np.float64)
        self.dR = np.empty([self.N, self.N_modes], dtype=np.float64)
        for n in range(self.N):
            self.dR[n] = (self.xats[n,:,:] - self.super_structure.coords * CC.Units.A_TO_BOHR).reshape(3*self.N_atoms)

    def tune_modes(self, modes, a, b, c, d):
        for mode in modes:
            self.params[mode,0] = a
            self.params[mode,1] = b
            self.params[mode,2] = c
            self.params[mode,3] = d

            self.omegas[mode] = np.sqrt(octic_second_deriv(0,a,b,c,d)+0j)

        self.eigenvalues = np.real(np.square(self.omegas))

        self.dyns =  self.pols.T @ np.diag(self.eigenvalues) @ self.pols
        self.phis = np.empty(self.dyns.shape)
        for i in range(self.N_atoms):
            for j in range(self.N_atoms):
                self.phis[i*3:(i+1)*3,j*3:(j+1)*3] = self.dyns[i*3:(i+1)*3,j*3:(j+1)*3] * np.sqrt(self.masses_sc[i] * self.masses_sc[j])

def octic_potential(q, a, b, c, d):
    """
    V(q) = a*q^2 + b*q^4 + c*q^6 + d*q^8
    """
    return a*q**2 + b*q**4 + c*q**6 + d*q**8

def octic_first_deriv(q, a, b, c, d):
    """
    Force related: dV/dq = 2*a*q + 4*b*q^3 + 6*c*q^5 + 8*d*q^7
    """
    return 2*a*q + 4*b*q**3 + 6*c*q**5 + 8*d*q**7

def octic_second_deriv(q, a, b, c, d):
    """
    Hessian (Curvature): d^2V/dq^2 = 2*a + 12*b*q^2 + 30*c*q^4 + 56*d*q^6
    """
    return 2*a + 12*b*q**2 + 30*c*q**4 + 56*d*q**6