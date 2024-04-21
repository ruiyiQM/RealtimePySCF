import numpy as np
import scipy
import scipy.linalg
from pyscf import gto, dft, scf, ao2mo
from func import *

class fields:
    """
    A class which manages field perturbations. Mirrors TCL_FieldMatrices.h
    """
    def __init__(self, the_scf_, params_):
        self.dip_ints = None  # AO dipole integrals.
        self.dip_ints_bo = None
        self.nuc_dip = None
        self.dip_mo = None  # Nuclear dipole (AO)
        self.Generate(the_scf_)
        self.fieldAmplitude = params_["FieldAmplitude"]
        self.tOn = params_["tOn"]
        self.Tau = params_["Tau"]
        self.FieldFreq = params_["FieldFreq"]
        self.pol = np.array([params_["ExDir"], params_["EyDir"], params_["EzDir"]])
        self.pol0 = None
        self.pol0AA = None

    def Generate(self, the_scf):
        """
        Performs the required PYSCF calls to generate the AO basis dipole matrices.
        """
        self.dip_ints = the_scf.mol.intor('cint1e_r_sph', comp=3)  # component, ao, ao.
        charges = the_scf.mol.atom_charges()
        coords = the_scf.mol.atom_coords()
        self.nuc_dip = np.einsum('i,ix->x', charges, coords)

    def Update(self, c_mat):
        """
        Args:
            c_mat: Transformation matrix (AOx??)
        Updates dip_int to (?? x ??)
        """
        return

    def ImpulseAmp(self, time):
        amp = self.fieldAmplitude * np.sin(self.FieldFreq * time) * (1.0 / np.sqrt(2.0 * np.pi * self.Tau * self.Tau)) * np.exp(-1.0 * np.power(time - self.tOn, 2.0) / (2.0 * self.Tau * self.Tau))
        IsOn = False
        if (np.abs(amp) > pow(10.0, -9.0)):
            IsOn = True
        return amp, IsOn

    def InitializeExpectation(self, rho0_, C_, nA=None):
        self.pol0 = self.Expectation(rho0_, C_)
        if nA is not None:
            self.dip_ints_bo = self.dip_ints.copy()
            for i in range(3):
                self.dip_ints_bo[i] = TransMat(self.dip_ints[i], C_)
            self.pol0AA = self.Expectation(rho0_, C_, True, nA)

    def ApplyField(self, a_mat, time):
        """
        Args:
            a_mat: an AO matrix to which the field is added.
            time: current time.
        Returns:
            a_mat + dipole field at this time.
            IsOn
        """
        amp, IsOn = self.ImpulseAmp(time)
        mpol = self.pol * amp
        if IsOn:
            print("Field on")
            return a_mat + 2.0 * np.einsum("kij,k->ij", self.dip_ints, mpol), True
        else:
            return a_mat, False

    def ApplyField(self, a_mat, c_mat, time, par=None):
        """
        Args:
            a_mat: an MO matrix to which the field is added.
            c_mat: a AO=>MO coefficient matrix.
            time: current time.
        Returns:
            a_mat + dipole field at this time.
            IsOn
        """
        amp, IsOn = self.ImpulseAmp(time)
        mpol = self.pol * amp
        if IsOn:
            return a_mat + 2.0 * TransMat(np.einsum("kij,k->ij", self.dip_ints, mpol), c_mat), True
        else:
            return a_mat, False

    def Expectation(self, rho_, C_, AA=False, nA=None, U=None):
        """
        Args:
            rho_: current MO density.
            C_: current AO=> MO Transformation. (ao X mo)
        Returns:
            [<Mux>, <Muy>, <Muz>]
        """
        rhoAO = TransMat(rho_, C_, -1)
        if AA:
            e_dip = np.einsum('xij,ji->x', self.dip_ints[:, :nA, :nA], rhoAO[:nA, :nA])
            if self.pol0AA is not None:
                return e_dip - self.pol0AA
            else:
                return e_dip
        else:
            mol_dip = np.einsum('xij,ji->x', self.dip_ints, rhoAO)
            if self.pol0 is not None:
                return mol_dip - self.pol0
            else:
                return mol_dip
