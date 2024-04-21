import tensorflow as tf
import numpy as np
import scipy, os, time
import scipy.linalg
from func import *
from pyscf import gto, dft, scf, ao2mo
from tdfields import *
from cmath import *
from pyscf import lib
import ctypes

libdft = lib.load_library('libdft')
#sess = tf.Session()

FsPerAu = 0.0241888

class tdscf:
    """
    A general TDSCF object.
    Other types of propagations may inherit from this.
    """
    def toTF(self, arg):
        arg = tf.convert_to_tensor(arg, dtype=tf.float64)
        return arg

    def __init__(self, the_scf_, prm=None, output='log.dat', prop_=True):
        self.Enuc = the_scf_.mol.energy_nuc()
        self.eri3c = None
        self.eri2c = None
        self.n_aux = None
        self.muxo = None
        self.muyo = None
        self.muzo = None
        self.mu0 = None
        self.hyb = the_scf_._numint.hybrid_coeff(the_scf_.xc, spin=(the_scf_.mol.spin>0)+1)
        self.adiis = None
        self.Exc = None
        self.t = 0.0
        self.n_ao = None
        self.n_mo = None
        self.n_occ = None
        self.n_virt = None
        self.n_e = None
        self.rho = None
        self.rhoM12 = None
        self.F = None
        self.K = None
        self.J = None
        self.eigs = None
        self.S = None
        self.C = None
        self.X = None
        self.V = None
        self.H = None
        self.B = None
        self.log = []
        self.the_scf = the_scf_
        self.mol = the_scf_.mol
        self.auxmol_set()
        self.params = {}
        self.auxmol_set()
        self.initialcondition(prm)
        self.field = fields(the_scf_, self.params)
        self.field.InitializeExpectation(self.rho, self.C)
        start = time.time()
        if prop_:
            self.prop(output)
        end = time.time()
        print("Propagation time:", end - start)

    def auxmol_set(self, auxbas="weigend"):
        print("GENERATING INTEGRALS")
        auxmol = gto.Mole()
        auxmol.atom = self.mol.atom
        auxmol.basis = auxbas
        auxmol.build()
        mol = self.mol
        nao = self.n_ao = mol.nao_nr()
        naux = self.n_aux = auxmol.nao_nr()
        atm, bas, env = gto.conc_env(mol._atm, mol._bas, mol._env, auxmol._atm, auxmol._bas, auxmol._env)
        eri3c = np.empty((nao, nao, naux))
        pi = 0
        for i in range(mol.nbas):
            pj = 0
            for j in range(mol.nbas):
                pk = 0
                for k in range(mol.nbas, mol.nbas + auxmol.nbas):
                    shls = (i, j, k)
                    buf = gto.getints_by_shell('cint3c2e_sph', shls, atm, bas, env)
                    di, dj, dk = buf.shape
                    eri3c[pi:pi+di, pj:pj+dj, pk:pk+dk] = buf
                    pk += dk
                pj += dj
            pi += di
        print("ERI3C INTEGRALS GENERATED")
        eri2c = np.empty((naux, naux))
        pk = 0
        for k in range(mol.nbas, mol.nbas + auxmol.nbas):
            pl = 0
            for l in range(mol.nbas, mol.nbas + auxmol.nbas):
                shls = (k, l)
                buf = gto.getints_by_shell('cint2c2e_sph', shls, atm, bas, env)
                dk, dl = buf.shape
                eri2c[pk:pk+dk, pl:pl+dl] = buf
                pl += dl
            pk += dk
        print("ERI2C INTEGRALS GENERATED")
        self.eri3c = eri3c
        self.eri2c = eri2c
        RSinv = MatrixPower(eri2c, -0.5)
        #with sess.as_default():
        self.B = np.einsum('ijp,pq->ijq', self.toTF(eri3c), self.toTF(RSinv))

    def FockBuild(self, P, it=-1):
        if self.params["Model"] in ["TDHF", "TDCIS"]:
            Pt = 2.0 * TransMat(P, self.X, -1)
            J, K = self.get_jk(Pt)
            Veff = 0.5 * (J + J.T.conj()) - 0.5 * (0.5 * (K + K.T.conj()))
            if self.adiis and it > 0:
                return TransMat(self.adiis.update(self.S, Pt, self.H + Veff), self.X)
            else:
                return TransMat(self.H + 0.5 * (J + J.T.conj()) - 0.5 * (0.5 * (K + K.T.conj())), self.X)
        elif self.params["Model"] == "TDDFT":
            Pt = 2 * TransMat(P, self.X, -1)
            J = self.get_j(Pt)
            Vxc = self.get_vxc(Pt)
            Veff = J + Vxc
            if self.adiis and it > 0:
                return TransMat(self.adiis.update(self.S, Pt, self.H + Veff), self.X)
            else:
                return TransMat(self.H + Veff, self.X)

    def get_vxc(self, P):
        Vxc = self.numint_vxc(self.the_scf._numint, P)
        if self.hyb > 0.01:
            K = self.get_k(P)
            Vxc += -0.5 * self.hyb * K
        return Vxc

    def numint_vxc(self, ni, P, max_mem=2000):
        xctype = self.the_scf._numint._xc_type(self.the_scf.xc)
        make_rho, nset, nao = self._gen_rho_evaluator(self.mol, P, 1)
        ngrids = len(self.the_scf.grids.weights)
        non0tab = self.the_scf._numint.non0tab
        vmat = np.zeros((nset, nao, nao)).astype(complex)
        excsum = np.zeros(nset)
        if xctype == 'LDA':
            ao_deriv = 0
            for ao, mask, weight, coords in ni.block_loop(self.mol, self.the_scf.grids, nao, ao_deriv, max_mem, non0tab):
                rho = make_rho(0, ao, mask, 'LDA')
                exc, vxc = ni.eval_xc(self.the_scf.xc, rho.real, 0, 0, 1, None)[:2]
                vrho = vxc[0]
                den = rho * weight
                excsum[0] += (den * exc).sum()
                aow = np.einsum('pi,p->pi', ao, 0.5 * weight * vrho)
                vmat += np.einsum('ij,jk->ik', ao.T, aow)
        elif xctype == 'GGA':
            ao_deriv = 1
            for ao, mask, weight, coords in ni.block_loop(self.mol, self.the_scf.grids, nao, ao_deriv, max_mem, non0tab):
                ngrid = weight.size
                rho = make_rho(0, ao, mask, 'GGA')
                exc, vxc = ni.eval_xc(self.the_scf.xc, rho.real, 0, 0, 1, None)[:2]
                vrho, vsigma = vxc[:2]
                wv = np.empty((4, ngrid))
                wv[0] = weight * vrho * 0.5
                wv[1:] = rho[1:] * (weight * vsigma * 2)
                aow = np.einsum('npi,np->pi', ao, wv)
                vmat += np.einsum('ij,jk->ik', ao[0].T, aow)
                den = rho[0] * weight
                excsum[0] += (den * exc).sum()
        else:
            assert all(x not in self.the_scf.xc.upper() for x in ('CC06', 'CS', 'BR89', 'MK00'))
            ao_deriv = 2
            for ao, mask, weight, coords in ni.block_loop(self.mol, self.the_scf.grids, nao, ao_deriv, max_mem, non0tab):
                ngrid = weight.size
                rho = make_rho(0, ao, mask, 'MGGA')
                exc, vxc = ni.eval_xc(self.the_scf.xc, rho.real, 0, 0, 1, None)[:2]
                vrho, vsigma, vlapl, vtau = vxc[:4]
                den = rho[0] * weight
                excsum[0] += (den * exc).sum()
                wv = np.empty((4, ngrid))
                wv[0] = weight * vrho * 0.5
                wv[1:] = rho[1:4] * (weight * vsigma * 2)
                aow = np.einsum('npi,np->pi', ao[:4], wv)
                vmat += _dot_ao_ao(mol, ao[0], aow, nao, ngrid, mask)
                wv = (0.5 * 0.5 * weight * vtau).reshape(-1, 1)
                vmat += func._dot_ao_ao(mol, ao[1], wv * ao[1], nao, ngrid, mask)
                vmat += func._dot_ao_ao(mol, ao[2], wv * ao[2], nao, ngrid, mask)
                vmat += func._dot_ao_ao(mol, ao[3], wv * ao[3], nao, ngrid, mask)
        self.Exc = excsum[0]
        Vxc = vmat.reshape(nao, nao)
        Vxc = Vxc + Vxc.T.conj()
        return Vxc

    def _gen_rho_evaluator(self, mol, dms, hermi=1):
        natocc = []
        natorb = []
        e, c = scipy.linalg.eigh(dms)
        natocc.append(e)
        natorb.append(c)
        nao = dms.shape[0]
        ndms = len(natocc)
        def make_rho(idm, ao, non0tab, xctype):
            return eval_rhoc(mol, ao, natorb[idm], natocc[idm], non0tab, xctype)
        return make_rho, ndms, nao

    def get_jk(self, P):
        return self.get_j(P), self.get_k(P)

    def get_j(self, P):
        naux = self.n_aux
        nao = self.n_ao
        rho = np.einsum('ijp,ij->p', self.eri3c, P)
        rho = np.linalg.solve(self.eri2c, rho)
        jmat = np.einsum('p,ijp->ij', rho, self.eri3c)
        return jmat

    def get_k(self, P):
        naux = self.n_aux
        nao = self.n_ao
        kpj = np.einsum('ijp,jk->ikp', self.eri3c, P)
        pik = np.linalg.solve(self.eri2c, kpj.reshape(-1, naux).T.conj())
        rkmat = np.einsum('pik,kjp->ij', pik.reshape(naux, nao, nao), self.eri3c)
        return rkmat

    def initialcondition(self, prm):
        print('''
        ===================================
        |  Realtime TDSCF module          |
        ===================================
        | J. Parkhill, T. Nguyen          |
        | J. Koh, J. Herr,  K. Yao        |
        ===================================
        | Refs: 10.1021/acs.jctc.5b00262  |
        |       10.1063/1.4916822         |
        ===================================
        ''')
        n_ao = self.n_ao = self.the_scf.make_rdm1().shape[0]
        n_mo = self.n_mo = n_ao  # should be fixed.
        n_occ = self.n_occ = int(sum(self.the_scf.mo_occ) / 2)
        self.n_virt = self.n_mo - self.n_occ
        print("n_ao:", n_ao, "n_mo:", n_mo, "n_occ:", n_occ)
        self.ReadParams(prm)
        self.InitializeLiouvillian()
        return

    def ReadParams(self, prm):
        self.params["Model"] = "TDDFT"  # "TDHF"; the difference of Fock matrix and energy
        self.params["Method"] = "MMUT"  # "MMUT"
        self.params["BBGKY"] = 0
        self.params["TDTDA"] = 0
        self.params["TDCIS"] = 1

        self.params["dt"] = 0.02
        self.params["MaxIter"] = 15000

        self.params["ExDir"] = 1.0
        self.params["EyDir"] = 1.0
        self.params["EzDir"] = 1.0
        self.params["FieldAmplitude"] = 0.01
        self.params["FieldFreq"] = 0.9202
        self.params["Tau"] = 0.07
        self.params["tOn"] = 7.0 * self.params["Tau"]
        self.params["ApplyImpulse"] = 1
        self.params["ApplyCw"] = 0
        self.params["SavePopulation"] = 1

        self.params["StatusEvery"] = 5000
        self.params["Print"] = 0
        if prm is not None:
            for line in prm.splitlines():
                s = line.split()
                if len(s) > 1:
                    if s[0] in ["MaxIter", "ApplyImpulse", "ApplyCw", "StatusEvery"]:
                        self.params[s[0]] = int(s[1])
                    elif s[0] in ["Model", "Method"]:
                        self.params[s[0]] = s[1]
                    else:
                        self.params[s[0]] = float(s[1])

        print("=============================")
        print("         Parameters")
        print("=============================")
        print("Model:", self.params["Model"])
        print("Method:", self.params["Method"])
        print("dt:", self.params["dt"])
        print("MaxIter:", self.params["MaxIter"])
        print("ExDir:", self.params["ExDir"])
        print("EyDir:", self.params["EyDir"])
        print("EzDir:", self.params["EzDir"])
        print("FieldAmplitude:", self.params["FieldAmplitude"])
        print("FieldFreq:", self.params["FieldFreq"])
        print("Tau:", self.params["Tau"])
        print("tOn:", self.params["tOn"])
        print("ApplyImpulse:", self.params["ApplyImpulse"])
        print("ApplyCw:", self.params["ApplyCw"])
        print("StatusEvery:", self.params["StatusEvery"])
        print("SavePopulation:", self.params["SavePopulation"])
        print("=============================\n\n")
        return

    def InitializeLiouvillian(self):
        S = self.S = self.the_scf.get_ovlp()
        self.X = MatrixPower(S, -1. / 2.)
        self.V = np.eye(self.n_ao)
        self.C = self.X.copy()  # Initial set of orthogonal coordinates.
        self.InitFockBuild()  # updates self.C
        self.rho = 0.5 * np.diag(self.the_scf.mo_occ).astype(complex)
        self.rhoM12 = self.rho.copy()
        return

    def InitFockBuild(self):
        start = time.time()
        n_occ = self.n_occ
        Ne = self.n_e = 2.0 * n_occ
        err = 100
        it = 0
        self.H = self.the_scf.get_hcore()
        S = self.S.copy()
        SX = np.dot(S, self.X)
        Plao = 0.5 * TransMat(self.the_scf.get_init_guess(self.mol, self.the_scf.init_guess), SX).astype(complex)
        adiis = self.the_scf.DIIS(self.the_scf, self.the_scf.diis_file)
        adiis.space = self.the_scf.diis_space
        adiis.rollback = self.the_scf.diis_space_rollback
        self.adiis = adiis
        self.F = self.FockBuild(Plao)
        Plao_old = Plao
        E = self.energy(Plao) + self.Enuc
        while (err > 10**-10):
            self.eigs, self.V = np.linalg.eig(self.F)
            idx = self.eigs.argsort()
            self.eigs.sort()
            self.V = self.V[:, idx].copy()
            Pmo = 0.5 * np.diag(self.the_scf.mo_occ).astype(complex)
            Plao = TransMat(Pmo, self.V, -1)
            print("Ne", np.trace(Plao), np.trace(Pmo))
            Eold = E
            E = self.energy(Plao)
            self.F = self.FockBuild(Plao, it)
            err = abs(E - Eold)
            if it % 1 == 0:
                print("Iteration:", it, "; Energy:", E, "; Error =", err)
            it += 1
        Pmo = 0.5 * np.diag(self.the_scf.mo_occ).astype(complex)
        Plao = TransMat(Pmo, self.V, -1)
        self.C = np.dot(self.X, self.V)
        self.rho = TransMat(Plao, self.V)
        self.rhoM12 = TransMat(Plao, self.V)
        print("Ne", np.trace(Plao), np.trace(self.rho))
        print("Energy:", E)
        print("Initial Fock Matrix(MO)\n", TransMat(self.F, self.V))
        end = time.time()
        print("Initial Fock Built time:", end - start)
        return Plao

    def Split_RK4_Step_MMUT(self, w, v, oldrho, time, dt, IsOn):
        Ud = np.exp(w * (-0.5j) * dt)
        U = TransMat(np.diag(Ud), v, -1)
        RhoHalfStepped = TransMat(oldrho, U, -1)
        newrho = TransMat(RhoHalfStepped, U, -1)
        return newrho

    def Posify(self, rho):
        n = self.n_ao
        rhob = np.eye(n).astype(complex)
        nocc = self.n_occ
        rhob[nocc:, nocc:] *= 0.0
        rho -= rhob
        rho = 0.5 * (rho + rho.T.conj())
        if abs(np.trace(rho)) < pow(10.0, -13.0):
            rho += rhob
        else:
            eig, eigv = np.linalg.eigh(rho)
            ne = 0.0
            nh = 0.0
            IsHole = np.zeros(n)
            IsE = np.zeros(n)
            for i in range(n):
                if abs(eig[i] > 1.0):
                    eig[i] = ((0 < eig[i]) - (eig[i] < 0)) * 1.0
                if eig[i] < 0.0:
                    IsHole[i] = 1.0
                    nh += eig[i]
                else:
                    IsE[i] = 1.0
                    ne += eig[i]
            if abs(sum(eig * IsHole)) < abs(sum(eig * IsE)):
                if abs(sum(eig * IsE)) != 0.0:
                    IsE *= abs(sum(eig * IsHole)) / abs(sum(eig * IsE))
                    eig = (eig * (IsE + IsHole))
            else:
                if abs(sum(eig * IsHole)) != 0.0:
                    IsHole *= abs(sum(eig * IsE)) / abs(sum(eig * IsHole))
                    eig = (eig * (IsE + IsHole))
            rho = TransMat(np.diag(eig), eigv, -1) + rhob
        if self.params["Stabilize"] > 1.0:
            print("Stabilization needed")
            eig, eigv = np.linalg.eigh(rho)
            for i in range(n):
                if eig[i] < 0.0:
                    eig[i] *= 0.0
                elif eig[i] > 1.0:
                    eig[i] *= 1.0
            missing = nocc - sum(eig)
            if abs(missing) < pow(10.0, -10.0):
                return
            if n - nocc - 1 < 0:
                rho = TransMat(np.diag(eig), eigv, -1)
                return

            if missing < 0.0 and eig[n - nocc - 1] + missing > 0.0:
                eig[n - nocc - 1] += missing
            elif missing < 0.0:
                eig[n - nocc] += missing
            elif missing > 0.0 and eig[n - nocc] + missing < 1.0:
                eig[n - nocc] += missing
            else:
                eig[n - nocc - 1] += missing
            missing = nocc - sum(eig)
            rho = TransMat(np.diag(eig), eigv, -1)
            for i in range(n):
                if rho[i, i].real < 0.0:
                    rho[i, i] *= 0.0
        self.rho = rho.copy()
        return

    def rhodot(self, rho, time, F):
        print("F", F)
        print("Rho", rho)
        Ft, IsOn = self.field.ApplyField(F, self.C, time, self)
        tmp = -1.j * (np.dot(Ft, rho) - np.dot(rho, Ft))
        print("Ft", Ft)
        print("tmp[0,1]", tmp[0, 1], tmp[1, 0], tmp[0, 1] / rho[0, 1])
        return -1.j * (np.dot(Ft, rho) - np.dot(rho, Ft))

    def TDDFTstep(self, time):
        if self.params["Method"] == "MMUT":
            self.F = self.FockBuild(TransMat(self.rho, self.V, -1))
            self.F = np.conj(self.F)
            Fmo_prev = TransMat(self.F, self.V)
            self.eigs, rot = np.linalg.eig(Fmo_prev)
            self.rho = TransMat(self.rho, rot)
            self.rhoM12 = TransMat(self.rhoM12, rot)
            self.V = np.dot(self.V, rot)
            self.C = np.dot(self.X, self.V)
            Fmo = np.diag(self.eigs).astype(complex)
            FmoPlusField, IsOn = self.field.ApplyField(Fmo, self.C, time, self)
            w, v = scipy.linalg.eig(FmoPlusField)
            NewRhoM12 = self.Split_RK4_Step_MMUT(w, v, self.rhoM12, time, self.params["dt"], IsOn)
            NewRho = self.Split_RK4_Step_MMUT(w, v, NewRhoM12, time, self.params["dt"] / 2.0, IsOn)
            self.rho = 0.5 * (NewRho + (NewRho.T.conj()))
            self.rhoM12 = 0.5 * (NewRhoM12 + (NewRhoM12.T.conj()))
        elif self.params["Method"] == "RK4":
            raise Exception("Broken.")
            dt = self.params["dt"]
            rho = self.rho.copy()
            C = self.C.copy()
            self.FockBuild(TransMat(rho, C, -1))
            Fmo = TransMat(self.F, C)
            if True:
                eigs, rot = np.linalg.eig(Fmo)
                print("Eigs", eigs)

            k1 = self.rhodot(rho, time, Fmo)
            v2 = (0.5 * dt) * k1
            v2 += rho
            self.FockBuild(TransMat(v2, C, -1))
            Fmo = TransMat(self.F, C)
            k2 = self.rhodot(v2, time + 0.5 * dt, Fmo)
            v3 = (0.5 * dt) * k2
            v3 += rho

            self.FockBuild(TransMat(v3, C, -1))
            Fmo = TransMat(self.F, C)
            k3 = self.rhodot(v3, time + 0.5 * dt, Fmo)
            v4 = (1.0 * dt) * k3
            v4 += rho

            self.FockBuild(TransMat(v4, C, -1))
            Fmo = TransMat(self.F, C)
            k4 = self.rhodot(v4, time + dt, Fmo)

            drho = dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            self.rho += drho
            self.rho = 0.5 * (self.rho + self.rho.T.conj())
        else:
            raise Exception("Unknown Method...")
        return

    def step(self, time):
        if self.params["Model"] in ["TDDFT", "TDHF"]:
            return self.TDDFTstep(time)
        elif self.params["Model"] == "EE2":
            return self.EE2step(time)
        elif self.params["Model"] == "Corr":
            return self.stepEE2(time)
        return

    def dipole(self):
        return self.field.Expectation(self.rho, self.C)

    def energy(self, Plao, IfPrint=False):
        if self.params["Model"] in ["TDHF", "BBGKY", "TDCIS"]:
            Hlao = TransMat(self.H, self.X)
            return (self.Enuc + np.trace(np.dot(Plao, Hlao + self.F))).real
        elif self.params["Model"] == "TDDFT":
            Hlao = TransMat(self.H, self.X)
            P = TransMat(Plao, self.X, -1)
            print("P", P)
            J = self.get_j(2 * P)
            Exc = self.Exc
            if self.hyb > 0.01:
                Exc -= 0.5 * self.hyb * np.trace(np.dot(P, self.K))
            EH = np.trace(np.dot(Plao, 2 * Hlao))
            EJ = np.trace(np.dot(P, J))
            E = EH + EJ + Exc + self.Enuc
            print("ONE e", EH.real)
            print("COLOUM", EJ.real)
            print("EXC", Exc.real)
            print("Nuclear", self.Enuc)
            print("", self.the_scf.energy_nuc())
            return E.real

    def loginstant(self, iter):
        np.set_printoptions(precision=7)
        if self.params["ApplyCw"] == 1:
            tore = str(self.t) + " " + str(np.diag(self.rho).real).rstrip(']').lstrip('[')
        else:
            tore = str(self.t) + " " + str(self.dipole().real).rstrip(']').lstrip('[') + " " + str(self.energy(TransMat(self.rho, self.V, -1), False)) + " " + str((np.trace(self.rho)).real)
        if iter % self.params["StatusEvery"] == 0 or iter == self.params["MaxIter"] - 1:
            print('t:', self.t * FsPerAu, " (Fs)  Energy:", self.energy(TransMat(self.rho, self.V, -1)), " Tr ", (np.trace(self.rho)).real)
            print('Dipole moment(X, Y, Z, au): %8.5f, %8.5f, %8.5f' % (self.dipole().real[0], self.dipole().real[1], self.dipole().real[2]))
        return tore

    def WriteEH(self, iter=1):
        rhoAO = 2.0 * TransMat(TransMat(self.rho, self.V, -1), self.X, -1)
        print("Trace(rhoAO*S):", np.trace(np.dot(rhoAO, self.S)))
        print("Trace(rhoAO0*S):", np.trace(np.dot(self.rhoAO0, self.S)))
        rhoAO -= self.rhoAO0

        print("Difference Trace(rhoAO*S):", np.trace(np.dot(rhoAO, self.S)))
        eig, eigv = np.linalg.eigh(np.dot(rhoAO, self.S).real)
        print("Diff Nocc:", eig)
        nao = int(self.n_ao)
        for i in range(nao):
            for j in range(nao):
                if not np.isfinite(eigv[i, j]):
                    eigv[i, j] *= 0.0

        eigv = eigv.T
        ho = np.zeros(nao)
        hc = np.zeros((nao, nao))
        eo = np.zeros(nao)
        ec = np.zeros((nao, nao))
        for i in range(nao):
            if eig[i] < 0.0:
                ho[i] = -1. * eig[i].real
                hc[i, :] += eigv[i, :].real
            else:
                eo[i] = eig[i].real
                ec[i, :] += eigv[i, :].real

        print("hole:", sum(ho).real, "electron:", sum(eo).real)
        import pyscf.tools
        hole = open("hole.molden", "w")
        electron = open("electron.molden", "w")
        hc = hc.T
        pyscf.tools.molden.from_mo(self.mol, hole.name, hc, "Alpha", None, None, ho)
        ec = ec.T
        pyscf.tools.molden.from_mo(self.mol, electron.name, ec, "Alpha", None, None, eo)
        hole.close()
        electron.close()

    def prop(self, output):
        self.rhoAO0 = 2.0 * TransMat(TransMat(self.rho, self.V, -1), self.X, -1)
        EH = 0
        iter = 0
        self.t = 0
        f = open(output, 'a')
        print("Energy Gap (eV)", abs(self.eigs[self.n_occ] - self.eigs[self.n_occ - 1]) * 27.2114)
        print("\n\nPropagation Begins")
        start = time.time()
        while iter < self.params["MaxIter"]:
            if self.t > 2 * self.params["tOn"] and EH == 0:
                self.WriteEH()
                EH = 1
            self.step(self.t)
            f.write(self.loginstant(iter) + "\n")
            self.t = self.t + self.params["dt"]
            if iter % self.params["StatusEvery"] == 0 or iter == self.params["MaxIter"] - 1:
                end = time.time()
                print((end - start) / (60 * 60 * self.t * FsPerAu * 0.001), "hr/ps")
            iter += 1
        f.close()
