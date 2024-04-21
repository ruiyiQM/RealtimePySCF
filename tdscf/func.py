import numpy as np
import os
import sys
import ctypes
from pyscf import lib

OCCDROP = 1e-12
BLKSIZE = 96

def load_library(libname):
    # numpy 1.6 has bug in ctypeslib.load_library, see numpy/distutils/misc_util.py
    # Update: As numpy 1.6 is very old and such compatibility is rarely needed, consider removing this check or updating numpy.
    so_ext = {'linux': '.so', 'darwin': '.dylib', 'win32': '.dll'}.get(sys.platform, None)
    if so_ext is None:
        raise OSError('Unknown platform')
    libname_so = libname + so_ext
    return ctypes.CDLL(os.path.join(os.path.dirname(__file__), libname_so))

libdft = lib.load_library('libdft')
libcvhf = lib.load_library('libcvhf')

def eval_rhoc(mol, ao, mo_coeff, mo_occ, non0tab=None, xctype='LDA', verbose=None):
    assert ao.flags.c_contiguous
    xctype = xctype.upper()
    if xctype == 'LDA':
        ngrids, nao = ao.shape
    else:
        ngrids, nao = ao[0].shape

    if non0tab is None:
        non0tab = np.ones(((ngrids + BLKSIZE - 1) // BLKSIZE, mol.nbas), dtype=np.int8)
    pos = mo_occ.real > OCCDROP
    cpos = np.einsum('ij,j->ij', mo_coeff[:, pos], np.sqrt(mo_occ[pos]))
    rho = np.zeros((6, ngrids)) if xctype != 'LDA' else np.zeros(ngrids)
    
    if pos.sum() > 0:
        if xctype == 'LDA':
            c0 = _dot_ao_dm(mol, ao, cpos, nao, ngrids, non0tab)
            rho = np.einsum('pi,pi->p', c0, c0.conj())
        elif xctype == 'GGA':
            c0 = _dot_ao_dm(mol, ao[0], cpos, nao, ngrids, non0tab)
            rho[0] = np.einsum('pi,pi->p', c0, c0.conj())
            for i in range(1, 4):
                c1 = _dot_ao_dm(mol, ao[i], cpos, nao, ngrids, non0tab)
                rho[i] = np.einsum('pi,pi->p', c0, c1.conj()) * 2  # *2 for +c.c.

    neg = mo_occ.real < -OCCDROP
    if neg.sum() > 0:
        cneg = np.einsum('ij,j->ij', mo_coeff[:, neg], np.sqrt(-mo_occ[neg]))
        if xctype == 'LDA':
            c0 = _dot_ao_dm(mol, ao, cneg, nao, ngrids, non0tab)
            rho -= np.einsum('pi,pi->p', c0, c0.conj())
        elif xctype == 'GGA':
            c0 = _dot_ao_dm(mol, ao[0], cneg, nao, ngrids, non0tab)
            rho[0] -= np.einsum('pi,pi->p', c0, c0.conj())
            for i in range(1, 4):
                c1 = _dot_ao_dm(mol, ao[i], cneg, nao, ngrids, non0tab)
                rho[i] -= np.einsum('pi,pi->p', c0, c1.conj()) * 2  # *2 for +c.c.
    
    return rho.real if xctype == 'LDA' else rho

def TransMat(M, U, inv=1):
    if inv == 1:
        # U.t() * M * U
        Mtilde = np.dot(np.dot(U.T.conj(), M), U)
    elif inv == -1:
        # U * M * U.t()
        Mtilde = np.dot(np.dot(U, M), U.T.conj())
    return Mtilde

def TrDot(A, B):
    return np.trace(np.dot(A, B))

def MatrixPower(A, p, PrintCondition=False):
    u, s, v = np.linalg.svd(A)
    if PrintCondition:
        print("MatrixPower: Minimal Eigenvalue =", np.min(s))
    s = np.maximum(s, np.power(10.0, -14.0))
    return np.dot(u, np.dot(np.diag(np.power(s, p)), v))

def _dot_ao_dm(mol, ao, dm, nao, ngrids, non0tab):
    natm = ctypes.c_int(mol._atm.shape[0])
    nbas = ctypes.c_int(mol._bas.shape[0])
    vm = np.empty((ngrids, dm.shape[1]), dtype=np.complex128)
    ao = np.asarray(ao, order='C')
    dm = np.asarray(dm, order='C')
    libdft.VXCdot_ao_dm(vm.ctypes.data_as(ctypes.c_void_p),
                        ao.ctypes.data_as(ctypes.c_void_p),
                        dm.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(nao), ctypes.c_int(dm.shape[1]),
                        ctypes.c_int(ngrids), ctypes.c_int(BLKSIZE),
                        non0tab.ctypes.data_as(ctypes.c_void_p),
                        mol._atm.ctypes.data_as(ctypes.c_void_p),
                        mol._bas.ctypes.data_as(ctypes.c_void_p),
                        mol._env.ctypes.data_as(ctypes.c_void_p))
    return vm
