#!/usr/bin/env
# cython: boundscheck=False
# cython. wraparound=False
# cython: cdivision=True
# cython: profile=False

from __future__ import division
cimport cython
import numpy as np
from scipy import sparse
cimport numpy as np
from func cimport *

cdef class MDP:

  cdef readonly int C, G, B, tau, S, nzz, nzzz
  cdef readonly double prequest
  cdef readonly np.ndarray pg, ctr, policy, Q, ecpi
  cdef readonly list T, R

  def __init__(self, int C, int G, int B, int tau, double prequest,
               np.ndarray[np.float64_t, ndim=1] Pg,
               np.ndarray[np.float64_t, ndim=2] eCPI,
               np.ndarray[np.float64_t, ndim=2] CTR):
    self.C = C
    self.G = G
    self.B = B
    self.tau = tau
    self.ctr = CTR
    self.ecpi = eCPI
    self.pg = Pg
    self.prequest = prequest
    self.S = ipow(B+1,C)*(G+1)
    self.nzz = G*(self.S-ipow(B+1,C-1)*2)*2 + self.S
    self.nzzz = self.S*(G+1)
    self.populateMtx()

  cpdef int getIndexOfState(self, np.ndarray[np.int_t, ndim=1, mode="c"] s):
    cdef long out
    getIndexOfState(&out, self.C, self.G, self.B, <long*> s.data)
    return out

  cpdef np.ndarray[np.int_t, ndim=1, mode="c"] getStateOfIndex(self, index):
    cdef np.ndarray[np.int_t, ndim=1, mode="c"] state = np.zeros(self.C+1, dtype=np.int)
    getStateOfIndex(index, self.C, self.G, self.B, <long*> state.data)
    return state

  def populateMtx(self):
    cdef np.ndarray[np.int_t, ndim=1, mode="c"] rows, cols
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] values, valuesR
    self.T = []
    self.R = []
    for i in xrange(self.C+1):
      if (i < self.C):
        rows = np.zeros(self.nzz, dtype=np.int)
        cols = np.zeros(self.nzz, dtype=np.int)
        values = np.zeros(self.nzz, dtype=np.float64)
        valuesR = np.zeros(self.S, dtype=np.float64)
      else:
        rows = np.zeros(self.nzzz, dtype=np.int)
        cols = np.zeros(self.nzzz, dtype=np.int)
        values = np.zeros(self.nzzz, dtype=np.float64)
        valuesR = np.zeros(self.S, dtype=np.float64)
      populate(i,self.C,self.G,self.B,self.prequest,<double*>self.pg.data,<double*>self.ctr.data,
               <double*>self.ecpi.data,<long*>rows.data,<long*>cols.data,<double*>values.data, <double*>valuesR.data)

      self.T.append(sparse.coo_matrix((values, (rows, cols)), shape=(self.S,self.S)))
      self.R.append(sparse.coo_matrix((valuesR), shape=(self.S,1)))
    for i in xrange(self.C+1):
      self.T[i] = self.T[i].tocsr()
      self.R[i] = self.R[i].tocsr()


  def valueIteration(self, double gamma, double error):
    cdef np.ndarray[np.float64_t, ndim=1] V
    cdef np.ndarray[np.float64_t, ndim=2] Q
    cdef np.ndarray[np.int_t, ndim=1] policy

    V = np.ones(self.S,dtype='float')
    Q = np.zeros((self.S,self.C+1),dtype='float')
    policy = np.zeros((self.S),dtype='int')

    for a in xrange(self.C+1):
      Q[:,a] = np.asarray(self.T[a].dot(gamma*V))
      Q[:,a] += self.R[a]

    errr = abs(max(Q.max(1)-V))

    while (errr > error):    
      for a in xrange(self.C+1):
        Q[:,a] = np.asarray(self.T[a].dot(gamma*V))
        Q[:,a] += self.R[a]
      errr = abs(max(Q.max(1)-V))
      V = Q.max(1)

    self.policy = Q.argmax(1)
    self.Q = Q

  def valueIterationT(self):
    cdef np.ndarray[np.float64_t, ndim=1] V
    cdef np.ndarray[np.float64_t, ndim=2] Q
    cdef np.ndarray[np.int_t, ndim=2] policy

    V = np.ones(self.S,dtype='float')
    Q = np.zeros((self.S,self.C+1),dtype='float')
    policy = np.zeros((self.S,self.tau),dtype='int')

    for t in xrange(self.tau):
      for a in xrange(self.C+1):
        Q[:,a] = self.T[a].dot(V).T
        Q[:,a] += self.R[a]
      V = Q.max(1)
      policy[:,t] = Q.argmax(1)

    self.policy = policy
    self.Q = Q