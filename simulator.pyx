#!/usr/bin/env
# cython: boundscheck=False
# cython. wraparound=False
# cython: cdivision=True
# cython: profile=True

from __future__ import division
import numpy as np
cimport numpy as np
cimport cython
from scipy import sparse
cimport numpy as np
from func cimport *

cdef extern from "stdlib.h":
  void free(void* ptr)
  void* malloc(size_t size)

@cython.profile(False)
cdef inline int randomWeighted(
  double throw,
  np.float64_t* acc,
  int size):
  cdef int i
  for i in xrange(size):
    if (throw < acc[i]):
      return i

cdef inline np.ndarray[np.float64_t, ndim=2] simulateT(
  long [:,:,:] sim,
  np.int_t* policy,
  np.float64_t* cpc,
  int nmeans,
  int tau,
  int S,
  int C,
  int G,
  int B,
  double [:,:] budgets):

  cdef np.ndarray[np.float64_t, ndim=2] values = np.empty((nmeans, tau), dtype=np.float64)
  cdef np.ndarray[np.int_t, ndim=1] s = np.empty(C+1, dtype=np.int)
  cdef int r, t, c, g,
  cdef long index

  cdef np.float64_t* val = &values[0,0]
  cdef np.int_t* ss
  for r in xrange(nmeans):
    getStateOfIndex(S-1,C,G,B,<long*>s.data)
    ss = &s[0]
    for t in xrange(tau):
      for blah in xrange(C):
        budgets[t,blah] += ss[blah]
      g = sim[C,t,r]
      ss[C] = g
      getIndexOfState(&index,C,G,B,<long*>s.data)
      c = policy[index*tau + tau - t -1]
      if ( (g>0) and (c<C) and (ss[c]>0) and (sim[c,t,r] > 0) ):
        val[r*tau+t] = sim[c,t,r]*cpc[c]
        ss[c] = ss[c]-1
      else:
        val[r*tau+t] = 0
  return values

cdef inline np.ndarray[np.float64_t, ndim=2] simulate(
  long [:,:,:] sim,
  np.int_t* policy,
  np.float64_t* cpc,
  int nmeans,
  int tau,
  int S,
  int C,
  int G,
  int B,
	double [:,:] budgets):

  cdef np.ndarray[np.float64_t, ndim=2] values = np.empty((nmeans, tau), dtype=np.float64)
  cdef np.ndarray[np.int_t, ndim=1] s = np.empty(C+1, dtype=np.int)
  cdef int r, t, c, g,
  cdef long index

  cdef np.float64_t* val = &values[0,0]
  cdef np.int_t* ss
  for r in xrange(nmeans):
    getStateOfIndex(S-1,C,G,B,<long*>s.data)
    ss = &s[0]
    for t in xrange(tau):
      for blah in xrange(C):
        budgets[t,blah] += ss[blah]
      g = sim[C,t,r]
      ss[C] = g
      getIndexOfState(&index,C,G,B,<long*>s.data)
      c = policy[index]
      if ( (g>0) and (c<C) and (ss[c]>0) and (sim[c,t,r] > 0) ):
        val[r*tau+t] = sim[c,t,r]*cpc[c]
        ss[c] = ss[c]-1
      else:
        val[r*tau+t] = 0
  return values

cdef class Simulator:

  cdef readonly int C, G, B, tau, nmeans, S
  cdef readonly double Prequest
  cdef readonly np.ndarray values, budgets, sim, CTR
  cdef np.float64_t *CPC, *Pg, *AccPg
  cdef int r, t, g, c

  def __init__(self, int C, int G, int B, int tau, int nmeans,
    double Prequest, np.ndarray[np.float64_t, ndim=2, mode="c"] CTR,
    np.ndarray[np.float64_t, ndim=1, mode="c"] CPC,
    np.ndarray[np.float64_t, ndim=1, mode="c"] Pg):
    self.C = C
    self.G = G
    self.B = B
    self.S = int(pow(B+1,C))*(G+1)
    self.tau = tau
    self.nmeans = nmeans
    self.Pg = &Pg[0]
    self.CTR = CTR
    self.CPC = &CPC[0]
    self.AccPg = <np.float64_t*>malloc((G+1)*sizeof(np.float64_t))
    self.AccPg[0] = 1-Prequest
    for g in xrange(1,G+1):
      self.AccPg[g] = Prequest*self.Pg[g-1]+self.AccPg[g-1]

    self.sim = np.zeros((C+1, tau, nmeans), dtype='int') #<np.int_t*>malloc((nmeans*(C+1)*tau)*sizeof(np.int_t))
    self.budgets = np.zeros((tau,C))
    for r in xrange(nmeans):
       for t in xrange(tau):
        g = randomWeighted(np.random.rand(), self.AccPg, G+1)
        if (g > 0):
          for c in xrange(C):
            if (np.random.rand() < self.CTR[(g-1),c]):
              self.sim[c,t,r] = 1
        self.sim[C,t,r] = g

  def __dealloc__(self):
    free(self.AccPg)

  def randomWeighted(self):
    return randomWeighted(np.random.rand(), self.AccPg, self.G+1)

  def simulate(self, np.ndarray[np.int_t, ndim=1, mode="c"] policy):
    self.budgets = np.zeros((self.tau,self.C))
    self.values = simulate(self.sim, &policy[0], self.CPC, self.nmeans, self.tau, self.S, self.C, self.G, self.B, self.budgets)

  def simulateT(self, np.ndarray[np.int_t, ndim=2, mode="c"] policy):
    self.budgets = np.zeros((self.tau,self.C))
    self.values = simulateT(self.sim, &policy[0,0], self.CPC, self.nmeans, self.tau, self.S, self.C, self.G, self.B, self.budgets)

