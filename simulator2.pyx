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

cdef extern from "math.h" nogil:
  int rand()

cdef extern from "stdio.h" nogil:
  printf(char* string)

cdef inline int randomWeighted(
  np.float64_t* acc,
  int size) nogil:
  cdef int i
  cdef double throw = crandom()
  for i in xrange(size):
    if (throw < acc[i]):
      return i

cdef double crandom() nogil:
    return rand()%10000/10000.0

cdef class Simulator:

  cdef readonly double [:,:] CTR
  cdef readonly np.ndarray values, budgets
  cdef readonly double [:] Pg, CPC
  cdef readonly int C, G, B, tau, nmeans, S
  cdef readonly double Prequest
  cdef np.float64_t *AccPg
  cdef long index

  def __init__(self, int C, int G, int B, int tau, int nmeans,
            double Prequest, double [:,:] CTR, double [:] CPC,
            double [:] Pg):

    cdef int g

    self.C = C
    self.G = G
    self.B = B
    self.tau = tau
    self.nmeans = nmeans
    self.Prequest = Prequest
    self.CTR = CTR
    self.CPC = CPC
    self.Pg = Pg
    self.S = int(pow(B+1,C))*(G+1)
    self.AccPg = <np.float64_t*>malloc((G+1)*sizeof(np.float64_t))
    self.AccPg[0] = 1-Prequest
    for g in xrange(1,G+1):
      self.AccPg[g] = Prequest*Pg[g-1]+self.AccPg[g-1]

  def simulate(self, long [:] policy):
    cdef int r, t, c, b
    cdef np.ndarray[np.int_t, ndim=1] s = np.empty(self.C+1, dtype=np.int)
    cdef np.int_t* ss
    cdef double [:] CPC = self.CPC
    cdef double [:,:] CTR = self.CTR
    cdef double [:,:] values
    cdef long [:,:] budgets
    self.values = np.zeros((1,self.tau), dtype='double')
    self.budgets = np.zeros((self.tau,self.C), dtype='int')
    values = self.values
    budgets = self.budgets

#    with nogil:
    for r in xrange(self.nmeans): #, schedule='static', num_threads=4):

      getStateOfIndex(self.S-1,self.C,self.G,self.B,<long*>s.data)
      ss = &s[0]

      for t in xrange(self.tau):
        for b in xrange(self.C):
          budgets[t,b] += ss[b]
        
        ss[self.C] = randomWeighted(self.AccPg, self.G+1) 
        getIndexOfState(&self.index,self.C,self.G,self.B,<long*>s.data)
        c = policy[self.index]
      
        if ( (ss[self.C]>0) and (c<self.C) and (ss[c] > 0) ):
          if ( crandom() < CTR[ss[self.C]-1,c] ):
            values[0,t] += CPC[c]
            ss[c] = ss[c]-1

      for t in xrange(self.tau):
        values[0,t] = values[0,t]/self.nmeans

