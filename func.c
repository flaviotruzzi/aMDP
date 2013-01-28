#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "func.h"

/*
 *  Return Random Weighted Sample
 */
int inline randomWeighted(double *acc, int size) {
  int i;
  double throw = crandom();
  for (i = 0; i < size; i++) 
    return i;
}

double inline crandom(void) {
  return rand()%10000/10000.0;
}


/*
 *  Long Exponentiation --- faster than math.h pow
 */
long inline ipow(long base, long exp)
{
    long result = 1;
    while (exp)
    {
        if (exp & 1)
            result *= base;
        exp >>= 1;
        base *= base;
    }

    return result;
}

/* 
 * Calculate Index of State, the index is stored in p variable.
 * The State Vector is translated to an index, no index colision is accepted
 */
void inline getIndexOfState(long *p, long C, long G, long B, long *state) {
  long i;
  long n = C-1;
  *p = 0;
  B = B+1;
  for (i = 0; i < C; i++) {
    *p = *p + state[i]*ipow(B,n);
    n = n-1;
  }
  *p = *p*(G+1) + state[C];
}

/*
 * Translate from index to State.
 * The state vector is changed.
 */
void inline getStateOfIndex(long p, long C, long G, long B, long *state) {
  long i;
  B = B+1;
  G = G+1;
  state[C] = p%G;
  p = p/G;
  for (i = C-1; i != -1; i--) {
    state[i] = p%B;
    p = p/B;
  }
}

/*
 * Populate Transition and Reinforcement Matrices for MDP class
 */
void populate(long a, long C, long G, long B, double prequest, double *pg,
              double *ctr, double *ecpi, long *rows, long *cols,
              double *values, double *valuesR) {

  long idx, S, s, g, aux;
  long *sa, *sc;
  double PI;

  // Auxiliary variables
  sa = (long*)malloc((C+1)*sizeof(long));
  sc = (long*)malloc((C+1)*sizeof(long));

  // Number of States of MDP
  S = ipow(B+1,C)*(G+1);
  idx = 0;


  // populate the matrices values, valuesR, rows and cols
  // Sparse matrix format: COO
  for (s = 0; s < S; s++) {
    getStateOfIndex(s,C,G,B,sa);
    for (g = 0; g < G+1; g++) {
      memcpy(sc,sa,C*sizeof(long));
      sc[C] = g;

      if (g == 0) PI = 1-prequest;
      else PI = prequest*pg[g-1];

      if ( (a < C) && (sa[C] > 0) && (sa[a] > 0) ) {
        rows[idx] = s;
        getIndexOfState(&cols[idx],C,G,B,sc);
        values[idx] = PI*(1-ctr[ (sa[C]-1)*(C) + a]);
        valuesR[s] = ecpi[ (sa[C]-1)*(C) + a];
        idx++;

        sc[a] = sc[a]-1;
        rows[idx] = s;
        getIndexOfState(&cols[idx],C,G,B,sc);
        values[idx] = PI*ctr[ (sa[C]-1)*(C) +a];
        idx++;
      } else {
        rows[idx] = s;
        getIndexOfState(&cols[idx],C,G,B,sc);
        values[idx] = PI;
        idx++;

      }
    }
  }
  free(sa);
  free(sc);
}

/*
void simulate(long C, long G, long B, long nmeans) {

  int r, t, c, b

  long *state = (long*)malloc((C+1)*sizeof(long));

  // Para cada task da simulação
  for (r = 0; r < nmeans; r++) {

    // Inicializa vetor
    for (b = 0; b < C; b++)
      state[b] = B;

    // Para cada instante de tempo
    for (t = 0; t < tau; t++) {

      // Salva budgets
      for (b = 0; b < C; b++)
        budgets[t,b] += state[b];

      state[C] = randomWeighted(AccPg, G+1);

      getIndexOfState(&index,C,G,B,state);
      c = policy[index];

      if ((state[C] > 0) && (c < C) && (state[c] > 0)) {
        if (crandom() < CTR[state[C]-1,c]) {
          values[t] += CPC[c];
          state[c] -= 1;
        }
      }


    }
    
          
  }

}
*/