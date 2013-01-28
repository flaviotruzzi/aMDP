#ifndef FUNC_H
#define FUNC_H

double crandom(void);
long ipow(long base, long exp);
void getIndexOfState(long *p, long C, long G, long B, long *state);
void getStateOfIndex(long p, long C, long G, long B, long *state);
void populate(long a, long C, long G, long B, double prequest, double *pg,
                double *ctr, double *ecpi, long *rows, long *cols,
              double *values, double *valuesR);

#endif
