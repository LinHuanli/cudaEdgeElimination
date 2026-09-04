/*
 * 仅用于研究期强度 oracle：把 Concorde 的 violated-cuts 回调转换为稳定文本。
 * 本文件不复制 Concorde 实现；构建时显式链接仓库外只读源码生成的静态库。
 */
#include "cut.h"
#include "machdefs.h"
#include "util.h"

static int emit_cut(double value, int count, int* nodes, void* unused) {
  int i;
  (void)value;
  (void)unused;
  printf("CUT %d", count);
  for (i = 0; i < count; ++i) {
    printf(" %d", nodes[i]);
  }
  printf("\n");
  return fflush(stdout) == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
  int rval = 0;
  int ncount = 0;
  int ecount = 0;
  int* elist = (int*)NULL;
  double* capacity = (double*)NULL;
  if (argc != 2) {
    fprintf(stderr, "usage: %s EDGE_CAPACITY_FILE\n", argv[0]);
    return 2;
  }
  rval = CCutil_getedges_double(&ncount, argv[1], &ecount, &elist, &capacity, 0);
  if (rval) {
    fprintf(stderr, "CCutil_getedges_double failed\n");
    goto CLEANUP;
  }
  rval = CCcut_violated_cuts(ncount, ecount, elist, capacity, 2.0 - CC_MINCUT_ONE_EPSILON, emit_cut,
                             (void*)NULL);
  if (rval) {
    fprintf(stderr, "CCcut_violated_cuts failed\n");
  }

CLEANUP:
  CC_IFFREE(elist, int);
  CC_IFFREE(capacity, double);
  return rval;
}
