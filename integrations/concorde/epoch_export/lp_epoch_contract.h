#ifndef CUDAEE_LP_EPOCH_CONTRACT_H
#define CUDAEE_LP_EPOCH_CONTRACT_H

/*
 * 此声明由 overlay 补丁加入 Concorde INCLUDE/lp.h。
 * edge_u/edge_v 必须按当前 LP 列顺序提供，长度等于 CClp_ncols(lp)。
 */
int CClp_dump_lp_epoch(CClp* lp,
                       const char* filename,
                       int edge_count,
                       const int* edge_u,
                       const int* edge_v);

#endif
