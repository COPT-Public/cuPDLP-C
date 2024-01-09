#include "wrapper_highs.h"

#include <cassert>
#include <string>

#include "Highs.h"
// using namespace std;
using std::cout;
using std::endl;

extern "C" void *createModel_highs() { return new Highs(); }

extern "C" void deleteModel_highs(void *model) {
  if (model != NULL) delete (Highs *)model;
  // free(model);
}

extern "C" int loadMps_highs(void *model, const char *filename) {
  string str = string(filename);
  // model is lhs <= Ax <= rhs, l <= x <= u
  cout << "--------------------------------------------------" << endl;
  cout << "reading file..." << endl;
  cout << "\t" << std::string(filename) << endl;
  cout << "--------------------------------------------------" << endl;

  HighsStatus return_status = HighsStatus::kOk;
  return_status = ((Highs *)model)->readModel(str);

  if (return_status != HighsStatus::kOk) {
    printf("Error: readModel return status = %d\n", (int)return_status);
    return 1;
  }

  // relax MIP to LP
  const HighsLp &lp = ((Highs *)model)->getLp();
  if (lp.integrality_.size()) {
    for (int i = 0; i < lp.num_col_; i++) {
      if (lp.integrality_[i] != HighsVarType::kContinuous) {
        ((Highs *)model)->changeColIntegrality(i, HighsVarType::kContinuous);
      }
    }
  }

  return 0;
}

extern "C" void *presolvedModel_highs(void *presolve, void *model) {
  cout << "--------------------------------------------------" << endl;
  cout << "running presolve" << endl;
  cout << "--------------------------------------------------" << endl;

  HighsStatus return_status;
  return_status = ((Highs *)model)->presolve();

  assert(return_status == HighsStatus::kOk);

  HighsPresolveStatus model_presolve_status =
      ((Highs *)model)->getModelPresolveStatus();
  if (model_presolve_status == HighsPresolveStatus::kTimeout) {
    printf("Presolve timeout: return status = %d\n", (int)return_status);
  }
  HighsLp lp = ((Highs *)model)->getPresolvedLp();
  ((Highs *)presolve)->passModel(lp);

  return presolve;
}

extern "C" int formulateLP_highs(void *model, double **cost, int *nCols,
                                 int *nRows, int *nnz, int *nEqs, int **csc_beg,
                                 int **csc_idx, double **csc_val, double **rhs,
                                 double **lower, double **upper, double *offset,
                                 double *sign_origin, int *nCols_origin,
                                 int **constraint_new_idx) {
  int retcode = 0;

  const HighsLp &lp = ((Highs *)model)->getLp();

  // problem size for malloc
  int nCols_clp = lp.num_col_;
  int nRows_clp = lp.num_row_;
  int nnz_clp = lp.a_matrix_.start_[lp.num_col_];
  *nCols_origin = nCols_clp;
  *nRows = nRows_clp;    // need not recalculate
  *nCols = nCols_clp;    // need recalculate
  *nEqs = 0;             // need recalculate
  *nnz = nnz_clp;        // need recalculate
  *offset = lp.offset_;  // need not recalculate
  if (lp.sense_ == ObjSense::kMinimize) {
    *sign_origin = 1.0;
    printf("Minimize\n");
  } else if (lp.sense_ == ObjSense::kMaximize) {
    *sign_origin = -1.0;
    printf("Maximize\n");
  }
  if (*offset != 0.0) {
    printf("Has obj offset %f\n", *offset);
  } else {
    printf("No obj offset\n");
  }
  // allocate buffer memory
  constraint_type *constraint_type_clp = NULL;  // the ONLY one need to free
  // int *constraint_original_idx = NULL;  // pass by user is better, for
  // postsolve recovering dual

  const double *lhs_clp = lp.row_lower_.data();
  const double *rhs_clp = lp.row_upper_.data();
  const int *A_csc_beg = lp.a_matrix_.start_.data();
  const int *A_csc_idx = lp.a_matrix_.index_.data();
  const double *A_csc_val = lp.a_matrix_.value_.data();
  int has_lower, has_upper;

  CUPDLP_INIT(constraint_type_clp, nRows_clp);
  CUPDLP_INIT(*constraint_new_idx, *nRows);

  // recalculate nRows and nnz for Ax - z = 0
  for (int i = 0; i < nRows_clp; i++) {
    has_lower = lhs_clp[i] > -1e20;
    has_upper = rhs_clp[i] < 1e20;

    // count number of equations and rows
    if (has_lower && has_upper && lhs_clp[i] == rhs_clp[i]) {
      constraint_type_clp[i] = EQ;
      (*nEqs)++;
    } else if (has_lower && !has_upper) {
      constraint_type_clp[i] = GEQ;
    } else if (!has_lower && has_upper) {
      constraint_type_clp[i] = LEQ;
    } else if (has_lower && has_upper) {
      constraint_type_clp[i] = BOUND;
      (*nCols)++;
      (*nnz)++;
      (*nEqs)++;
    } else {
      // printf("Error: constraint %d has no lower and upper bound\n", i);
      // retcode = 1;
      // goto exit_cleanup;

      // what if regard free as bounded
      printf("Warning: constraint %d has no lower and upper bound\n", i);
      constraint_type_clp[i] = BOUND;
      (*nCols)++;
      (*nnz)++;
      (*nEqs)++;
    }
  }

  // allocate memory
  CUPDLP_INIT(*cost, *nCols);
  CUPDLP_INIT(*lower, *nCols);
  CUPDLP_INIT(*upper, *nCols);
  CUPDLP_INIT(*csc_beg, *nCols + 1);
  CUPDLP_INIT(*csc_idx, *nnz);
  CUPDLP_INIT(*csc_val, *nnz);
  CUPDLP_INIT(*rhs, *nRows);

  // cost, lower, upper
  for (int i = 0; i < nCols_clp; i++) {
    (*cost)[i] = lp.col_cost_[i] * (*sign_origin);
    (*lower)[i] = lp.col_lower_[i];

    (*upper)[i] = lp.col_upper_[i];
  }
  // slack costs
  for (int i = nCols_clp; i < *nCols; i++) {
    (*cost)[i] = 0.0;
  }
  // slack bounds
  for (int i = 0, j = nCols_clp; i < *nRows; i++) {
    if (constraint_type_clp[i] == BOUND) {
      (*lower)[j] = lhs_clp[i];
      (*upper)[j] = rhs_clp[i];
      j++;
    }
  }

  for (int i = 0; i < *nCols; i++) {
    if ((*lower)[i] < -1e20) (*lower)[i] = -INFINITY;
    if ((*upper)[i] > 1e20) (*upper)[i] = INFINITY;
  }

  // permute LP rhs
  // EQ or BOUND first
  for (int i = 0, j = 0; i < *nRows; i++) {
    if (constraint_type_clp[i] == EQ) {
      (*rhs)[j] = lhs_clp[i];
      (*constraint_new_idx)[i] = j;
      j++;
    } else if (constraint_type_clp[i] == BOUND) {
      (*rhs)[j] = 0.0;
      (*constraint_new_idx)[i] = j;
      j++;
    }
  }
  // then LEQ or GEQ
  for (int i = 0, j = *nEqs; i < *nRows; i++) {
    if (constraint_type_clp[i] == LEQ) {
      (*rhs)[j] = -rhs_clp[i];  // multiply -1
      (*constraint_new_idx)[i] = j;
      j++;
    } else if (constraint_type_clp[i] == GEQ) {
      (*rhs)[j] = lhs_clp[i];
      (*constraint_new_idx)[i] = j;
      j++;
    }
  }

  // formulate and permute LP matrix
  // beg remains the same
  for (int i = 0; i < nCols_clp + 1; i++) (*csc_beg)[i] = A_csc_beg[i];
  for (int i = nCols_clp + 1; i < *nCols + 1; i++)
    (*csc_beg)[i] = (*csc_beg)[i - 1] + 1;

  // row idx changes
  for (int i = 0, k = 0; i < nCols_clp; i++) {
    // same order as in rhs
    // EQ or BOUND first
    for (int j = (*csc_beg)[i]; j < (*csc_beg)[i + 1]; j++) {
      if (constraint_type_clp[A_csc_idx[j]] == EQ ||
          constraint_type_clp[A_csc_idx[j]] == BOUND) {
        (*csc_idx)[k] = (*constraint_new_idx)[A_csc_idx[j]];
        (*csc_val)[k] = A_csc_val[j];
        k++;
      }
    }
    // then LEQ or GEQ
    for (int j = (*csc_beg)[i]; j < (*csc_beg)[i + 1]; j++) {
      if (constraint_type_clp[A_csc_idx[j]] == LEQ) {
        (*csc_idx)[k] = (*constraint_new_idx)[A_csc_idx[j]];
        (*csc_val)[k] = -A_csc_val[j];  // multiply -1
        k++;
      } else if (constraint_type_clp[A_csc_idx[j]] == GEQ) {
        (*csc_idx)[k] = (*constraint_new_idx)[A_csc_idx[j]];
        (*csc_val)[k] = A_csc_val[j];
        k++;
      }
    }
  }

  // slacks for BOUND
  for (int i = 0, j = nCols_clp; i < *nRows; i++) {
    if (constraint_type_clp[i] == BOUND) {
      (*csc_idx)[(*csc_beg)[j]] = (*constraint_new_idx)[i];
      (*csc_val)[(*csc_beg)[j]] = -1.0;
      j++;
    }
  }

exit_cleanup:
  // free buffer memory
  if (constraint_type_clp != NULL) {
    free(constraint_type_clp);
    constraint_type_clp = NULL;
  }

  return retcode;
}