#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Sparse>
#include <iostream>

namespace py = pybind11;

#include "../cupdlp/cupdlp.h"
#include "../interface/mps_lp.h"
#include "../interface/wrapper_highs.h"

using namespace Eigen;
typedef Eigen::SparseMatrix<cupdlp_float, Eigen::ColMajor> CscMat;

class cupdlp_interface {
 private:
  // params
  cupdlp_bool ifChangeIntParam[N_INT_USER_PARAM];
  cupdlp_int intParam[N_INT_USER_PARAM];
  cupdlp_bool ifChangeFloatParam[N_FLOAT_USER_PARAM];
  cupdlp_float floatParam[N_FLOAT_USER_PARAM];
  char *fout;
  char *fout_sol;
  cupdlp_bool ifSaveSol;
  cupdlp_bool ifPresolve;
  // work
  CUPDLPwork *w;
  // highs models
  void *original_model;
  void *presolved_model;
  void *model2solve;
  // problem data modifiable by users
  cupdlp_int nRows_pdlp;
  cupdlp_int nCols_pdlp;
  cupdlp_int nEqs_pdlp;
  cupdlp_int nnz_pdlp;
  cupdlp_float *rhs;
  cupdlp_float *cost;
  cupdlp_float *lower;
  cupdlp_float *upper;
  cupdlp_float offset;
  cupdlp_float sense;
  CscMat csc_cpu_input;
  // problem data not modifiable by users
  cupdlp_int ifReadMPS;
  cupdlp_int ifFreed;
  cupdlp_int ifDataChanged;
  cupdlp_int ifSovled;
  cupdlp_int *constraint_new_idx;
  cupdlp_int *constraint_type;
  cupdlp_int nCols;
  cupdlp_int nRows;
  cupdlp_float *col_value;
  cupdlp_float *col_dual;
  cupdlp_float *row_value;
  cupdlp_float *row_dual;
  cupdlp_int nCols_org;
  cupdlp_int nRows_org;
  cupdlp_int nCols_pre;
  cupdlp_int nRows_pre;
  cupdlp_int value_valid;
  cupdlp_int dual_valid;
  cupdlp_int *csc_beg, *csc_idx;
  cupdlp_float *csc_val;
  CUPDLPcsc *csc_cpu;
  CUPDLPscaling *scaling;
  CUPDLPproblem *prob;
  cupdlp_float alloc_matrix_time;
  cupdlp_float copy_vec_time;
  // solve info to return
  cupdlp_int iters;
  cupdlp_float solve_time;
  cupdlp_float scaling_time;
  cupdlp_float presolve_time;
  cupdlp_float PrimalObj;
  cupdlp_float DualObj;
  cupdlp_float DualityGap;
  cupdlp_float Comp;
  cupdlp_float PrimalFeas;
  cupdlp_float DualFeas;
  cupdlp_float PrimalObjAvg;
  cupdlp_float DualObjAvg;
  cupdlp_float DualityGapAvg;
  cupdlp_float CompAvg;
  cupdlp_float PrimalFeasAvg;
  cupdlp_float DualFeasAvg;

 public:
  cupdlp_interface() {
    // params
    for (int i = 0; i < N_INT_USER_PARAM; i++) {
      ifChangeIntParam[i] = false;
      intParam[i] = 0;
    }
    for (int i = 0; i < N_FLOAT_USER_PARAM; i++) {
      ifChangeFloatParam[i] = false;
      floatParam[i] = 0.0;
    }
    fout = "./solution-sum.json";
    fout_sol = "./solution.json";
    ifSaveSol = false;
    ifPresolve = false;
    // work
    w = NULL;
    // highs models
    original_model = NULL;
    presolved_model = NULL;
    model2solve = NULL;
    // problem data modifiable by users
    nRows_pdlp = 0;
    nCols_pdlp = 0;
    nEqs_pdlp = 0;
    nnz_pdlp = 0;
    rhs = NULL;
    cost = NULL;
    lower = NULL;
    upper = NULL;
    offset = 0.0;
    sense = 1;
    // problem data not modifiable by users
    ifReadMPS = 0;
    ifFreed = 0;
    ifDataChanged = 0;
    ifSovled = 0;
    constraint_new_idx = NULL;
    constraint_type = NULL;
    nCols = 0;
    nRows = 0;
    col_value = cupdlp_NULL;
    col_dual = cupdlp_NULL;
    row_value = cupdlp_NULL;
    row_dual = cupdlp_NULL;
    nCols_org = 0;
    nRows_org = 0;
    nCols_pre = 0;
    nRows_pre = 0;
    value_valid = 0;
    dual_valid = 0;
    csc_beg = NULL;
    csc_idx = NULL;
    csc_val = NULL;
    csc_cpu = NULL;
    scaling = NULL;
    prob = NULL;
    alloc_matrix_time = 0.0;
    copy_vec_time = 0.0;
    // solve info to return
    iters = 0;
    solve_time = 0.0;
    scaling_time = 0.0;
    presolve_time = 0.0;
    PrimalObj = 0.0;
    DualObj = 0.0;
    DualityGap = 0.0;
    Comp = 0.0;
    PrimalFeas = 0.0;
    DualFeas = 0.0;
    PrimalObjAvg = 0.0;
    DualObjAvg = 0.0;
    DualityGapAvg = 0.0;
    CompAvg = 0.0;
    PrimalFeasAvg = 0.0;
    DualFeasAvg = 0.0;
  }

  ~cupdlp_interface() {
    if (ifFreed == 0) {
      free_solver_mem();
    }
    free_highs_model();
  }

  void readMPS(const std::string &filename) {
    cupdlp_retcode retcode = RETCODE_OK;

    // free previous highs model
    free_highs_model();

    const char *fname = filename.c_str();

    original_model = createModel_highs();
    CUPDLP_CALL(loadMps_highs(original_model, fname));
    getModelSize_highs(original_model, &nCols_org, &nRows_org, NULL);
    nCols = nCols_org;
    nRows = nRows_org;
    ifReadMPS = 1;
    ifDataChanged = 1;
    ifSovled = 0;

  exit_cleanup:
    return;
  }

  void solve() {
    cupdlp_retcode retcode = RETCODE_OK;

    if (ifDataChanged == 1) {
      // free_solver_mem();

      if (ifReadMPS == 1) {
        retcode = get_data_from_highsmodel();
        if (retcode != RETCODE_OK) {
          cupdlp_printf("Error reading MPS file\n");
          goto exit_cleanup;
        }
      } else {
        retcode = get_data_from_input();
        if (retcode != RETCODE_OK) {
          cupdlp_printf("Error getting input data\n");
          goto exit_cleanup;
        }
      }
      retcode = prepare_lpdata();
      if (retcode != RETCODE_OK) {
        cupdlp_printf("Error loading data\n");
        goto exit_cleanup;
      }
    }

    retcode = solve_lpdata();
    if (retcode != RETCODE_OK) {
      cupdlp_printf("Error solving\n");
      goto exit_cleanup;
    }
  exit_cleanup:
    // free_solver_mem();
    return;
  }

  cupdlp_retcode get_data_from_highsmodel() {
    cupdlp_retcode retcode = RETCODE_OK;

    model2solve = original_model;

    if (ifChangeIntParam[IF_PRESOLVE]) {
      ifPresolve = intParam[IF_PRESOLVE];
    }

    presolve_time = getTimeStamp();
    if (ifPresolve) {
      presolved_model = createModel_highs();

      int presolve_status =
          presolvedModel_highs(presolved_model, original_model);
      getModelSize_highs(presolved_model, &nCols_pre, &nRows_pre, NULL);
      model2solve = presolved_model;
      nCols = nCols_pre;
      nRows = nRows_pre;
    }
    presolve_time = getTimeStamp() - presolve_time;

    CUPDLP_CALL(formulateLP_highs(
        model2solve, &cost, &nCols_pdlp, &nRows_pdlp, &nnz_pdlp, &nEqs_pdlp,
        &csc_beg, &csc_idx, &csc_val, &rhs, &lower, &upper, &offset, &sense,
        &nCols, &constraint_new_idx, &constraint_type));

    CUPDLP_CALL(csc_create(&csc_cpu));
    csc_cpu->nRows = nRows_pdlp;
    csc_cpu->nCols = nCols_pdlp;
    csc_cpu->nMatElem = nnz_pdlp;
    csc_cpu->colMatBeg = csc_beg;
    csc_cpu->colMatIdx = csc_idx;
    csc_cpu->colMatElem = csc_val;
  exit_cleanup:
    return retcode;
  }

  cupdlp_retcode get_data_from_input() {
    cupdlp_retcode retcode = RETCODE_OK;

    // if(csc_cpu_input == NULL){
    //     cupdlp_printf("Please input matrix A in csc format!\n");
    //     retcode = RETCODE_FAILED;
    //     goto exit_cleanup;
    // }
    if (rhs == NULL) {
      cupdlp_printf("Please input rhs vector!\n");
      retcode = RETCODE_FAILED;
      goto exit_cleanup;
    }
    if (cost == NULL) {
      cupdlp_printf("Please input cost vector!\n");
      retcode = RETCODE_FAILED;
      goto exit_cleanup;
    }
    if (lower == NULL) {
      cupdlp_printf("Please input lower vector!\n");
      retcode = RETCODE_FAILED;
      goto exit_cleanup;
    }
    if (upper == NULL) {
      cupdlp_printf("Please input upper vector!\n");
      retcode = RETCODE_FAILED;
      goto exit_cleanup;
    }

    nnz_pdlp = csc_cpu_input.nonZeros();
    nRows_pdlp = csc_cpu_input.rows();
    nCols_pdlp = csc_cpu_input.cols();
    nRows = nRows_pdlp;
    nCols = nCols_pdlp;

    if (nEqs_pdlp < 0 || nEqs_pdlp > nRows_pdlp) {
      cupdlp_printf("Please input correct number of equality constraints!\n");
      retcode = RETCODE_FAILED;
      goto exit_cleanup;
    }

    CUPDLP_CALL(csc_create(&csc_cpu));
    csc_cpu->nRows = nRows_pdlp;
    csc_cpu->nCols = nCols_pdlp;
    csc_cpu->nMatElem = nnz_pdlp;

    csc_cpu->colMatBeg = csc_cpu_input.outerIndexPtr();
    csc_cpu->colMatIdx = csc_cpu_input.innerIndexPtr();
    csc_cpu->colMatElem = csc_cpu_input.valuePtr();

  exit_cleanup:
    return retcode;
  }

  cupdlp_retcode prepare_lpdata() {
    cupdlp_retcode retcode = RETCODE_OK;
    cupdlp_int ifScaling = 1;
    CUPDLP_MATRIX_FORMAT src_matrix_format = CSC;
    CUPDLP_MATRIX_FORMAT dst_matrix_format = CSR_CSC;

    scaling = (CUPDLPscaling *)cupdlp_malloc(sizeof(CUPDLPscaling));
    CUPDLP_CALL(problem_create(&prob));
    CUPDLP_CALL(Init_Scaling(scaling, nCols_pdlp, nRows_pdlp, cost, rhs));

    if (ifChangeIntParam[IF_SCALING]) {
      ifScaling = intParam[IF_SCALING];
    }

    if (ifChangeIntParam[IF_RUIZ_SCALING]) {
      scaling->ifRuizScaling = intParam[IF_RUIZ_SCALING];
    }

    if (ifChangeIntParam[IF_L2_SCALING]) {
      scaling->ifL2Scaling = intParam[IF_L2_SCALING];
    }

    if (ifChangeIntParam[IF_PC_SCALING]) {
      scaling->ifPcScaling = intParam[IF_PC_SCALING];
    }

    // the work object needs to be established first
    // free inside cuPDLP

#if !(CUPDLP_CPU)
    csc_cpu->cuda_csc = NULL;
#endif

    scaling_time = getTimeStamp();
    CUPDLP_CALL(PDHG_Scale_Data(csc_cpu, ifScaling, scaling, cost, lower, upper, rhs));
    scaling_time = getTimeStamp() - scaling_time;

    alloc_matrix_time = 0.0;
    copy_vec_time = 0.0;

    CUPDLP_CALL(problem_alloc(prob, nRows_pdlp, nCols_pdlp, nEqs_pdlp, cost,
                              offset, sense, csc_cpu, src_matrix_format,
                              dst_matrix_format, rhs, lower, upper,
                              &alloc_matrix_time, &copy_vec_time));

    CUPDLP_INIT(col_value, nCols);
    CUPDLP_INIT(col_dual, nCols);
    CUPDLP_INIT(row_value, nRows);
    CUPDLP_INIT(row_dual, nRows);

    ifFreed = 0;

  exit_cleanup:
    return retcode;
  }

  cupdlp_retcode solve_lpdata() {
    cupdlp_retcode retcode = RETCODE_OK;
    cupdlp_float cuda_prepare_time;
    // solve
    CUPDLP_INIT_ZERO(w, 1);
#if !(CUPDLP_CPU)
    cuda_prepare_time = getTimeStamp();
    CHECK_CUSPARSE(cusparseCreate(&w->cusparsehandle));
    CHECK_CUBLAS(cublasCreate(&w->cublashandle));
    cuda_prepare_time = getTimeStamp() - cuda_prepare_time;
#endif

    w->problem = prob;
    w->scaling = scaling;
    PDHG_Alloc(w);
    w->timers->dScalingTime = scaling_time;
    w->timers->dPresolveTime = presolve_time;
    CUPDLP_COPY_VEC(w->rowScale, scaling->rowScale, cupdlp_float, nRows_pdlp);
    CUPDLP_COPY_VEC(w->colScale, scaling->colScale, cupdlp_float, nCols_pdlp);

#if !(CUPDLP_CPU)
    w->timers->AllocMem_CopyMatToDeviceTime += alloc_matrix_time;
    w->timers->CopyVecToDeviceTime += copy_vec_time;
    w->timers->CudaPrepareTime = cuda_prepare_time;
#endif

    cupdlp_printf("--------------------------------------------------\n");
    cupdlp_printf("enter main solve loop\n");
    cupdlp_printf("--------------------------------------------------\n");

    // CUPDLP_CALL(LP_SolvePDHG(w, ifChangeIntParam, intParam,
    // ifChangeFloatParam,
    //                         floatParam, fout, nCols, col_value, col_dual,
    //                         row_value, row_dual, &value_valid, &dual_valid,
    //                         0, fout_sol, constraint_new_idx,
    //                         constraint_type));

    // need the infos in w, so we do not use LP_SolvePDHG as it frees w
    PDHG_PrintHugeCUPDHG();
    CUPDLP_CALL(PDHG_SetUserParam(w, ifChangeIntParam, intParam,
                                  ifChangeFloatParam, floatParam));
    CUPDLP_CALL(PDHG_Solve(w));
    CUPDLP_CALL(PDHG_PostSolve(w, nCols, constraint_new_idx, constraint_type,
                               col_value, col_dual, row_value, row_dual,
                               &value_valid, &dual_valid));
    if (ifSaveSol && fout_sol) {
      if (strcmp(fout, fout_sol) != 0) {
        writeSol(fout_sol, nCols, w->problem->nRows, col_value, col_dual,
                 row_value, row_dual);
      } else {
        cupdlp_printf(
            "Warning: fp and fp_sol are the same, stop saving solution.\n");
      }
    }

    iters = w->timers->nIter;
    solve_time = w->timers->dSolvingTime;
    PrimalObj = w->resobj->dPrimalObj;
    DualObj = w->resobj->dDualObj;
    DualityGap = w->resobj->dDualityGap;
    Comp = w->resobj->dComplementarity;
    PrimalFeas = w->resobj->dPrimalFeas;
    DualFeas = w->resobj->dDualFeas;
    PrimalObjAvg = w->resobj->dPrimalObjAverage;
    DualObjAvg = w->resobj->dDualObjAverage;
    DualityGapAvg = w->resobj->dDualityGapAverage;
    CompAvg = w->resobj->dComplementarityAverage;
    PrimalFeasAvg = w->resobj->dPrimalFeasAverage;
    DualFeasAvg = w->resobj->dDualFeasAverage;

    // if (ifSaveSol) {
    //     writeSol(fout_sol, nCols, nRows, col_value, col_dual, row_value,
    //     row_dual);
    // }

    ifDataChanged = 1;
    ifSovled = 1;

  exit_cleanup:
    PDHG_Destroy(&w);
    return retcode;
  }

  void set_params(const py::dict &params_dict) {
    cupdlp_retcode retcode = RETCODE_OK;

    std::string tem_string;
    int argc = 2 * py::len(params_dict);
    // create char** argv length of argc
    char **argv = new char *[argc];
    int i = 0;
    for (auto item : params_dict) {
      tem_string = "-" + py::cast<const std::string>(py::str(item.first));
      argv[i++] = new char[tem_string.length() + 1];
      std::strcpy(argv[i - 1], tem_string.c_str());
      tem_string = py::cast<const std::string>(py::str(item.second));
      argv[i++] = new char[tem_string.length() + 1];
      std::strcpy(argv[i - 1], tem_string.c_str());
    };

    // load parameters
    for (cupdlp_int i = 0; i < argc - 1; i++) {
      // if (strcmp(argv[i], "-fname") == 0) {
      // fname = argv[i + 1];
      // } else
      if (strcmp(argv[i], "-out") == 0) {
        fout = argv[i + 1];
      } else if (strcmp(argv[i], "-h") == 0) {
        print_script_usage();
        break;
      } else if (strcmp(argv[i], "-savesol") == 0) {
        ifSaveSol = atoi(argv[i + 1]);
      } else if (strcmp(argv[i], "-ifPre") == 0) {
        if (ifPresolve != atoi(argv[i + 1])) {
          ifDataChanged = 1;
        }
        ifPresolve = atoi(argv[i + 1]);
      } else if (strcmp(argv[i], "-outSol") == 0) {
        fout_sol = argv[i + 1];
      }
    }
    if (strcmp(argv[argc - 1], "-h") == 0) {
      print_script_usage();
    }

    cupdlp_int ifScaling_old = intParam[IF_SCALING];
    cupdlp_int ifRuizScaling_old = intParam[IF_RUIZ_SCALING];
    cupdlp_int ifL2Scaling_old = intParam[IF_L2_SCALING];
    cupdlp_int ifPcScaling_old = intParam[IF_PC_SCALING];

    CUPDLP_CALL(getUserParam(argc, argv, ifChangeIntParam, intParam,
                             ifChangeFloatParam, floatParam));

    if (ifScaling_old != intParam[IF_SCALING]) {
      ifDataChanged = 1;
    }
    if (ifRuizScaling_old != intParam[IF_RUIZ_SCALING]) {
      ifDataChanged = 1;
    }
    if (ifL2Scaling_old != intParam[IF_L2_SCALING]) {
      ifDataChanged = 1;
    }
    if (ifPcScaling_old != intParam[IF_PC_SCALING]) {
      ifDataChanged = 1;
    }

  exit_cleanup:
    return;
  }

  void free_highs_model() {
    if (ifReadMPS == 1) {
      if (original_model != NULL) {
        deleteModel_highs(original_model);
      }
      if (ifPresolve) {
        if (presolved_model != NULL) {
          deleteModel_highs(presolved_model);
        }
      }
    }
  }

  void free_solver_mem() {
    if (col_value != NULL) cupdlp_free(col_value);
    if (col_dual != NULL) cupdlp_free(col_dual);
    if (row_value != NULL) cupdlp_free(row_value);
    if (row_dual != NULL) cupdlp_free(row_dual);

    // free problem
    if (scaling) {
      scaling_clear(scaling);
    }

    if (cost != NULL) cupdlp_free(cost);
    if (rhs != NULL) cupdlp_free(rhs);
    if (lower != NULL) cupdlp_free(lower);
    if (upper != NULL) cupdlp_free(upper);
    if (constraint_new_idx != NULL) cupdlp_free(constraint_new_idx);
    if (constraint_type != NULL) cupdlp_free(constraint_type);

    // free memory
    csc_clear(csc_cpu);
    problem_clear(prob);

    ifFreed = 1;
  }

  void load_lp_data(CscMat &csc_cpu_input_in,
                    const py::array_t<cupdlp_float> &cost_in,
                    const py::array_t<cupdlp_float> &rhs_in,
                    const py::array_t<cupdlp_float> &lower_in,
                    const py::array_t<cupdlp_float> &upper_in,
                    const cupdlp_int nEqs_pdlp_in) {
    // load data
    csc_cpu_input = csc_cpu_input_in;
    cupdlp_int csc_m = csc_cpu_input.rows();
    cupdlp_int csc_n = csc_cpu_input.cols();

    if (csc_m != rhs_in.size()) {
      cupdlp_printf("Row dims inconsistent !\n");
      return;
    }
    if (csc_n != cost_in.size() || csc_n != lower_in.size() ||
        csc_n != upper_in.size()) {
      cupdlp_printf("Col dims inconsistent !\n");
      return;
    }
    py::buffer_info cost_buf = cost_in.request();
    auto *cost_ptr = static_cast<cupdlp_float *>(cost_buf.ptr);
    cost = (cupdlp_float *)cupdlp_malloc(cost_in.size() * sizeof(cupdlp_float));
    memcpy(cost, cost_ptr, cost_in.size() * sizeof(cupdlp_float));
    py::buffer_info rhs_buf = rhs_in.request();
    auto *rhs_ptr = static_cast<cupdlp_float *>(rhs_buf.ptr);
    rhs = (cupdlp_float *)cupdlp_malloc(rhs_in.size() * sizeof(cupdlp_float));
    memcpy(rhs, rhs_ptr, rhs_in.size() * sizeof(cupdlp_float));
    py::buffer_info lower_buf = lower_in.request();
    auto *lower_ptr = static_cast<cupdlp_float *>(lower_buf.ptr);
    lower =
        (cupdlp_float *)cupdlp_malloc(lower_in.size() * sizeof(cupdlp_float));
    memcpy(lower, lower_ptr, lower_in.size() * sizeof(cupdlp_float));
    py::buffer_info upper_buf = upper_in.request();
    auto *upper_ptr = static_cast<cupdlp_float *>(upper_buf.ptr);
    upper =
        (cupdlp_float *)cupdlp_malloc(upper_in.size() * sizeof(cupdlp_float));
    memcpy(upper, upper_ptr, upper_in.size() * sizeof(cupdlp_float));
    nEqs_pdlp = nEqs_pdlp_in;
    ifDataChanged = 1;
    ifReadMPS = 0;
    ifSovled = 0;
  }

  // functon return dict in python

  py::dict get_solution() {
    py::dict solution_dict;
    if (ifSovled == 0) {
      cupdlp_printf("Please solve the problem first!\n");
    } else {
      solution_dict["x"] = py::array_t<cupdlp_float>(nCols, col_value);
      solution_dict["y"] = py::array_t<cupdlp_float>(nRows, row_dual);
      solution_dict["iters"] = iters;
      solution_dict["solve_time"] = solve_time;
      solution_dict["scaling_time"] = scaling_time;
      solution_dict["presolve_time"] = presolve_time;
      solution_dict["PrimalObj"] = PrimalObj;
      solution_dict["DualObj"] = DualObj;
      solution_dict["DualityGap"] = DualityGap;
      solution_dict["Comp"] = Comp;
      solution_dict["PrimalFeas"] = PrimalFeas;
      solution_dict["DualFeas"] = DualFeas;
      solution_dict["PrimalObjAvg"] = PrimalObjAvg;
      solution_dict["DualObjAvg"] = DualObjAvg;
      solution_dict["DualityGapAvg"] = DualityGapAvg;
      solution_dict["CompAvg"] = CompAvg;
      solution_dict["PrimalFeasAvg"] = PrimalFeasAvg;
      solution_dict["DualFeasAvg"] = DualFeasAvg;
    }
    return solution_dict;
  }

  void helper() { PDHG_PrintUserParamHelper(); }
};

PYBIND11_MODULE(pycupdlp, m) {
  m.doc() = "python interface for cuPDLP-C";

  py::class_<cupdlp_interface>(m, "cupdlp")
      .def(py::init())
      .def("readMPS", &cupdlp_interface::readMPS)
      .def("solve", &cupdlp_interface::solve)
      .def("setParams", &cupdlp_interface::set_params)
      .def("loadData", &cupdlp_interface::load_lp_data)
      .def("helper", &cupdlp_interface::helper)
      .def("getSolution", &cupdlp_interface::get_solution);
}