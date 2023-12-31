//
// Created by Brent Zhang on 2023/12/26.
//

#ifndef CUPDLP_ONLINELP_H
#define CUPDLP_ONLINELP_H

#define DBG_ONLINE_LP (0)


#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <random>
#include <unordered_set>

using namespace Eigen;
typedef Eigen::SparseMatrix<double, Eigen::ColMajor> SpMat;
typedef Eigen::Triplet<double> T;
using eigen_array = ArrayXd;
using eigen_array_int = ArrayXi;
using eigen_buff = Eigen::Map<ArrayXd>;

SpMat getRandomSpMat(size_t nRows, size_t nCols, double p);


#endif  // CUPDLP_ONLINELP_H
