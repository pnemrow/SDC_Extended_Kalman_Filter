#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;
  
  // Check if estimation vector is of nonzero and same length as ground truth vector length
  if (estimations.size() != ground_truth.size() || estimations.size() == 0) {
    std::cout << "Invalid estimation or ground truth data" << std::endl;
    return rmse;
  }
  
  // Sum squared residuals
  for (unsigned int i = 0; i < estimations.size(); ++i) {
    
    VectorXd residual = estimations[i] - ground_truth[i];
    
    // Multiply coefficients
    residual = residual.array() * residual.array();
    
    rmse += residual;
  }
  
  //Calculate the mean
  rmse = rmse / estimations.size();
  
  //Calculate the square root
  rmse = rmse.array().sqrt();
  
  //return rmse
  return rmse;
  
}

MatrixXd Tools::CalculateJacobian(const VectorXd &x_state) {
  
  MatrixXd Hj(3, 4);
  
  // State parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);
  
  // Pre-compute for non-repetition
  float c1 = (px * px) + (py * py);
  float c2 = sqrt(c1);
  float c3 = (c1 * c2);
  
  // Check for division by zero
  if (fabs(c1) < 0.001) {
    
    std::cout << "CalculateJacobian () - Error - Division by Zero" << std::endl;
    
    return Hj;
  }
  
  //Calculate the Jacobian matrix values
  Hj << (px / c2), (py / c2), 0, 0,
        -(py / c1), (px / c1), 0, 0,
        (py*(vx * py - vy * px)/c3),  (px*(vy * px - vx * py)/c3),  (px/c2), (py/c2);
  
  return Hj;
  
}
