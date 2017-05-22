#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;
  
  long long previous_timestamp_ = 0.0;
  
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);
  
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  H_laser_ << 1,0,0,0,
              0,1,0,0;
  
  Hj_ <<  0,0,0,0,
          0,0,0,0,
          0,0,0,0;
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}


void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  double rho, phi, rhodot;
  
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    
    cout << "EKF Initialization: " << endl;
    
    ekf_.x_ = VectorXd(4);
    
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      double x, y, vx, vy;
      
      // Convert polar to cartesian coorditates
      rho = measurement_pack.raw_measurements_(0);
      phi = measurement_pack.raw_measurements_(1);
      rhodot = measurement_pack.raw_measurements_(2);
      x = rho * cos(phi);
      y = rho * sin(phi);
      vx = rhodot * cos(phi);
      vy = rhodot * sin(phi);
      
      // Initialize state
      ekf_.x_ << x, y, vx, vy;
    } else {
      
      // Initialize state with location and velocity of zero
      ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0.0, 0.0;
    }
    
    // Initialize Covariance Matrix
    ekf_.P_ = MatrixXd(4,4);
    ekf_.P_ <<  1,0,0,0,
                0,1,0,0,
                0,0,1000,0,
                0,0,0,1000;
    previous_timestamp_ = measurement_pack.timestamp_;
    is_initialized_ = true;
    
    return;
  }
  
  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  // Get difference between current and previous measurements
  float dt = (measurement_pack.timestamp_ - previous_timestamp_)/1000000.0;
  
  // Update previous to current
  previous_timestamp_ = measurement_pack.timestamp_;
  
  // Initialize transition matrix F_
  ekf_.F_ = MatrixXd(4,4);
  ekf_.F_ <<  1,0,dt,0,
              0,1,0,dt,
              0,0,1,0,
              0,0,0,1;
  
  // Set acceleration noise components
  float noise_ax = 9.0;
  float noise_ay = 9.0;
  
  // Pre-compute for non-repetition
  float dt_2 = dt * dt;
  float dt_3 = dt_2 * dt;
  float dt_4 = dt_3 * dt;
  float c1 = dt_4 / 4;
  float c2 = dt_3 / 2;
  
  // Update process noise covariance matrix
  ekf_.Q_ = MatrixXd(4,4);
  ekf_.Q_ <<  (c1 * noise_ax), 0, (c2 * noise_ax), 0,
              0, (c1 * noise_ay), 0, (c2 * noise_ay),
              (c2 * noise_ax), 0, (dt_2 * noise_ax), 0,
              0, (c2 * noise_ay), 0, (dt_2 * noise_ay);
  
  ekf_.Predict();
  
  /*****************************************************************************
   *  Update
   ****************************************************************************/
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      
        // Update radar
        VectorXd z_Radar(3);
        z_Radar << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], measurement_pack.raw_measurements_[2];
        
        Tools t;
        ekf_.H_ = t.CalculateJacobian(ekf_.x_);
        ekf_.R_ = R_radar_;
        ekf_.UpdateEKF(z_Radar);
      
    } else {
      
        // Update laser
        VectorXd z_Laser(2);
        z_Laser << measurement_pack.raw_measurements_(0), measurement_pack.raw_measurements_(1);
        ekf_.H_ = H_laser_ ;
        ekf_.R_ = R_laser_ ;
        ekf_.Update(z_Laser);
      }
  
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
