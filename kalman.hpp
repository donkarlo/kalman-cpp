#include <Eigen/Dense>

#pragma once

class KalmanFilter {

public:

  /**
  *   The constructor
  *   processMatrix - System dynamics matrix
  *   obsMtx - observation matrix
  *   processNoiseCov - Process noise covariance
  *   obsNoiseCov - Measurement noise covariance
  *   estimatErrCov - Estimate error covariance
  */
  KalmanFilter(
      double dt,
      const Eigen::MatrixXd processMatrix,
      const Eigen::MatrixXd obsMtx,
      const Eigen::MatrixXd processNoiseCov,
      const Eigen::MatrixXd obsNoiseCov,
      const Eigen::MatrixXd estimatErrCov
  );

  /**
  * Create a blank estimator.
  */
  KalmanFilter();

  /**
  * Initialize the filter with initial states as zero.
  */
  void init();

  /**
  * Initialize the filter with a guess for initial states.
  */
  void init(double t0, const Eigen::VectorXd& x0);

  /**
  * Update the estimated state based on measured values. The
  * time step is assumed to remain constant.
  */
  void update(const Eigen::VectorXd& obs);

  /**
  * Update the estimated state based on measured values,
  * using the given time step and dynamics matrix.
  */
  void update(const Eigen::VectorXd& obs, double dt, const Eigen::MatrixXd processMtx);

  /**
  * Return the current state and time.
  */
  Eigen::VectorXd state() { return x_hat; };
  double time() { return t; };

private:

  // Matrices for computation
  Eigen::MatrixXd processMtx, obsMtx, processNoiseCov, obsNoiseCov, estimatErrCov, kalmanGain, initEstimatedErrCov;

  // System dimensions
  int stateDim, obsDim;

  // Initial and current time
  double t0, t;

  // Discrete time step
  double dt;

  // Is the filter initialized?
  bool initialized;

  // obsDim-size identity
  Eigen::MatrixXd identityMtx;

  // Estimated states
  Eigen::VectorXd x_hat, x_hat_new;

};
