#include <iostream>
#include <stdexcept>

#include "kalman.hpp"

KalmanFilter::KalmanFilter(
    double dt,
    const Eigen::MatrixXd processMatrix,
    const Eigen::MatrixXd obsMtx,
    const Eigen::MatrixXd processNoiseCov,
    const Eigen::MatrixXd obsNoiseCov,
    const Eigen::MatrixXd estimatErrCov)
    //initializing the values - constructing member variables
    : processMtx(processMatrix)
    , obsMtx(obsMtx)
    , processNoiseCov(processNoiseCov)
    , obsNoiseCov(obsNoiseCov)
    , initEstimatedErrCov(estimatErrCov)
    , stateDim(obsMtx.rows())
    , obsDim(processMatrix.rows())
    , dt(dt)
    , initialized(false)
    , identityMtx(obsDim, obsDim)
    , x_hat(obsDim)
    , x_hat_new(obsDim)
{
  identityMtx.setIdentity();
}

KalmanFilter::KalmanFilter() {}

void KalmanFilter::init(double t0, const Eigen::VectorXd& x0) {
  x_hat = x0;
    estimatErrCov = initEstimatedErrCov;
  this->t0 = t0;
  t = t0;
  initialized = true;
}

void KalmanFilter::init() {
  x_hat.setZero();
    estimatErrCov = initEstimatedErrCov;
  t0 = 0;
  t = t0;
  initialized = true;
}

void KalmanFilter::update(const Eigen::VectorXd& obs) {

  if(!initialized)
    throw std::runtime_error("Filter is not initialized!");

  x_hat_new = processMtx * x_hat;
  estimatErrCov = processMtx * estimatErrCov * processMtx.transpose() + processNoiseCov;
    kalmanGain = estimatErrCov * obsMtx.transpose() * (obsMtx * estimatErrCov * obsMtx.transpose() + obsNoiseCov).inverse();
  x_hat_new += kalmanGain * (obs - obsMtx * x_hat_new);
  estimatErrCov = (identityMtx - kalmanGain * obsMtx) * estimatErrCov;
  x_hat = x_hat_new;

  t += dt;
}

void KalmanFilter::update(const Eigen::VectorXd& obs, double dt, const Eigen::MatrixXd processMtx) {

  this->processMtx = processMtx;
  this->dt = dt;
  update(obs);
}
