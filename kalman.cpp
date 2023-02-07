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
        : processMtx(processMatrix), obsMtx(obsMtx), processNoiseCov(processNoiseCov), obsNoiseCov(obsNoiseCov),
          initEstimatedErrCov(estimatErrCov), stateDim(obsMtx.rows()), obsDim(processMatrix.rows()), timeDiff(dt),
          initialized(false), identityMtx(obsDim, obsDim), estimatedState(obsDim), newEstimatedState(obsDim) {
    identityMtx.setIdentity();
}

KalmanFilter::KalmanFilter() {}

void KalmanFilter::init(double initTime, const Eigen::VectorXd initState) {
    estimatedState = initState;
    estimatErrCov = initEstimatedErrCov;
    this->initTime = initTime;
    curTime = initTime;
    initialized = true;
}

void KalmanFilter::init() {
    estimatedState.setZero();
    estimatErrCov = initEstimatedErrCov;
    initTime = 0;
    curTime = initTime;
    initialized = true;
}

void KalmanFilter::update(const Eigen::VectorXd obs) {

    if (!initialized)
        throw std::runtime_error("Filter is not initialized!");

    newEstimatedState = processMtx * estimatedState;
    estimatErrCov = processMtx * estimatErrCov * processMtx.transpose() + processNoiseCov;
    kalmanGain =
            estimatErrCov * obsMtx.transpose() * (obsMtx * estimatErrCov * obsMtx.transpose() + obsNoiseCov).inverse();
    newEstimatedState += kalmanGain * (obs - obsMtx * newEstimatedState);
    estimatErrCov = (identityMtx - kalmanGain * obsMtx) * estimatErrCov;
    estimatedState = newEstimatedState;

    curTime += timeDiff;
}

void KalmanFilter::update(const Eigen::VectorXd obs, double timeDiff, const Eigen::MatrixXd processMtx) {

    this->processMtx = processMtx;
    this->timeDiff = timeDiff;
    update(obs);
}
