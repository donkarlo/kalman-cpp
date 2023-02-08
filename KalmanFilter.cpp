#include <iostream>
#include <stdexcept>

#include "KalmanFilter.hpp"
using namespace std;

KalmanFilter::KalmanFilter(
        double dt,
        Eigen::MatrixXd processMatrix,
        const Eigen::MatrixXd obsMtx,
        const Eigen::MatrixXd processNoiseCov,
        const Eigen::MatrixXd obsNoiseCov,
        const Eigen::MatrixXd estimatedErrCov)
        //initializing the values - constructing member variables
        : processMtx(processMatrix), obsMtx(obsMtx), processNoiseCov(processNoiseCov), obsNoiseCov(obsNoiseCov),
          initEstimatedErrCov(estimatedErrCov), stateDim(obsMtx.rows()), obsDim(processMatrix.rows()), timeDiff(dt),
          initialized(false), identityMtx(obsDim, obsDim), estimatedState(obsDim), newEstimatedState(obsDim) {
    identityMtx.setIdentity();
}

void KalmanFilter::init(double initTime, const Eigen::VectorXd initState) {
    estimatedState = initState;
    estimatedErrCov = initEstimatedErrCov;
    this->initTime = initTime;
    curTime = initTime;
    initialized = true;
}

void KalmanFilter::init() {
    estimatedState.setZero();
    estimatedErrCov = initEstimatedErrCov;
    initTime = 0;
    curTime = initTime;
    initialized = true;
}

void KalmanFilter::update(const Eigen::VectorXd obs) {

    if (!initialized)
        throw std::runtime_error("Filter is not initialized!");

    newEstimatedState = processMtx * estimatedState;
    estimatedErrCov = processMtx * estimatedErrCov * processMtx.transpose() + processNoiseCov;
    kalmanGain = estimatedErrCov * obsMtx.transpose() * (obsMtx * estimatedErrCov * obsMtx.transpose() + obsNoiseCov).inverse();
    newEstimatedState += kalmanGain * (obs - obsMtx * newEstimatedState);
    estimatedErrCov = (identityMtx - kalmanGain * obsMtx) * estimatedErrCov;
    estimatedState = newEstimatedState;

    curTime += timeDiff;
    obsCounter ++;
    reportAfterUpdate(obs);
}

void KalmanFilter::update(const Eigen::VectorXd obs, double timeDiff, const Eigen::MatrixXd processMtx) {
    this->processMtx = processMtx;
    this->timeDiff = timeDiff;
    update(obs);
}

void KalmanFilter::report() {
    cout << "Process Matrix: \n" << processMtx << endl;
    cout << "Observation Matrix: \n" << obsMtx << endl;
    cout << "Process noise covariance: \n" << processNoiseCov << endl;
    cout << "Observation noise covariance: \n" << obsNoiseCov << endl;
    cout << "Estimated error covariance: \n" << estimatedErrCov << endl;
    cout << "current time = " << curTime << ", " << "estimatedState[0]: " << estimatedState.transpose() << endl;
}

void KalmanFilter::reportAfterUpdate(const Eigen::VectorXd obs) {
    cout << "current time = " << curTime << ", " << "observation: " << obsCounter << " = " << obs.transpose()
         << ", estimatedState[" << obsCounter << "] = " << getEstimatedState().transpose() << endl;
    cout << "estimated error cov determinant: "<<getEstimatedErrCov().determinant()<<endl;
}