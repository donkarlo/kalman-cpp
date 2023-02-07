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
    void init(double initTime, const Eigen::VectorXd initState);

    /**
    * Update the estimated getEstimatedState based on measured values. The
    * getCurTime step is assumed to remain constant.
    */
    void update(const Eigen::VectorXd obs);

    /**
    * Update the estimated getEstimatedState based on measured values,
    * using the given getCurTime step and dynamics matrix.
    */
    void update(const Eigen::VectorXd obs, double timeDiff, const Eigen::MatrixXd processMtx);

    /**
    * Return the current getEstimatedState and getCurTime.
    */
    Eigen::VectorXd getEstimatedState() { return estimatedState; };

    double getCurTime() { return curTime; };

private:

    // Matrices for computation
    Eigen::MatrixXd processMtx, obsMtx, processNoiseCov, obsNoiseCov, estimatErrCov, kalmanGain, initEstimatedErrCov;

    // System dimensions
    int stateDim, obsDim;

    // Initial and current getCurTime
    double initTime, curTime;

    // Discrete getCurTime step
    double timeDiff;

    // Is the filter initialized?
    bool initialized;

    // obsDim-size identity
    Eigen::MatrixXd identityMtx;

    // Estimated states
    Eigen::VectorXd estimatedState, newEstimatedState;

};
