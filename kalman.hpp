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
    *   estimatedErrCov - Estimate error covariance
    */
    KalmanFilter(
            double dt, // can be changede in this->update(.,.,.)
            Eigen::MatrixXd processMatrix, //can be changed in this->update(.,.,.)
            const Eigen::MatrixXd obsMtx,
            const Eigen::MatrixXd processNoiseCov,
            const Eigen::MatrixXd obsNoiseCov,
            const Eigen::MatrixXd estimatedErrCov
    );

    /**
    * Initialize the filter with initial states as zero.
    */
    void init();

    /**
    * Initialize the filter with a guess for initial states.
    */
    void init(double initTime, const Eigen::VectorXd initState);

    /**
    * update when timeDiff and processMtx are not changing
    */
    void update(const Eigen::VectorXd obs);

    /**
    * Update when timeDiff and processMtx vary over time
    */
    void update(const Eigen::VectorXd obs, double timeDiff, const Eigen::MatrixXd processMtx);

    /**
    * Return the current getEstimatedState.
    */
    Eigen::VectorXd getEstimatedState() { return estimatedState; };

    /**
     * @return current time
     */
    double getCurTime() { return curTime; };

    /**
     * @return current estimated time
     */
    Eigen::MatrixXd getEstimatedErrCov(){ return estimatedErrCov; };

private:

    // Matrices for computation
    Eigen::MatrixXd  processMtx
                    ,estimatedErrCov
                    ,kalmanGain;
    const Eigen::MatrixXd obsMtx
                        ,initEstimatedErrCov
                        ,obsNoiseCov
                        ,processNoiseCov
                        ;

    // System dimensions
    int stateDim, obsDim;

    // Initial and current getCurTime
    double initTime
         , curTime;

    // Discrete getCurTime step
    double timeDiff;

    // Is the filter initialized?
    bool initialized;

    // obsDim-size identity
    Eigen::MatrixXd identityMtx;

    // Estimated states
    Eigen::VectorXd estimatedState
                    ,newEstimatedState;

};
