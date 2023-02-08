#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "KalmanFilter.hpp"

using namespace std;

int main(int argc, char *argv[]) {

    int stateDim = 3; // Number of states
    int obsDim = 1; // Number of obss

    double timeDiff = 1.0 / 30; // Time step

    /*
     * Defining the matrices
     */
    Eigen::MatrixXd processMtx(stateDim, stateDim); // System dynamics matrix
    Eigen::MatrixXd obsMtx(obsDim, stateDim); // Output matrix
    Eigen::MatrixXd processNoiseCov(stateDim, stateDim); // Process noise covariance
    Eigen::MatrixXd obsNoiseCov(obsDim, obsDim); // observation noise covariance
    Eigen::MatrixXd estimatedErrCov(stateDim, stateDim); // Estimate error covariance

    // Discrete LTI projectile motion, measuring position only
    processMtx << 1, timeDiff, 0,
                  0, 1, timeDiff,
                  0, 0,        1;

    obsMtx << 1, 0, 0;

    //covariance matrices
    processNoiseCov << .05, .05, .0,
                       .05, .05, .0,
                       .0 ,  .0, .0;
    obsNoiseCov << 5;
    estimatedErrCov << .1, .1,    .1,
                       .1, 10000, 10,
                       .1, 10,    100;



    //filter construction
    KalmanFilter kf(timeDiff
                    , processMtx
                    , obsMtx
                    , processNoiseCov
                    , obsNoiseCov
                    , estimatedErrCov);

    // one dimensional observations obss (obsEigenVect) @todo change to vectors - this just for demontration
    vector<double> obss = {
            1.04202710058, 1.10726790452, 1.2913511148, 1.48485250951, 1.72825901034,
            1.74216489744, 2.11672039768, 2.14529225112, 2.16029641405, 2.21269371128,
            2.57709350237, 2.6682215744, 2.51641839428, 2.76034056782, 2.88131780617,
            2.88373786518, 2.9448468727, 2.82866600131, 3.0006601946, 3.12920591669,
            2.858361783, 2.83808170354, 2.68975330958, 2.66533185589, 2.81613499531,
            2.81003612051, 2.88321849354, 2.69789264832, 2.4342229249, 2.23464791825,
            2.30278776224, 2.02069770395, 1.94393985809, 1.82498398739, 1.52526230354,
            1.86967808173, 1.18073207847, 1.10729605087, 0.916168349913, 0.678547664519,
            0.562381751596, 0.355468474885, -0.155607486619, -0.287198661013, -0.602973173813
    };

    // Best guess of initial states
    Eigen::VectorXd initState(stateDim);
    double curTime = 0;
    initState << obss[0], 0, -9.81;
    //overloaded method init - either with an initial guess or set it to yero
    kf.init(curTime, initState);
    // kf.init();

    Eigen::VectorXd obsEigenVect(obsDim);
    kf.report();
    for (int obsCounter = 0; obsCounter < obss.size(); obsCounter++) {
        curTime += timeDiff;
        obsEigenVect << obss[obsCounter];
        kf.update(obsEigenVect);
    }

    return 0;
}


