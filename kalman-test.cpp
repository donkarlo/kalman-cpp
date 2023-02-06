#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "kalman.hpp"

using namespace std;

int main(int argc, char* argv[]) {

  int stateDim = 3; // Number of states
  int obsDim = 1; // Number of obss

  double dt = 1.0/30; // Time step

  /*
   * Defining the matrices
   */
  Eigen::MatrixXd processMtx(stateDim, stateDim); // System dynamics matrix
  Eigen::MatrixXd obsMtx(obsDim, stateDim); // Output matrix
  Eigen::MatrixXd processNoiseCov(stateDim, stateDim); // Process noise covariance
  Eigen::MatrixXd obsNoiseCov(obsDim, obsDim); // Measurement noise covariance
  Eigen::MatrixXd estimatErrCov(stateDim, stateDim); // Estimate error covariance

  // Discrete LTI projectile motion, measuring position only
  processMtx << 1, dt, 0,
                0, 1, dt,
                0, 0, 1;
  obsMtx << 1, 0, 0;

  // Reasonable covariance matrices
  processNoiseCov << .05, .05, .0,
                     .05, .05, .0,
                     .0, .0, .0;
  obsNoiseCov << 5;
  estimatErrCov << .1, .1, .1,
                   .1, 10000, 10,
                   .1, 10, 100;

  cout << "processMtx: \n" << processMtx << endl;
  cout << "obsMtx: \n" << obsMtx << endl;
  cout << "processNoiseCov: \n" << processNoiseCov << endl;
  cout << "obsNoiseCov: \n" << obsNoiseCov << endl;
  cout << "estimatErrCov: \n" << estimatErrCov << endl;

  // Construct the filter
  KalmanFilter kf(dt, processMtx, obsMtx, processNoiseCov, obsNoiseCov, estimatErrCov);

  // List of noisy position obss (y)
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
  Eigen::VectorXd x0(stateDim);
  double t = 0;
  x0 << obss[0], 0, -9.81;
  kf.init(t, x0);

  // Feed obss into filter, output estimated states

  Eigen::VectorXd y(obsDim);
  cout << "t = " << t << ", " << "x_hat[0]: " << kf.state().transpose() << endl;
  for(int i = 0; i < obss.size(); i++) {
    t += dt;
    y << obss[i];
    kf.update(y);
    cout << "t = " << t << ", " << "obss[" << i << "] = " << y.transpose()
        << ", x_hat[" << i << "] = " << kf.state().transpose() << endl;
  }

  return 0;
}