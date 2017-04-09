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

  //validate inputs
  if ((estimations.size() == 0) || estimations.size() != ground_truth.size())
  {
    std::cout << "ERROR - invalid inputs to CalculateRMSE()!";
    return rmse;
  }

  //squared residual accumulation
  for (int i=0; i<estimations.size(); ++i)
  {
    VectorXd diff = estimations[i]-ground_truth[i];
    diff = diff.array() * diff.array();
    rmse += diff;
  }

  //calculate mean
  rmse /= estimations.size();

  //calculate square root
  rmse = rmse.array().sqrt();

  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  MatrixXd Hj(3, 4);

  Hj << 0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0;

  //state parameters
  float px, py, vx, vy;
  px = x_state[0];
  py = x_state[1];
  vx = x_state[2];
  vy = x_state[3];

  //computation for Jacobian matrix
  float c1, c2, c3;

  c1 = px*px+py*py;
  //divide by zero check
  if(fabs(c1) < 0.00001) {
    return Hj;
  }
  c2 = sqrt(c1);
  c3 = (c1*c2);

  //compute Jacobian
  Hj <<              (px/c2),             (py/c2),     0,     0,
                    -(py/c1),             (px/c1),     0,     0,
         py*(vx*py-vy*px)/c3, px*(vy*px-vx*py)/c3, px/c2, py/c2;

 return Hj;
}
