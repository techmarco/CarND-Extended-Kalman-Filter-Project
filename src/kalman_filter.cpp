#include "kalman_filter.h"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
  I_ = MatrixXd::Identity(4, 4);
}

void KalmanFilter::Predict() {
  //predict the state
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  //perform measurement update
  VectorXd y = z - H_ * x_;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = P_ * Ht * Si;

  //new state
  x_ = x_ + (K * y);
  P_ = (I_ - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  Tools tools;

  float px, py, px2, py2, vx, vy;
  px = x_[0];
  py = x_[1];
  vx = x_[2];
  vy = x_[3];

  px2 = px * px;
  py2 = py * py;

  VectorXd h = VectorXd(3);
  h(0) = sqrt(px2 + py2);
  h(1) = atan2(py, px);
  //divide by zero check
  if (h(0) < 0.00001)
    h(2) = 0;
  else
    h(2) = (px*vx+py*vy) / h(0);

  //perform measurement update
  VectorXd y = z - h;
  MatrixXd Hj = tools.CalculateJacobian(x_);
  MatrixXd Ht = Hj.transpose();
  MatrixXd S = Hj * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = P_ * Ht * Si;

  //new state
  x_ = x_ + (K * y);
  P_ = (I_ - K * Hj) * P_;
}
