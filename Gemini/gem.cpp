#include <iostream>
#include <vector>
#include <random>
#include <cmath>

// Function to generate random numbers
std::vector<double> generateRandomNumbers(int size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<double> distribution(0.0, 1.0);

  std::vector<double> result(size);
  for (int i = 0; i < size; ++i) {
    result[i] = distribution(gen);
  }
  return result;
}

// Gradient Descent Function
std::pair<double, double> descend(const std::vector<double>& x, 
                                   const std::vector<double>& y, 
                                   double w, double b, double learning_rate) {
  double dldw = 0.0;
  double dldb = 0.0;
  int N = x.size();

  for (size_t i = 0; i < N; ++i) {
    double xi = x[i];
    double yi = y[i];
    dldw += -2 * xi * (yi - (w * xi + b));
    dldb += -2 * (yi - (w * xi + b));
  }

  w -= learning_rate * (1.0 / N) * dldw;
  b -= learning_rate * (1.0 / N) * dldb;

  return std::make_pair(w, b);
}

int main() {
  // Generate random data
  int num_datapoints = 10;
  std::vector<double> x = generateRandomNumbers(num_datapoints);
  std::vector<double> y;
  for (double xi : x) {
    y.push_back(5.0 * xi + std::rand() / (double)(RAND_MAX)); // Assuming RAND_MAX is defined
  }

  // Initialize parameters
  double w = 0.0;
  double b = 0.0;

  // Hyperparameter
  double learning_rate = 0.01;

  // Training loop
  for (int epoch = 0; epoch < 800; ++epoch) {
    // Update weights and bias
    std::pair<double, double> updated_params = descend(x, y, w, b, learning_rate);
    w = updated_params.first;
    b = updated_params.second;

    // Calculate predicted output
    std::vector<double> yhat(num_datapoints);
    for (int i = 0; i < num_datapoints; ++i) {
      yhat[i] = w * x[i] + b;
    }

    // Calculate loss
    double loss = 0.0;
    for (int i = 0; i < num_datapoints; ++i) {
      loss += std::pow(y[i] - yhat[i], 2.0);
    }
    loss /= num_datapoints;

    // Print results
    std::cout << epoch << " loss is " << loss << ", parameters w: " << w << ", b: " << b << std::endl;
  }

  // Print original data (optional)
  // std::cout << "x: ";
  // for (double xi : x) {
  //   std::cout << xi << " ";
  // }
  // std::cout << std::endl;
  // std::cout << "y: ";
  // for (double yi : y) {
  //   std::cout << yi << " ";
  // }
  // std::cout << std::endl;

  return 0;
}
