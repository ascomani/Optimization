#include <iostream>
#include <cmath>
#include <vector>
#include <random>


std::vector<std::vector<double>> generateRandomNumbers(int rows, int cols){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> distribution(0.0, 1.0);

    std::vector<std::vector<double>> result(rows, std::vector<double>(cols));

    for(size_t i=0; i<rows; i++){
        for(size_t j=0; j<cols; j++){
            result[i][j] = distribution(gen);
        }
    }

    return result;
}

std::pair<double, double> descend(const std::vector<std::vector<double>>& x, 
                                   const std::vector<std::vector<double>>& y, 
                                   double w, double b, double learning_rate) {
    double dldw = 0.0;
    double dldb = 0.0;
    int N = x.size() * x[0].size();

    for(size_t i=0; i<x.size(); i++){
        for(size_t j=0; j<x[i].size(); j++){
            double xi = x[i][j];
            double yi = y[i][j];
            dldw += -2*xi*(yi-(w*xi+b));
            dldb += -2*(yi-(w*xi+b));
        }
    }
    w -= (learning_rate * dldw)/N; 
    b -= (learning_rate * dldb)/N;

    return std::make_pair(w, b);
}

int main(){

    std::vector<std::vector<double>> x = generateRandomNumbers(10, 1);
    std::vector<std::vector<double>> y(x.size(), std::vector<double>(x[0].size()));


    for(size_t i=0; i<x.size(); i++){
        for(size_t j=0; j<x[i].size(); j++){
            y[i][j] = 5.0 * x[i][j] + std::rand() / (double)(RAND_MAX);
        }
    }

    // for(size_t i=0; i<y.size(); i++){
    //     for(size_t j=0; j<y[i].size(); j++){
    //         std::cout<<"x: ["<<x[i][j]<<"] "<<"y: ["<<y[i][j]<<"]"<<std::endl;
    //     }
    // }

    double w = 0.0;
    double b = 0.0;

    double learning_rate = 0.01;

    for (int epoch = 0; epoch < 800; ++epoch) {
        // Update weights and bias
        std::pair<double, double> updated_params = descend(x, y, w, b, learning_rate);
        w = updated_params.first;
        b = updated_params.second;


        // Calculate predicted output
        std::vector<std::vector<double>> yhat(y.size(), std::vector<double>(y[0].size()));
        for (int i = 0; i <y.size(); ++i) {
            for(int j = 0; j <y[0].size(); ++j) {
                yhat[i][j] = w * x[i][j] + b;
            }
        }

        // Calculate loss
        double loss = 0.0;
        for (int i = 0; i <y.size(); ++i) {
            for(int j = 0; j <y[0].size(); ++j) {
                loss += std::pow((yhat[i][j] - y[i][j]), 2);
            }
        }
        loss /= (y.size() * y[0].size());

        std::cout << epoch << " loss is " << loss << ", parameters w: " << w << ", b: " << b << std::endl;
        }
    return 0;
}