#include <iostream>
#include <random>
#include <vector>
#include <iomanip>

std::vector<std::vector<double>> generateRandomNumbers(int rows, int columns){
    std::vector<std::vector<double>> result(rows, std::vector<double>(columns));
    

    // Initialize random number generator with a seed
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> distribution(0.0, 1.0);

    for(int i=0; i<rows; ++i){
        for(int j=0; j<columns; ++j){
            result[i][j] = distribution(gen);
        }
    }

    return result;

}

std::vector<std::vector<double>> generateY(const std::vector<std::vector<double>> &x){
    std::vector<std::vector<double>> y(x.size(), std::vector<double>(x[0].size()));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> distribution(0.0, 1.0);

    for(size_t i=0; i<x.size(); ++i){
        for(size_t j=0; j<x[i].size(); ++j){
            double noise = distribution(gen);
            y[i][j] = 2 * x[i][j] + noise;
        }
    }

    return y;
}

void printMatrix(const std::vector<std::vector<double>>& matrix) {
    std::cout<<"[";
    for (const auto& row : matrix) {
        // Iterate over each element in the row
        for (const auto& elem : row) {
            std::cout << std::setw(2) << std::fixed << std::setprecision(6) <<"["<< elem<<"]\n";
        };
    }
    std::cout<<"]\n";
}

int main(){
    
    int rows = 10;
    int cols = 1;
    std::vector<std::vector<double>> x = generateRandomNumbers(rows, cols);

    
    std::cout<<"Random Matrix: "<<std::endl;
    printMatrix(x);

    std::vector<std::vector<double>> y = generateY(x);
    printMatrix(y);
    return 0;
}