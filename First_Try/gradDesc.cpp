#include <iostream>
#include <random>
#include <vector>
#include <iomanip>


//Fucntion for generating random Gaussian numbers for a matrix or vector
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

//Function for calculating the shape of a vector or matrix
std::vector<int> shapeofVec(std::vector<std::vector<double>> &matrix){
    int rows = matrix.size();
    int columns = (rows > 0) ? matrix[0].size() : 0;

    std::vector<int> shape = {rows, columns};

    return shape;
}

//Function for generating the output: Y
std::vector<std::vector<double>> generateY(const std::vector<std::vector<double>> &x){
    std::vector<std::vector<double>> y(x.size(), std::vector<double>(x[0].size()));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> distribution(0.0, 1.0);

    for(size_t i=0; i<x.size(); ++i){
        for(size_t j=0; j<x[i].size(); ++j){
            double noise = distribution(gen);
            y[i][j] = 4 * x[i][j] + noise;
        }
    }

    return y;
}

//Create a function for printing matrices
void printMatrix(const std::vector<std::vector<double>>& matrix){
    for (const auto& row : matrix) {
        for (const auto& elem : row) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(2) << elem;
        }
        std::cout << std::endl;
    }
}

std::vector<std::vector<double>> generateYhat(std::vector<std::vector<double>> &x, double w, double b){

    std::vector<std::vector<double>> yhat(x.size(), std::vector<double>(x[0].size()));
    for(size_t i=0; i<x.size(); ++i){
            for(size_t j=0; j<x[i].size(); ++j){
                yhat[i][j] = w * x[i][j] +b;
            }
        }
    return yhat;
}

//Build Gradient Descent Function
std::pair<double, double> descend(std::vector<std::vector<double>>& x, std::vector<std::vector<double>>& y, double w, double b, double learning_rate){
    //Initialize variables for the derivative of the Loss with respect to W and B
    double dl_db = 0.0;
    double dl_dw = 0.0;

    int N = shapeofVec(x)[0];

    for(size_t i = 0; i < N; ++i){
        for(size_t j = 0; j<x[i].size(); ++j){
            double xi = x[i][j];
            double yi = y[i][j];

            dl_db += -2 * xi * (yi - (w*xi+b));
            dl_dw += -2 * (yi - (w*xi+b));
        }
    }

    w -= (learning_rate * dl_dw)/N; 
    b -= (learning_rate * dl_db)/N;

    return std::make_pair(w, b);
}

std::vector<double> MSE(std::vector<std::vector<double>>& y, std::vector<std::vector<double>>& yhat){
    std::vector<double> loss(1, 0.0);

    for(size_t i=0; i<y.size(); ++i){
        for(size_t j=0; j<y[0].size(); ++j){
            loss[0] += (y[i][j] - yhat[i][j]) * (y[i][j] - yhat[i][j]);
        }
    }
    int num_elements = (y.size() * y[0].size());
    loss[0] /= num_elements;

    return loss;
}


int main(){
    
    //Generate the input data and print it
    int rows = 10;
    int cols = 1;
    std::vector<std::vector<double>> x = generateRandomNumbers(rows, cols);
    //std::cout<<"Input Data: "<<std::endl;
    //printMatrix(x);

    //Generate the output data and print it
    std::vector<std::vector<double>> y = generateY(x);
    //std::cout<<"Output Data: \n";
    //printMatrix(y);

    //Inititialize Parameters
    double w = 0.0;
    double b = 0.0;

    //Initialize Hypaparameters
    double learning_rate = 0.0001;
    double prev_loss = std::numeric_limits<double>::infinity();


    for(int epoch=0; epoch<401; ++epoch){
        //Update weights and bias using the gradient descent function
        std::pair<double, double> updatedParams = descend(x, y, w, b, learning_rate);
        w = updatedParams.first;
        b = updatedParams.second;

        //calculate predicted output, yhat
        std::vector<std::vector<double>> yhat(x.size(), std::vector<double>(x[0].size()));
        yhat = generateYhat(x, w, b);

        //calculate loss
        std::vector<double> loss(1, 0.0);
        loss = MSE(y, yhat);
        double current_loss = loss[0];


        std::cout<<"For epoch "<<epoch<<" loss is: "<<current_loss<<", parameters w: "<<w<<", b: "<<b<<std::endl;

        //set loss and epoch limit to prevent overfitting
        if(std::abs(prev_loss - current_loss) < 1e-6){
            std::cout<<"Early stopping at epoch: "<<epoch<<std::endl;
            break;
        }
        prev_loss = current_loss;
    }
    return 0;
}