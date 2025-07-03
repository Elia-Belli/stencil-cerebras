#pragma once 

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

using namespace std;

void readCSV(string filename, float* buff, int rows, int cols);
void printMatrix(float *grid, int rows, int cols);


void readCSV(string filename, float* buff, int rows, int cols) {
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "Error opening file!" << endl;
        return; 
    }

    string line;
    int i = 1;
    while (getline(file, line)) {
        stringstream ss(line);
        string value;      

        int j = 1;
        while (getline(ss, value, ' ')) {
            buff[i*cols + j] = stof(value);
            j++;
        }
        i++;
    }

    file.close();
}

void printMatrix(float *grid, int rows, int cols){
    int i, j;
    for(i = 0; i < rows; ++i){
        for(j = 0; j < cols; ++j){
            cout << " " << grid[i*cols + j] << " ";
        }
        cout << endl;
    }

    cout << endl;
}
