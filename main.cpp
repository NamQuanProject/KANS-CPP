#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "dataset.h"
#include "KAN.cpp"
#include "KANsLinear.cpp"
#include "train.cpp"





int main() {
    trainMNIST();
    return 0;
}