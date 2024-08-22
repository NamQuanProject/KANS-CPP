#include <iostream>
#include <filesystem>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <SFML/Graphics.hpp>
#include "dataset.h"
#include "KAN.cpp"
#include "KANsLinear.cpp"
#include "train.cpp"
#include "UI.cpp"
#include "testKAN.cpp"

int main() {
    App app;
    
    app.run();
    return 0;
}