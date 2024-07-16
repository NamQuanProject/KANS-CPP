#pragma once
#include <torch/torch.h>
#include "dataset.h"
#include "KAN.h"
#include "KANsLinear.h"
#include "UI.h"
#include <SFML/Graphics.hpp>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

void trainMNIST();
void showImage(const torch::Tensor& tensor, const std::string& window_name);