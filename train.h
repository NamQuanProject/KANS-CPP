#pragma once
#include <torch/torch.h>
#include "dataset.h"
#include "KAN.h"
#include "KANsLinear.h"

void trainMNIST();
void showImage(const torch::Tensor& tensor, const std::string& window_name);