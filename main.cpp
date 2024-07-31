// #include <iostream>
// #include <torch/torch.h>
// #include <opencv2/opencv.hpp>
// #include <SFML/Graphics.hpp>

// #include "dataset.h"
// #include "KAN.cpp"
// #include "KANsLinear.cpp"
// #include "train.cpp"
// #include "UI.cpp"



// int main() {
//     App app;
//     app.run();
//     return 0;
// }

#include <torch/torch.h>
#include <vector>
#include <iostream>
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

std::vector<torch::Tensor> createBatch() {
    return {
        torch::arange(10, torch::kFloat),
        torch::arange(10, torch::kFloat) * 2,
        torch::arange(10, torch::kFloat) * 3,
        torch::arange(10, torch::kFloat) * 4,
        torch::arange(10, torch::kFloat) * 5,
        torch::arange(10, torch::kFloat) * 6,
        torch::arange(10, torch::kFloat) * 7,
        torch::arange(10, torch::kFloat) * 8,
        torch::arange(10, torch::kFloat) * 9,
        torch::arange(10, torch::kFloat) * 10
    };
}

std::vector<float> tensorToVector(const torch::Tensor& tensor) {
    auto tensor_cpu = tensor.cpu().contiguous();
    int64_t num_elements = tensor_cpu.numel();
    std::vector<float> vec(num_elements);
    std::memcpy(vec.data(), tensor_cpu.data_ptr<float>(), num_elements * sizeof(float));
    return vec;
}

std::vector<std::vector<float>> batchToVectors(const std::vector<torch::Tensor>& batch) {
    std::vector<std::vector<float>> batch_vectors;
    for (const auto& tensor : batch) {
        batch_vectors.push_back(tensorToVector(tensor));
    }
    return batch_vectors;
}

void plotBatch(const std::vector<std::vector<float>>& batch_vectors) {
    const int num_plots = batch_vectors.size();
    const int plots_per_row = 10;
    
    const int subplot_size = 100;
    const int figure_width = subplot_size * plots_per_row;
    const int figure_height = subplot_size * ((num_plots + plots_per_row - 1) / plots_per_row);

    plt::figure_size(figure_width, figure_height);  
    
    for (size_t i = 0; i < num_plots; ++i) {
        plt::subplot((num_plots + plots_per_row - 1) / plots_per_row, plots_per_row, i + 1);
        plt::plot(batch_vectors[i]);
        plt::title("Plot " + std::to_string(i + 1));
        plt::xlabel("Index");
        plt::ylabel("Value");
    }
    
    plt::tight_layout();
    plt::show();

}

int main() {
    std::vector<torch::Tensor> batch = createBatch();
    std::vector<std::vector<float>> batch_vectors = batchToVectors(batch);

    // Plot the batch of vectors
    plotBatch(batch_vectors);

    return 0;
}


