#include <torch/torch.h>
#include <iostream>
#include <vector>

torch::Tensor add_padding(torch::Tensor matrix, std::pair<int, int> padding) {
    int n = matrix.size(0);
    int m = matrix.size(1);
    int r = padding.first;
    int c = padding.second;
    
    auto padded_matrix = torch::zeros({n + 2 * r, m + 2 * c}, matrix.options());
    padded_matrix.slice(0, r, r + n).slice(1, c, c + m).copy_(matrix);
    
    return padded_matrix;
}

std::tuple<torch::Tensor, torch::Tensor, std::vector<int64_t>, int64_t, int64_t>
_check_params(torch::Tensor matrix, torch::Tensor kernel,
              std::pair<int64_t, int64_t> stride, std::pair<int64_t, int64_t> dilation,
              std::pair<int64_t, int64_t> padding) {
    TORCH_CHECK(matrix.ndimension() == 2, "Input matrix must be 2D");
    TORCH_CHECK(kernel.ndimension() == 2, "Kernel must be 2D");

    int64_t n = matrix.size(0);
    int64_t m = matrix.size(1);
    int64_t k1 = kernel.size(0);
    int64_t k2 = kernel.size(1);

    TORCH_CHECK(k1 % 2 == 1 && k2 % 2 == 1, "Kernel size must be odd");

    int64_t h_out = std::floor((n + 2 * padding.first - k1 - (k1 - 1) * (dilation.first - 1)) / stride.first) + 1;
    int64_t w_out = std::floor((m + 2 * padding.second - k2 - (k2 - 1) * (dilation.second - 1)) / stride.second) + 1;

    TORCH_CHECK(h_out > 0 && w_out > 0, "Invalid convolution parameters");

    return std::make_tuple(matrix, kernel, std::vector<int64_t>{k1, k2}, h_out, w_out);
}


torch::Tensor conv2d(torch::Tensor matrix, torch::Tensor kernel,
                     std::pair<int64_t, int64_t> stride = {1, 1},
                     std::pair<int64_t, int64_t> dilation = {1, 1},
                     std::pair<int64_t, int64_t> padding = {0, 0}) {
    auto [matrix_pad, kernel_, k, h_out, w_out] = _check_params(matrix, kernel, stride, dilation, padding);
    matrix_pad = add_padding(matrix_pad, {k[0] / 2, k[1] / 2});

    torch::Tensor matrix_out = torch::zeros({h_out, w_out}, matrix.options());

    for (int64_t i = 0; i < h_out; ++i) {
        int64_t center_x = k[0] / 2 + i * stride.first;
        for (int64_t j = 0; j < w_out; ++j) {
            int64_t center_y = k[1] / 2 + j * stride.second;
            auto submatrix = matrix_pad.index({torch::indexing::Slice(center_x - k[0] / 2, center_x + k[0] / 2 + 1),
                                               torch::indexing::Slice(center_y - k[1] / 2, center_y + k[1] / 2 + 1)});
            matrix_out[i][j] = (submatrix * kernel).sum();
        }
    }

    return matrix_out;
}


torch::Tensor apply_filter_to_image(torch::Tensor image, torch::Tensor kernel) {
    int64_t channels = image.size(2);
    std::vector<torch::Tensor> output_channels;

    for (int64_t z = 0; z < channels; ++z) {
        auto filtered_channel = conv2d(image.index({torch::indexing::Slice(), torch::indexing::Slice(), z}),
                                       kernel, {1, 1}, {1, 1}, {kernel.size(0) / 2, kernel.size(1) / 2});
        output_channels.push_back(filtered_channel);
    }

    return torch::stack(output_channels, 2);
}


int main() {
    // Example usage
    torch::Tensor image = torch::rand({224, 224, 3});  // Example image tensor (HWC format)
    torch::Tensor kernel = torch::rand({3, 3});        // Example kernel tensor (3x3)

    auto filtered_image = apply_filter_to_image(image, kernel);

    // Display the shape of the filtered image tensor
    std::cout << "Filtered image shape: " << filtered_image.sizes() << std::endl;

    return 0;
}