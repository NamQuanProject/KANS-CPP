#pragma once
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <torch/torch.h>

class MNISTDataset {
public:
    int batch_size;
    std::string data_dir;
    int num_lines;

    explicit MNISTDataset(int batch_size) : batch_size(batch_size) {
        data_dir = "../data/train.csv";
        num_lines = countLines(data_dir) / 2; 
        std::cout << num_lines << std::endl;
    }

    std::tuple<torch::Tensor, torch::Tensor> loadData(int batch_num) {
        torch::Tensor batch_data = torch::zeros({batch_size, 1, 28, 28}, torch::kFloat32);
        torch::Tensor label = torch::zeros({batch_size}, torch::kInt64);

        std::ifstream file(data_dir);

        if (file.is_open()) {
            // Skip lines to the start of the desired batch
            for (int i = 0; i < batch_num * batch_size + 1; i++) {
                std::string line;
                std::getline(file, line);
            }

            int row = 0;
            while (row < batch_size) {
                std::string line;
                if (!std::getline(file, line))
                    // Handle error if file ends before all rows are read
                    break;

                std::stringstream lineStream(line);
                std::string cell;
                for (int col = 0; col < 785; ++col) {
                    if (!std::getline(lineStream, cell, ','))
                        // Handle error if row ends before all columns are read
                        break;
                    if (col == 0) {
                        label[row] = std::stoi(cell);
                    } else {
                        int pixel_index = col - 1;
                        int row_index = pixel_index / 28;
                        int col_index = pixel_index % 28;
                        batch_data[row][0][row_index][col_index] = std::stof(cell) / 255.0;
                    }
                }
                row++;
            }
        } else {
            std::cout << "Cannot open data file" << std::endl;
        }

        torch::Tensor one_hot_label = torch::zeros({batch_size, 10}, torch::kFloat32);
        for (int i = 0; i < batch_size; i++) {
            one_hot_label[i][label[i].item<int>()] = 1.0;
        }

        return std::make_tuple(batch_data, one_hot_label);
    }

    int countLines(const std::string& filename) {
        std::ifstream file(filename);
        if (!file) {
            std::cerr << "Failed to open the file: " << filename << std::endl;
            return 0;
        }

        int lineCount = 0;
        std::string line;
        while (std::getline(file, line)) {
            lineCount++;
        }

        file.close();

        return lineCount;
    }
};
