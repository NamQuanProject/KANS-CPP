#pragma once
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>
#include <torch/torch.h>

class MNISTDataset : public torch::data::Dataset<MNISTDataset> {
public:
    explicit MNISTDataset(const std::string& data_dir) : data_dir(data_dir) {
        num_lines = countLines(data_dir);
        num_samples = num_lines - 1; 
        std::cout << "Number of samples in dataset: " <<  num_samples << std::endl;
        loadData();
    }

    torch::data::Example<> get(size_t index) override {
        return {images_[index], labels_[index]};
    }

    torch::optional<size_t> size() const override {
        return images_.size();
    }

    std::pair<torch::Tensor, torch::Tensor> getBatch(size_t index, size_t batch_size) {
        size_t end_index = std::min(index + batch_size, images_.size());
        auto batch_data = torch::stack(std::vector<torch::Tensor>(images_.begin() + index, images_.begin() + end_index));
        auto batch_labels = torch::stack(std::vector<torch::Tensor>(labels_.begin() + index, labels_.begin() + end_index));
        return std::make_pair(batch_data, batch_labels);
    }

    std::pair<std::vector<std::vector<torch::Tensor>>, std::vector<std::vector<torch::Tensor>>> createDataset() {
        std::vector<std::vector<torch::Tensor>> train_data;
        std::vector<std::vector<torch::Tensor>> test_data;

        size_t train_split = static_cast<size_t>(num_samples * 0.9);

        for (size_t i = 0; i < train_split; ++i) {
            train_data.push_back({images_[i], labels_[i]});
        }
        for (size_t i = train_split; i < num_samples; ++i) {
            test_data.push_back({images_[i], labels_[i]});
        }

        return {train_data, test_data};
    }

    std::string data_dir;
    std::vector<torch::Tensor> images_;
    std::vector<torch::Tensor> labels_;
    int num_lines;
    int num_samples;

    void loadData() {
        std::ifstream file(data_dir);
        if (!file.is_open()) {
            std::cerr << "Cannot open data file: " << data_dir << std::endl;
            return;
        }

        std::string line;
        std::getline(file, line);
        while (std::getline(file, line)) {
            std::stringstream lineStream(line);
            std::string cell;
            torch::Tensor image = torch::zeros({784}, torch::kFloat32);
            int64_t label;
            torch::Tensor one_hot_label = torch::zeros({10}, torch::kFloat32);

            for (int col = 0; col < 785; ++col) {
                std::getline(lineStream, cell, ',');
                if (col == 0) {
                    label = std::stoi(cell);
                    one_hot_label[label] = 1.0;
                } else {
                    int pixel_index = col - 1;
                    image[pixel_index] = std::stof(cell) / 255.0;
                }
            }
            images_.push_back(image);
            labels_.push_back(one_hot_label);
        }
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


std::vector<std::pair<torch::Tensor, torch::Tensor>> createBatches(
    const std::vector<std::vector<torch::Tensor>>& data, size_t batch_size) {

    
    
    std::vector<std::pair<torch::Tensor, torch::Tensor>> batches;
    size_t num_batches = data.size() / batch_size;

    for (size_t i = 0; i < num_batches; ++i) {
        std::vector<torch::Tensor> batch_images;
        std::vector<torch::Tensor> batch_labels;

        for (size_t j = 0; j < batch_size; ++j) {
            batch_images.push_back(data[i * batch_size + j][0]);
            batch_labels.push_back(data[i * batch_size + j][1]);
        }

        auto batch_data = torch::stack(batch_images);
        auto batch_label_data = torch::stack(batch_labels);
        batches.emplace_back(batch_data, batch_label_data);
    }

    // Handle the remaining data
    if (data.size() % batch_size != 0) {
        std::vector<torch::Tensor> batch_images;
        std::vector<torch::Tensor> batch_labels;

        for (size_t i = num_batches * batch_size; i < data.size(); ++i) {
            batch_images.push_back(data[i][0]);
            batch_labels.push_back(data[i][1]);
        }

        auto batch_data = torch::stack(batch_images);
        auto batch_label_data = torch::stack(batch_labels);
        batches.emplace_back(batch_data, batch_label_data);
    }

    return batches;
}