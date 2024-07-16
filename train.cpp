#include "train.h"

void showImage(const torch::Tensor& tensor, const std::string& window_name) {
    auto img_tensor = tensor.cpu();
    img_tensor = img_tensor.detach();
    img_tensor = img_tensor.squeeze();

    cv::Mat img(cv::Size(28, 28), CV_32FC1, img_tensor.data_ptr<float>());

    img.convertTo(img, CV_8UC1, 255.0);

    cv::imshow(window_name, img);
    cv::waitKey(0);
}

void trainMNIST() {
    torch::manual_seed(1);
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    
    // Hyperparameters 
    int input_size = 28 * 28;
    int hidden_size = 64;
    int num_classes = 10;
    int num_epochs = 15;
    int batch_size = 64;
    double learning_rate = 1e-3;
    double weight_decay = 1e-4;
    double gamma = 0.8;


    // Load data
    std::string data_path = "../data/train.csv";
    MNISTDataset dataset(data_path);
    auto [train_data, test_data] = dataset.createDataset();
    std::cout << "Train data size: " << train_data.size() << std::endl;
    std::cout << "Test data size: " << test_data.size() << std::endl;
    auto train_batches = createBatches(train_data, batch_size);
    auto test_batches = createBatches(test_data, batch_size);



    // Model Set up
    std::vector<int64_t> layers_hidden = {input_size, hidden_size, num_classes};
    KAN kan(layers_hidden);
    kan->to(device);

    torch::optim::AdamW optimizer(kan->parameters(), torch::optim::AdamWOptions(learning_rate).weight_decay(weight_decay));
    torch::nn::CrossEntropyLoss criterion;


    // Training loop
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        kan->train();

        double epoch_loss = 0.0;
        for (const auto& batch : train_batches) {
            auto images = batch.first.to(device);
            auto labels = batch.second.to(device);

            // Forward pass
            auto outputs = kan->forward(images);
            auto loss = criterion(outputs, labels);

            // Backward pass and optimization
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            epoch_loss += loss.item<double>();
        }

        std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs << "], Loss: " << epoch_loss / train_batches.size() << std::endl;
    }

    // Evalution steps
    kan->eval();
    double test_loss = 0.0;
    int correct = 0;
    int total = 0;

    torch::NoGradGuard no_grad;
    for (const auto& batch : test_batches) {
        auto images = batch.first.to(device);
        auto labels = batch.second.to(device);

        // Forward 
        auto outputs = kan->forward(images);
        auto loss = criterion(outputs, labels);

        test_loss += loss.item<double>();

        // Get predicted labels
        auto predicted = outputs.argmax(1);
        auto actual = labels.argmax(1);

        correct += predicted.eq(actual).sum().item<int>();
        total += labels.size(0);
    }

    std::cout << "Test Loss: " << test_loss / test_batches.size() << std::endl;
    std::cout << "Test Accuracy: " << static_cast<double>(correct) / total << std::endl;

    // Testing with one batch in test set:
    for (const auto& batch : test_batches) {
        auto images = batch.first.to(device);
        auto labels = batch.second.to(device);

        // Forward pass
        auto outputs = kan->forward(images);
        auto predicted = outputs.argmax(1);
        auto actual = labels.argmax(1);

        // Find the first correct prediction and visualize it
        for (int i = 0; i < batch_size; ++i) {
            if (predicted[i].item<int>() == actual[i].item<int>()) {
                std::cout << "Corrected: " << std::to_string(predicted[i].item<int>()) << std::endl;
                std::cout << "Prediction: Label " << std::to_string(predicted[i].item<int>()) << std::endl;
            }
        }
        return;
    }
}








