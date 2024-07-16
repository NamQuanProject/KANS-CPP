#include "UI.h"


std::string App::vectorToString(std::vector<int64_t>& vec) {
    std::ostringstream oss;
    oss << "{";
    for (size_t i = 0; i < vec.size(); ++i) {
        oss << vec[i];
        if (i != vec.size() - 1) {
            oss << ", ";
        }
    }
    oss << "}";
    return oss.str();
}

App::App()
    : window(sf::VideoMode(1200, 800), "Deep Learning Model Board"),
      epochs(15),
      currentEpoch(0),
      currentLoss(0.0),
      modelStructure({784, 64, 10}),
      device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
      dataPath("../data/train.csv")  {

    font.loadFromFile("/Users/quannguyennam/Documents/Projects/KANS/arial/ARIALBD.TTF");

    if (!backgroundTexture.loadFromFile("/Users/quannguyennam/Documents/Projects/KANS/background.jpg")) {
        std::cerr << "Failed to load background image." << std::endl;
        return;
    }
    // BACKGROUND SET UP
    backgroundSprite.setTexture(backgroundTexture);
    sf::Vector2u windowSize = window.getSize();
    sf::Vector2u textureSize = backgroundTexture.getSize();
    backgroundSprite.setScale(static_cast<float>(windowSize.x) / textureSize.x, static_cast<float>(windowSize.y) / textureSize.y);
    
    // TITLE
    title.setFont(font);
    title.setString("Training Custom Board");
    title.setCharacterSize(30);
    title.setFillColor(sf::Color::White);
    title.setPosition(430.f, 100.f);

    
    // MAIN PAGE:
    // EPOCH ADJUSTMENTS:
    epochText.setCharacterSize(20);
    epochText.setFont(font);
    epochText.setString("Epochs: " + std::to_string(epochs));
    epochText.setPosition(100.f, 650.f);


    increaseEpochButton.setSize(sf::Vector2f(25, 25));
    increaseEpochButton.setPosition(110.f, 700.f);
    
    increaseEpochText.setFont(font);
    increaseEpochText.setString("+");
    increaseEpochText.setCharacterSize(24); 
    increaseEpochText.setFillColor(sf::Color::Black);
    
    sf::FloatRect increaseTextRect = increaseEpochText.getLocalBounds();
    increaseEpochText.setOrigin(increaseTextRect.left + increaseTextRect.width / 2.0f, increaseTextRect.top + increaseTextRect.height / 2.0f);
    increaseEpochText.setPosition(
        increaseEpochButton.getPosition().x + increaseEpochButton.getSize().x / 2.0f,
        increaseEpochButton.getPosition().y + increaseEpochButton.getSize().y / 2.0f
    );

    decreaseEpochButton.setSize(sf::Vector2f(25, 25));
    decreaseEpochButton.setPosition(170.f, 700.f);

    decreaseEpochText.setFont(font);
    decreaseEpochText.setString("-");
    decreaseEpochText.setCharacterSize(24); 
    decreaseEpochText.setFillColor(sf::Color::Black);
    
    sf::FloatRect decreaseTextRect = decreaseEpochText.getLocalBounds();
    decreaseEpochText.setOrigin(decreaseTextRect.left + decreaseTextRect.width / 2.0f, decreaseTextRect.top + decreaseTextRect.height / 2.0f);
    decreaseEpochText.setPosition(
        decreaseEpochButton.getPosition().x + decreaseEpochButton.getSize().x / 2.0f,
        decreaseEpochButton.getPosition().y + decreaseEpochButton.getSize().y / 2.0f
    );

    // STRUCTURE ADJUSTMENT
    structureInputBox.setSize(sf::Vector2f(200, 25));
    structureInputBox.setPosition(800.f, 700.f);
    structureInputBox.setFillColor(sf::Color::White);

    structureInputText.setFont(font);
    structureInputText.setString("Structure Input: ");
    structureInputText.setCharacterSize(20);
    structureInputText.setFillColor(sf::Color::Black);

    sf::FloatRect structureInputTextRect = structureInputText.getLocalBounds();
    structureInputText.setOrigin(structureInputTextRect.left + structureInputTextRect.width / 2.0f, structureInputTextRect.top + structureInputTextRect.height / 2.0f);

    structureInputText.setPosition(
        structureInputBox.getPosition().x + structureInputBox.getSize().x / 2.0f,
        structureInputBox.getPosition().y + structureInputBox.getSize().y / 2.0f
    );
    
    updateVectorButton.setSize(sf::Vector2f(100, 25));
    updateVectorButton.setPosition(1015.f, 700.f);

    updateVectorText.setFont(font);
    updateVectorText.setString("Update");
    updateVectorText.setFillColor(sf::Color::Black);
    updateVectorText.setCharacterSize(20);
    sf::FloatRect updateVectorRect = updateVectorText.getLocalBounds();
    updateVectorText.setOrigin(updateVectorRect.left + updateVectorRect.width / 2.0f, updateVectorRect.top + updateVectorRect.height / 2.0f);

    updateVectorText.setPosition(
        updateVectorButton.getPosition().x + updateVectorButton.getSize().x / 2.0f,
        updateVectorButton.getPosition().y + updateVectorButton.getSize().y / 2.0f
    );

    vectorText.setFont(font);
    vectorText.setString("Model Structure:  " + vectorToString(modelStructure));
    vectorText.setCharacterSize(20);
    vectorText.setPosition(815.f, 650.f);


    // START TRAINING BUTTONS AND TESTING BUTTONS
    startTrainingButton.setSize(sf::Vector2f(200, 50));
    startTrainingButton.setPosition(800, 250);


    startTrainingText.setFont(font);
    startTrainingText.setFillColor(sf::Color::Blue);
    startTrainingText.setString("Start Training: ");
    startTrainingText.setCharacterSize(20);

    sf::FloatRect trainingTextRect = startTrainingText.getLocalBounds();
    startTrainingText.setOrigin(trainingTextRect.left + trainingTextRect.width / 2.0f, trainingTextRect.top + trainingTextRect.height / 2.0f);
    startTrainingText.setPosition(
        startTrainingButton.getPosition().x + startTrainingButton.getSize().x / 2.0f,
        startTrainingButton.getPosition().y + startTrainingButton.getSize().y / 2.0f
    );

    


    startTestingButton.setSize(sf::Vector2f(200, 50));
    startTestingButton.setPosition(800, 350);

    startTestingText.setFont(font);
    startTestingText.setFillColor(sf::Color::Blue);
    startTestingText.setString("Start Testing");
    startTestingText.setCharacterSize(20);

    sf::FloatRect testingTextRect = startTestingText.getLocalBounds();
    startTestingText.setOrigin(testingTextRect.left + testingTextRect.width / 2.0f, testingTextRect.top + testingTextRect.height / 2.0f);
    startTestingText.setPosition(
        startTestingButton.getPosition().x + startTestingButton.getSize().x / 2.0f,
        startTestingButton.getPosition().y + startTestingButton.getSize().y / 2.0f
    );



    currentEpochText.setFont(font);
    currentEpochText.setString("Current Epoch: 0");
    currentEpochText.setPosition(100, 400);

    currentLossText.setFont(font);
    currentLossText.setString("Current Loss: 0.0");
    currentLossText.setPosition(100, 450);


    // TRAINING PAGE: 
    datasetText.setFont(font);
    datasetText.setString("Dataset: MNIST Num-Samples: 42000 Train: 90% Test: 10% ");
    datasetText.setCharacterSize(18);
    datasetText.setPosition(700.f, 50.f);

    
    currentEpochText.setFont(font);
    currentEpochText.setString("Epoch: 0");
    currentEpochText.setCharacterSize(18);
    currentEpochText.setPosition(700.f, 100.f);
    
    currentLossText.setFont(font);
    currentLossText.setString("Loss: 0.0");
    currentLossText.setCharacterSize(18);
    currentLossText.setPosition(700.f, 150.f);

    accuracyText.setFont(font);
    accuracyText.setString("Accuracy: 0.0%");
    accuracyText.setCharacterSize(18);
    accuracyText.setPosition(700.f, 200.f);
    
    batchText.setFont(font);
    batchText.setString("Batch: 0");
    batchText.setCharacterSize(18);
    batchText.setPosition(700.f, 250.f);




    progressBar.setSize(sf::Vector2f(400, 20));
    progressBar.setPosition(400.f, 600.f);
    progressBar.setFillColor(sf::Color::Green);
    progressBar.setOutlineThickness(1);
    progressBar.setOutlineColor(sf::Color::Black);

    trainingProgressIndicator.setSize(sf::Vector2f(0, 20));
    trainingProgressIndicator.setPosition(100.f, 500.f);
    trainingProgressIndicator.setFillColor(sf::Color::Blue);

    dataProgressIndicator.setSize(sf::Vector2f(0, 20));
    dataProgressIndicator.setPosition(100.f, 600.f);
    dataProgressIndicator.setFillColor(sf::Color::Red);

    currentPage = Page::MainMenu;
}

void App::run() {
    while (window.isOpen()) {
        processEvents();
        render();
    }
}

void App::handleTextInput(const sf::Event& event) {
    if (event.type == sf::Event::TextEntered) {
        if (event.text.unicode == '\b' && userStructureInput.getSize() > 0) { // Handle backspace
            userStructureInput.erase(userStructureInput.getSize() - 1, 1);
            structureInputText.setString(userStructureInput.toAnsiString());
        } else if (event.text.unicode >= 32 && event.text.unicode <= 126) { // Handle printable characters
            userStructureInput += static_cast<char>(event.text.unicode);
            structureInputText.setString(userStructureInput.toAnsiString());
        }
    }
}

void App::handleButtonClick(const sf::Vector2i& mousePosition) {
    if (increaseEpochButton.getGlobalBounds().contains(mousePosition.x, mousePosition.y)) {
        epochs++;
        epochText.setString("Epochs: " + std::to_string(epochs));
    } else if (decreaseEpochButton.getGlobalBounds().contains(mousePosition.x, mousePosition.y)) {
        epochs--;
        epochText.setString("Epochs: " + std::to_string(epochs));
    } else if (structureInputBox.getGlobalBounds().contains(mousePosition.x, mousePosition.y)) {
        structureInputText.setString("|");
    } else if (updateVectorButton.getGlobalBounds().contains(mousePosition.x, mousePosition.y)) {
        updateModelStructure();
    } else if (startTrainingButton.getGlobalBounds().contains(mousePosition.x, mousePosition.y)) {
        currentPage = Page::Training;
        trainModel();
    } else if (startTestingButton.getGlobalBounds().contains(mousePosition.x, mousePosition.y)) {
        currentPage = Page::Testing;
    }
}

void App::updateModelStructure() {
    modelStructure.clear(); 

    std::string inputString = userStructureInput.toAnsiString();

    // Tokenize the inputString by comma
    std::istringstream iss(inputString);
    std::string token;
    while (std::getline(iss, token, ',')) {
        // Convert token to int and add to modelStructure
        try {
            int value = std::stoi(token);
            modelStructure.push_back(value);
        } catch (const std::invalid_argument& e) {
            std::cerr << "Invalid input: " << token << std::endl;
        } catch (const std::out_of_range& e) {
            std::cerr << "Out of range input: " << token << std::endl;
        }
    }
    structureInputText.setString(vectorToString(modelStructure));
    vectorText.setString("Model Structure:  " + vectorToString(modelStructure));
    userStructureInput.clear();
}

void App::processEvents() {
    sf::Event event;
    while (window.pollEvent(event)) {
        if (event.type == sf::Event::Closed) {
            window.close();
        }
        else if (event.type == sf::Event::MouseButtonPressed) {
            handleButtonClick(sf::Mouse::getPosition(window));
        }
        else if (event.type == sf::Event::TextEntered) {
            handleTextInput(event);
        }
        
    }
}

void App::render() {
    window.clear();

    window.draw(backgroundSprite);
    switch (currentPage) {
        case Page::MainMenu:
            drawMainMenu();
            break;
        case Page::Training:
            drawTrainingPage();
            break;
        case Page::Testing:
            drawTestPage();
            break;
    }
    window.display();
}



void App::drawMainMenu() {
    // Draw background and UI elements
    window.draw(title);

    window.draw(epochText);
    window.draw(increaseEpochButton);
    window.draw(increaseEpochText);
    window.draw(decreaseEpochButton);
    window.draw(decreaseEpochText);


    window.draw(structureInputBox);
    window.draw(structureInputText);
    window.draw(updateVectorButton);
    window.draw(updateVectorText);
    window.draw(vectorText);


    window.draw(startTrainingButton);
    window.draw(startTrainingText);

    window.draw(startTestingButton);
    window.draw(startTestingText);
    
}

void App::drawTrainingPage() {
    window.draw(datasetText);
    window.draw(currentEpochText);
    window.draw(currentLossText);
    window.draw(accuracyText);
    window.draw(batchText);

    window.draw(imageSprite);

    


    window.draw(startTestingButton);
    window.draw(startTestingText);

    window.draw(dataProgressIndicator);
    window.draw(trainingProgressIndicator);
}

void App::drawTestPage() {
    window.draw(testResultText);
    window.draw(startTrainingButton);
    window.draw(startTestingButton);
}

sf::Image App::tensorToSFMLImage(const torch::Tensor& tensor) {
    int scale_factor = 10; // Adjust scale factor as needed
    auto img_tensor = tensor.cpu().detach();
    img_tensor = img_tensor.view({28, 28});

    cv::Mat img(cv::Size(28, 28), CV_32FC1, img_tensor.data_ptr<float>());
    img.convertTo(img, CV_8UC1, 255.0);

    cv::Mat scaled_img;
    cv::resize(img, scaled_img, cv::Size(), scale_factor, scale_factor, cv::INTER_NEAREST);

    sf::Image image;
    image.create(scaled_img.cols, scaled_img.rows, scaled_img.ptr());

    
    const sf::Uint8* pixels = image.getPixelsPtr();
    sf::Uint8* writablePixels = const_cast<sf::Uint8*>(pixels); 

    for (std::size_t i = 0; i < image.getSize().x * image.getSize().y; ++i) {
        writablePixels[i * 4 + 0] = scaled_img.at<uchar>(i); 
        writablePixels[i * 4 + 1] = scaled_img.at<uchar>(i); 
        writablePixels[i * 4 + 2] = scaled_img.at<uchar>(i); 
        writablePixels[i * 4 + 3] = 255; 
    }
    return image;
}



void App::trainModel() {
    render();
    float dataProgress = 1.0 / 3.0;
    dataProgressIndicator.setSize(sf::Vector2f(progressBar.getSize().x * dataProgress, progressBar.getSize().y));
    render();
    MNISTDataset dataset(dataPath);
    
    train_batches = createBatches(dataset.createDataset().first, 64);
    dataProgress = 2.0 / 3.0;
    dataProgressIndicator.setSize(sf::Vector2f(progressBar.getSize().x * dataProgress, progressBar.getSize().y));
    render();
    test_batches = createBatches(dataset.createDataset().second, 64);
    dataProgress = 3.0 / 3.0;
    dataProgressIndicator.setSize(sf::Vector2f(progressBar.getSize().x * dataProgress, progressBar.getSize().y));
    render();

    KAN kan(modelStructure);
    kan->to(device);
    int batch_size = 64;
    double learning_rate = 1e-3;
    double weight_decay = 1e-4;
    torch::optim::AdamW optimizer(kan->parameters(), torch::optim::AdamWOptions(learning_rate).weight_decay(weight_decay));
    torch::nn::CrossEntropyLoss criterion;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        kan->train();
        double epoch_loss = 0.0;
        int batchCount = 0;
        window.clear(); // Clear the window at the beginning of each epoch
        render(); // Render the cleared window

        for (const auto& batch : train_batches) {
            batchCount++;
            auto images = batch.first.to(device);
            auto labels = batch.second.to(device);

            auto outputs = kan->forward(images);
            auto loss = criterion(outputs, labels);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            epoch_loss += loss.item<double>();

            if (batchCount % 10 == 0 || batchCount == 591) {
                sf::Image sfImage = tensorToSFMLImage(images[0].cpu()); // Retrieve the first image
                if (!imageTexture.loadFromImage(sfImage)) {
                    std::cout << "Failed to load texture from image." << std::endl;
                    continue;
                }
                imageSprite.setTexture(imageTexture);
                imageSprite.setPosition(100.f, 100.f); // Adjust position as needed
                
                currentLossText.setString("Loss: " + std::to_string(epoch_loss / batchCount));
                batchText.setString("Batch: " + std::to_string(batchCount));

                render();
            }
        }
        currentEpochText.setString("Epochs" + std::to_string(epoch + 1));
        currentLossText.setString("Loss: " + std::to_string(epoch_loss / train_batches.size()));
        
        float progress = static_cast<float>(epoch + 1) / epochs;
        trainingProgressIndicator.setSize(sf::Vector2f(progressBar.getSize().x * progress, progressBar.getSize().y));
        render();
    }

    kan->eval();
    double test_loss = 0.0;
    int correct = 0;
    int total = 0;

    torch::NoGradGuard no_grad;
    for (const auto& batch : test_batches) {
        auto images = batch.first.to(device);
        auto labels = batch.second.to(device);

        auto outputs = kan->forward(images);
        auto loss = criterion(outputs, labels);

        test_loss += loss.item<double>();

        auto predicted = outputs.argmax(1);
        auto actual = labels.argmax(1);

        correct += predicted.eq(actual).sum().item<int>();
        total += labels.size(0);
    }

    std::cout << "Test Loss: " << test_loss / test_batches.size() << std::endl;
    std::cout << "Test Accuracy: " << static_cast<double>(correct) / total * 100.0 << "%" << std::endl;
    accuracyText.setString("Accuracy: " + std::to_string(static_cast<double>(correct) / total * 100.0) + "%");
    render();
}




