#include "UI.h"
namespace plt = matplotlibcpp;

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
    : window(sf::VideoMode(1200, 800), "Kolmogorov-Arnold Model Board"),
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
    
    


    // MAIN PAGE:
    // EPOCH ADJUSTMENTS:

    // TITLE
    title.setFont(font);
    title.setString("KANs Training Hyperparamers Settings");
    title.setCharacterSize(30);
    title.setFillColor(sf::Color::White);
    title.setPosition(360.f, 50.f);


    epochText.setCharacterSize(25);
    epochText.setFont(font);
    epochText.setString("Epochs: " + std::to_string(epochs));
    epochText.setPosition(450.f, 150.f);


    increaseEpochButton.setSize(sf::Vector2f(25, 25));
    increaseEpochButton.setPosition(450.f, 200.f);
    
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
    decreaseEpochButton.setPosition(500.f, 200.f);

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
    // Hyperparameter 
    learningRateText.setFont(font);
    learningRateText.setString("Learning Rate: " + std::to_string(learningRate));
    learningRateText.setCharacterSize(25); 
    learningRateText.setFillColor(sf::Color::White);
    learningRateText.setPosition(450.f, 450.f);

    weightDecayText.setFont(font);
    weightDecayText.setString("Weight Decay: " + std::to_string(weightDecay));
    weightDecayText.setCharacterSize(25); 
    weightDecayText.setFillColor(sf::Color::White);
    weightDecayText.setPosition(450.f, 530.f);
    


    /*--------------------------------------------STRUCTURES CHANGE--------------------------------------------*/
    structureInputBox.setSize(sf::Vector2f(200, 25));
    structureInputBox.setPosition(450.f, 350.f);
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
    updateVectorButton.setPosition(680.f, 350.f);

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
    vectorText.setCharacterSize(25);
    vectorText.setPosition(450.f, 300.f);

    hyperparametersBorder.setSize(sf::Vector2f(445.f, 500.f));
    hyperparametersBorder.setPosition(400.f, 130.f);
    hyperparametersBorder.setFillColor(sf::Color::Transparent);
    hyperparametersBorder.setOutlineThickness(5.f);
    hyperparametersBorder.setOutlineThickness(5.f);
    hyperparametersBorder.setOutlineColor(sf::Color::White);

    /*-----------------------------------------START TRAINING AND TEST BUTTONS--------------------------------------------*/
    startTrainingButton.setSize(sf::Vector2f(200, 50));
    startTrainingButton.setPosition(400.f, 700.f);

    
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
    startTestingButton.setPosition(650.f, 700.f);

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


    /*--------------------------------------------TRAINING PAGE--------------------------------------------*/
    datasetText.setFont(font);
    datasetText.setString("Dataset: MNIST Num-Samples: 42000\nTrain: 90% Test: 10% ");
    datasetText.setCharacterSize(20);
    datasetText.setPosition(700.f, 50.f);

    
    currentEpochText.setFont(font);
    currentEpochText.setString("Epoch: 0");
    currentEpochText.setCharacterSize(20);
    currentEpochText.setPosition(700.f, 120.f);
    
    currentLossText.setFont(font);
    currentLossText.setString("Loss: 0.0");
    currentLossText.setCharacterSize(20);
    currentLossText.setPosition(700.f, 170.f);



    trainingLabelText.setFont(font);
    trainingLabelText.setCharacterSize(20);
    trainingLabelText.setPosition(130.f, 480.f);
    trainingLabelText.setFillColor(sf::Color::White);

    accuracyText.setFont(font);
    accuracyText.setString("Accuracy: 0.0%");
    accuracyText.setCharacterSize(20);
    accuracyText.setPosition(700.f, 220.f);
    
    batchText.setFont(font);
    batchText.setString("Batch: 0");
    batchText.setCharacterSize(18);
    batchText.setPosition(700.f, 270.f);




    progressBar.setSize(sf::Vector2f(350, 20));
    progressBar.setPosition(400.f, 600.f);
    progressBar.setFillColor(sf::Color::Green);
    progressBar.setOutlineThickness(1);
    progressBar.setOutlineColor(sf::Color::Black);


    trainingProgressText.setFont(font);
    trainingProgressText.setCharacterSize(20);
    trainingProgressText.setFillColor(sf::Color::White);
    trainingProgressText.setPosition(700.f, 500.f);
    
    trainingProgressIndicator.setSize(sf::Vector2f(0, 20));
    trainingProgressIndicator.setPosition(700.f, 530.f);
    trainingProgressIndicator.setFillColor(sf::Color::Blue);

    dataProgressText.setFont(font);
    dataProgressText.setCharacterSize(20);
    dataProgressText.setFillColor(sf::Color::White);
    dataProgressText.setPosition(700.f, 420.f);


    dataProgressIndicator.setSize(sf::Vector2f(0, 20));
    dataProgressIndicator.setPosition(700.f, 450.f);
    dataProgressIndicator.setFillColor(sf::Color::Red);


    configBoardButton.setSize(sf::Vector2f(200, 50));
    configBoardButton.setPosition(400.f, 700.f);


    configBoardText.setFont(font);
    configBoardText.setFillColor(sf::Color::Blue);
    configBoardText.setString("Config Section:");
    configBoardText.setCharacterSize(20);

    sf::FloatRect configBoardRect = configBoardText.getLocalBounds();
    configBoardText.setOrigin(configBoardRect .left + configBoardRect.width / 2.0f, configBoardRect.top + configBoardRect .height / 2.0f);
    configBoardText.setPosition(
        configBoardButton.getPosition().x + configBoardButton.getSize().x / 2.0f,
        configBoardButton.getPosition().y + configBoardButton.getSize().y / 2.0f
    );

    
    
    metricInfoBorder.setSize(sf::Vector2f(400.f, 300.f));
    metricInfoBorder.setPosition(680.f, 40.f);
    metricInfoBorder.setFillColor(sf::Color::Transparent);
    metricInfoBorder.setOutlineThickness(5.f);
    metricInfoBorder.setOutlineThickness(5.f);
    metricInfoBorder.setOutlineColor(sf::Color::White);
    
    
    progressBarBorder.setSize(sf::Vector2f(400.f, 200.f));
    progressBarBorder.setPosition(680.f, 400.f);
    progressBarBorder.setFillColor(sf::Color::Transparent);
    progressBarBorder.setOutlineThickness(5.f);
    progressBarBorder.setOutlineThickness(5.f);
    progressBarBorder.setOutlineColor(sf::Color::White);
    



    /*-------------------------------------VISUALIZE THE MODEL STRUCTURE------------------------------------*/
    visualizeModelButton.setSize(sf::Vector2f(200.f, 100.f));
    visualizeModelButton.setPosition(100.f, 600.f);


    visualizeModelButtonText.setFont(font);
    visualizeModelButtonText.setFillColor(sf::Color::Blue);
    visualizeModelButtonText.setString("Model Visulization");
    visualizeModelButtonText.setCharacterSize(20);

    sf::FloatRect visualizeModelButtonRec = visualizeModelButtonText.getLocalBounds();
    visualizeModelButtonText.setOrigin(visualizeModelButtonRec.left + visualizeModelButtonRec.width / 2.0f, visualizeModelButtonRec.top + visualizeModelButtonRec.height / 2.0f);
    visualizeModelButtonText.setPosition(
        visualizeModelButton.getPosition().x + visualizeModelButton.getSize().x / 2.0f,
        visualizeModelButton.getPosition().y + visualizeModelButton.getSize().y / 2.0f
    );

    /*--------------------------------------------TESTING PARTS--------------------------------------------*/
    testTitle.setFont(font);
    testTitle.setString("KANs Testing Board");
    testTitle.setCharacterSize(30);
    testTitle.setFillColor(sf::Color::White);
    testTitle.setPosition(450.f, 100.f);

    predictionText.setFont(font); 
    predictionText.setCharacterSize(20); 
    predictionText.setFillColor(sf::Color::White); 
    predictionText.setPosition(180.f, 700.f);


    numImageInputBox.setSize(sf::Vector2f(200, 30));
    numImageInputBox.setPosition(700.f, 550.f);
    numImageInputBox.setFillColor(sf::Color::White);

    numImageInputText.setFont(font);
    numImageInputText.setString("Image num: ");
    numImageInputText.setCharacterSize(20);
    numImageInputText.setFillColor(sf::Color::Black);

    
    

    sf::FloatRect numImageTextRect = numImageInputText.getLocalBounds();
    numImageInputText.setOrigin(numImageTextRect.left + numImageTextRect.width / 2.0f, numImageTextRect.top + numImageTextRect.height / 2.0f);
    numImageInputText.setPosition(
        numImageInputBox.getPosition().x + numImageInputBox.getSize().x / 2.0f,
        numImageInputBox.getPosition().y + numImageInputBox.getSize().y / 2.0f
    );

    testButton.setSize(sf::Vector2f(400, 30));
    testButton.setPosition(700.f, 700.f);
    testButton.setFillColor(sf::Color::White);

    testButtonText.setFont(font);
    testButtonText.setString("TEST");
    testButtonText.setCharacterSize(20);
    testButtonText.setFillColor(sf::Color::Black);
    sf::FloatRect testButtonRect = testButtonText.getLocalBounds();
    testButtonText.setOrigin(testButtonRect.left + testButtonRect.width / 2.0f, testButtonRect.top + testButtonRect.height / 2.0f);
    testButtonText.setPosition(
        testButton.getPosition().x + testButton.getSize().x / 2.0f,
        testButton.getPosition().y + testButton.getSize().y / 2.0f
    );

    imageNumberText.setFont(font);
    imageNumberText.setCharacterSize(20);
    imageNumberText.setString("Checking Image Number: ...");
    imageNumberText.setPosition(700.f, 250.f);
    

    testingFrame.setSize(sf::Vector2f(450.f, 530.f));
    testingFrame.setPosition(680.f, 230.f);
    testingFrame.setFillColor(sf::Color::Transparent);
    testingFrame.setOutlineThickness(5.f);
    testingFrame.setOutlineColor(sf::Color::White);



    // SETUP CURRENT PAGE:
    currentPage = Page::MainMenu;
}

void App::resetParameters() {
    epochs = 15;
    currentEpoch = 0;
    epochText.setString("Epochs:"  + std::to_string(epochs));
    currentEpochText.setString("Epochs: " + std::to_string(currentEpoch));
    batchText.setString("Batch: 0");
    currentLoss = 0.0;
    currentLossText.setString("Current Loss: " + std::to_string(currentLoss));
    modelStructure = {784, 64, 10};
    accuracyText.setString("Accuracy: 0.0%");
    dataProgressText.setString("Data process: 0%");
    trainingProgressIndicator.setSize(sf::Vector2f(0, 0));
    trainingProgressText.setString("Training Progress: 0%");
    
}

void App::run() {
    while (window.isOpen()) {
        processEvents();
        render();
    }
}

void App::handleTextInput(const sf::Event& event) {
    if (event.type == sf::Event::TextEntered) {
        if (currentPage == Page::MainMenu) {
            if (event.text.unicode == '\b' && userStructureInput.getSize() > 0) { 
                userStructureInput.erase(userStructureInput.getSize() - 1, 1);
                structureInputText.setString(userStructureInput.toAnsiString());
            } else if (event.text.unicode >= 32 && event.text.unicode <= 126) { 
                userStructureInput += static_cast<char>(event.text.unicode);
                structureInputText.setString(userStructureInput.toAnsiString());
            }
        }
        else if (currentPage == Page::Testing) {
            if (event.text.unicode == '\b' && userNumImageInput.getSize() > 0) { 
                userNumImageInput.erase(userNumImageInput.getSize() - 1, 1);
                numImageInputText.setString(userNumImageInput.toAnsiString() + "|");
            
            } else if (event.text.unicode >= 32 && event.text.unicode <= 126) { 
                userNumImageInput += static_cast<char>(event.text.unicode);
                numImageInputText.setString(userNumImageInput.toAnsiString() + "|");
            }
        }
        
    }
}

void App::handleButtonClick(const sf::Vector2i& mousePosition) {
    if (increaseEpochButton.getGlobalBounds().contains(mousePosition.x, mousePosition.y) && currentPage == Page::MainMenu) {
        epochs++;
        epochText.setString("Epochs: " + std::to_string(epochs));
    } else if (decreaseEpochButton.getGlobalBounds().contains(mousePosition.x, mousePosition.y) && currentPage == Page::MainMenu) {
        epochs--;
        epochText.setString("Epochs: " + std::to_string(epochs));
    } else if (structureInputBox.getGlobalBounds().contains(mousePosition.x, mousePosition.y) && currentPage == Page::MainMenu) {
        structureInputText.setString("|");
    } else if (updateVectorButton.getGlobalBounds().contains(mousePosition.x, mousePosition.y) && currentPage == Page::MainMenu) {
        updateModelStructure();
    } else if (configBoardButton.getGlobalBounds().contains(mousePosition.x, mousePosition.y) && currentPage != Page::MainMenu) {
        resetParameters();
        currentPage = Page::MainMenu;
        
    } else if (startTrainingButton.getGlobalBounds().contains(mousePosition.x, mousePosition.y) && currentPage == Page::MainMenu) {
        if (currentPage == Page::MainMenu) {
            currentPage = Page::Training;
            trainModel();
        }
    } else if (startTestingButton.getGlobalBounds().contains(mousePosition.x, mousePosition.y)) {
        currentPage = Page::Testing;
    }
    else if (visualizeModelButton.getGlobalBounds().contains(mousePosition.x, mousePosition.y) && currentPage == Page::Training) {
        for (auto grid_vector: grid_vectors) {
            plotVectors(grid_vectors);
        }
        
    }  
    
    else if (numImageInputBox.getGlobalBounds().contains(mousePosition.x, mousePosition.y) && currentPage == Page::Testing) {
        if (numImageInputText.getString() == "Image num: ") numImageInputText.setString("|");
        
    }

    else if (testButton.getGlobalBounds().contains(mousePosition.x, mousePosition.y) && currentPage == Page::Testing) {
        if (userNumImageInput.toAnsiString() != "") {
            testModel();
            userNumImageInput.clear();
            numImageInputText.setString("Image num: ");
        }
        
    }

    else {
        if (currentPage == Page::Testing) {
            numImageInputText.setString("Image num: ");
            userNumImageInput.clear();
        }
        
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

    window.draw(weightDecayText);
    window.draw(learningRateText);

    window.draw(hyperparametersBorder);

    window.draw(startTrainingButton);
    window.draw(startTrainingText);

    window.draw(startTestingButton);
    window.draw(startTestingText);


    
}

void App::drawTrainingPage() {
    // UPPER PART:
    window.draw(datasetText);
    window.draw(currentEpochText);
    window.draw(currentLossText);
    window.draw(accuracyText);
    window.draw(batchText);



    // FIRST IMAGE OF EACH BATCH:
    window.draw(imageSprite);

    window.draw(visualizeModelButton);
    window.draw(visualizeModelButtonText);

    // MOVE TO TEST OR CONFIG SECTIONS:
    window.draw(startTestingButton);
    window.draw(startTestingText);

    window.draw(configBoardButton);
    window.draw(configBoardText);




    // PROGRESSING BAR:
    window.draw(dataProgressText);
    window.draw(dataProgressIndicator);

    window.draw(trainingProgressText);
    window.draw(trainingProgressIndicator);


    window.draw(metricInfoBorder);
    window.draw(progressBarBorder);
    window.draw(trainingLabelText);
}

void App::drawTestPage() {
    window.draw(testTitle);
    window.draw(numImageInputBox);
    window.draw(numImageInputText);
    window.draw(predictionText);
    window.draw(testButton);
    window.draw(testButtonText);

    window.draw(testingFrame);
    window.draw(imageNumberText);

    window.draw(testImageSprite);

}

sf::Image App::tensorToSFMLImage(const torch::Tensor& tensor, int scaleFactor) {
    auto img_tensor = tensor.cpu().detach();
    img_tensor = img_tensor.view({28, 28});

    cv::Mat img(cv::Size(28, 28), CV_32FC1, img_tensor.data_ptr<float>());
    img.convertTo(img, CV_8UC1, 255.0);

    cv::Mat scaled_img;
    cv::resize(img, scaled_img, cv::Size(), scaleFactor, scaleFactor, cv::INTER_NEAREST);

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

sf::Image App::matToSFImage(const cv::Mat& mat) {
    sf::Image image;
    cv::Mat rgbMat;
    cv::cvtColor(mat, rgbMat, cv::COLOR_GRAY2RGB); // Convert grayscale to RGB

    image.create(rgbMat.cols, rgbMat.rows, rgbMat.ptr());
    return image;
}


void App::trainModel() {
    render();
    float dataProgress = 1.0 / 3.0;
    dataProgressIndicator.setSize(sf::Vector2f(progressBar.getSize().x * dataProgress, progressBar.getSize().y));
    dataProgressText.setString("Data Processing: " + std::to_string(dataProgress * 100) + "%");
    render();
    MNISTDataset dataset(dataPath);
    
    train_batches = createBatches(dataset.createDataset().first, 64);
    dataProgress = 2.0 / 3.0;
    dataProgressIndicator.setSize(sf::Vector2f(progressBar.getSize().x * dataProgress, progressBar.getSize().y));
    dataProgressText.setString("Data Processing: " + std::to_string(dataProgress * 100) + "%");
    render();
    test_batches = createBatches(dataset.createDataset().second, 64);
    
    dataProgress = 3.0 / 3.0;
    dataProgressText.setString("Data Processing: " + std::to_string(dataProgress * 100) + "%");
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
                std::vector<torch::Tensor> layer_grids;
                sf::Image sfImage = tensorToSFMLImage(images[0].cpu(), 15); 
                if (!imageTexture.loadFromImage(sfImage)) {
                    std::cout << "Failed to load texture from image." << std::endl;
                    continue;
                }
                imageSprite.setTexture(imageTexture);
                imageSprite.setPosition(100.f, 50.f); 
                
                currentLossText.setString("Loss: " + std::to_string(epoch_loss / batchCount));
                batchText.setString("Batch: " + std::to_string(batchCount));
                trainingLabelText.setString("Predicted Label: " + std::to_string(outputs[0].argmax(0).item<int>()) + " Correct Label: " + std::to_string(labels[0].argmax(0).item<int>()));
                
                grid_vectors.clear();
                for (auto layer : kan->layers) {
                    grid_vectors.push_back(layer->grid);
                }
                render();
            }
        }
        currentEpochText.setString("Current Epochs: " + std::to_string(epoch + 1));
        currentLossText.setString("Loss: " + std::to_string(epoch_loss / train_batches.size()));
        
        float progress = static_cast<float>(epoch + 1) / epochs;
        trainingProgressIndicator.setSize(sf::Vector2f(progressBar.getSize().x * progress, progressBar.getSize().y));
        trainingProgressText.setString("Training Progress: " + std::to_string(progress * 100) + "%");
        render();
    }

    kan->eval();
    double test_loss = 0.0;
    int correct = 0;
    int total = 0;
    int image_count = 0;

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

        // for (int i = 0; i < images.size(0); ++i) {
        //     std::string filename = "image_" + std::to_string(image_count++) + ".png";
        //     saveImageToFolder(images[i].cpu(), filename);
        // }
    }

    std::cout << "Test Loss: " << test_loss / test_batches.size() << std::endl;
    std::cout << "Test Accuracy: " << static_cast<double>(correct) / total * 100.0 << "%" << std::endl;
    accuracyText.setString("Accuracy: " + std::to_string(static_cast<double>(correct) / total * 100.0) + "%");
    torch::save(kan, "/Users/quannguyennam/Documents/Projects/KANS/model/KAN.pt");
    render();
    
}




std::vector<float> App::tensorToVector(const torch::Tensor& tensor) {
    auto cpu_tensor = tensor.to(torch::kCPU).to(torch::kFloat32).contiguous();
    std::vector<float> vec(cpu_tensor.data_ptr<float>(), cpu_tensor.data_ptr<float>() + cpu_tensor.numel());
    return vec;
}

// Function to plot the vectors
void App::plotVectors(const std::vector<torch::Tensor>& grid_vectors) {
    int max_grids = 10;  
    int num_features = std::min(static_cast<int>(grid_vectors.size()), max_grids);
    int num_rows = static_cast<int>(std::ceil(std::sqrt(num_features)));
    int num_cols = num_rows;

    // Create a new figure
    plt::figure_size(800, 800);

    for (int i = 0; i < num_features; ++i) {
        auto tensor = grid_vectors[i].cpu();
        auto sizes = tensor.sizes();
        int size = sizes[0];

        // Convert tensor to std::vector<double>
        std::vector<double> data(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());

        // Create a subplot for each grid
        plt::subplot(num_rows, num_cols, i + 1);
        plt::plot(data, "o-");
        plt::plot(data, {{"markersize", "3"}, {"linewidth", "1"}});
        plt::title("Feature " + std::to_string(i + 1), {{"fontsize", "8"}});
        plt::ylim(*std::min_element(data.begin(), data.end()) - 0.5, *std::max_element(data.begin(), data.end()) + 0.5);
        plt::grid(true);

        // Setting tick params (only 1 argument for map)
        plt::tick_params({{"axis", "both"}, {"which", "major"}, {"labelsize", "6"}});
    }

    // Turn off unused subplots
    for (int j = num_features; j < num_rows * num_cols; ++j) {
        plt::subplot(num_rows, num_cols, j + 1);
        plt::axis("off");
    }

    plt::tight_layout();
    plt::show();
}

void App::testModel() {
    KAN kan(modelStructure);
    torch::load(kan, "/Users/quannguyennam/Documents/Projects/KANS/model/KAN.pt");
    kan->to(device);
    kan->eval();
    
    
    std::string filepath = handleFilePath(userNumImageInput);
    auto image_tensor = loadImageFromFolder(filepath);
    sf::Image testsfImage = tensorToSFMLImage(image_tensor[0], 15);

    testImageTexture.loadFromImage(testsfImage);
    testImageSprite.setTexture(testImageTexture);
    testImageSprite.setPosition(100.f, 200.f);
    

    auto outputs = kan->forward(image_tensor);
    auto prediction = outputs.argmax(1).item<int>();

    predictionText.setString("Prediction From Models: " + std::to_string(prediction));
    render();
}


std::string App::handleFilePath(sf::String number) {
    auto string = number.toAnsiString();
    std::string file_path = "image_" + string + ".png";
    return file_path;
}



void App::saveImageToFolder(const torch::Tensor& tensor, const std::string& filename) {
    sf::Image sfImage = tensorToSFMLImage(tensor[0].cpu(), 1);
    std::string filepath = "/Users/quannguyennam/Documents/Projects/KANS/testImage/" + filename;
    if (!sfImage.saveToFile(filepath)) {
        std::cout << "Failed to save image to " << filepath << std::endl;
    } else {
        std::cout << "Image saved to " << filepath << std::endl;
    }
}

torch::Tensor App::loadImageFromFolder(const std::string& filename) {
    std::string filepath = "/Users/quannguyennam/Documents/Projects/KANS/testImage/" + filename;
    cv::Mat cvImage = cv::imread(filepath, cv::IMREAD_GRAYSCALE); 
    if (cvImage.empty()) {
        std::cerr << "Failed to load image from " << filepath << std::endl;
        return torch::Tensor();
    }

    cv::resize(cvImage, cvImage, cv::Size(28, 28));

    cv::Mat cvImageFloat;
    cvImage.convertTo(cvImageFloat, CV_32F, 1.0 / 255.0);

    torch::Tensor tensor = torch::from_blob(cvImageFloat.data, {1, 28 * 28}, torch::kFloat32); 
    return tensor.clone(); 
}







