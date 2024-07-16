#pragma once

#include <SFML/Graphics.hpp>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include "dataset.h"
#include "KAN.h"
#include "KANsLinear.h"
#include "train.h"
#include <memory>
#include <iostream>
#include <sstream>

class App {
public:
    App();
    // UI OPERATIONS:
    void run();
    void processEvents();
    void render();
    void drawMainMenu();
    void drawTrainingPage();
    void drawTestPage();
    void updateModelStructure();
    void trainModel();
    void testModel();


    // EVENTS OPERATIONS:
    void handleButtonClick(const sf::Vector2i& mousePosition);
    void handleTextInput(const sf::Event& event);



    // PAGE SETUPS
    enum class Page {MainMenu, Training, Testing};
    Page currentPage;

    // TENSOR AND IMAGE OPERATIONS FOR VISULIZE
    sf::Image tensorToSFMLImage(const torch::Tensor& tensor);

    sf::RenderWindow window;
    sf::Font font;
    sf::Text title;
    
    sf::Text startTrainingText;
    sf::Text startTestingText;
    sf::Text testResultText;


    // EPOCHS ADJUSTMENTS:
    sf::Text epochText;
    sf::Text increaseEpochText;
    sf::Text decreaseEpochText;
    sf::RectangleShape increaseEpochButton;
    sf::RectangleShape decreaseEpochButton;

    
    
    sf::RectangleShape startTrainingButton;
    sf::RectangleShape startTestingButton;
    
    sf::Texture backgroundTexture;
    sf::Sprite backgroundSprite;


    // MODEL ADJUSTMENTS:
    sf::String userStructureInput;
    sf::RectangleShape structureInputBox;
    sf::RectangleShape updateVectorButton;
    sf::Text structureInputText;
    sf::Text updateVectorText;
    sf::Text vectorText;
    std::vector<int64_t> modelStructure;


    // TRAINING PROGRESS:
    sf::Text datasetText;
    sf::Text currentEpochText;
    sf::Text currentLossText;
    sf::Text accuracyText;
    sf::Text batchText;
    sf::RectangleShape progressBar;
    sf::RectangleShape trainingProgressIndicator;
    sf::RectangleShape dataProgressIndicator;
    sf::Sprite imageSprite;
    sf::Texture imageTexture;
    

    int epochs;
    int currentEpoch;
    int batchSize = 64;
    double currentLoss;
    std::string dataPath;
    torch::Device device;
    std::vector<std::pair<torch::Tensor, torch::Tensor>> train_batches;
    std::vector<std::pair<torch::Tensor, torch::Tensor>> test_batches;
    std::string vectorToString(std::vector<int64_t>& vec);
};
