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
#include "matplotlibcpp.h"


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
    void resetParameters();

    // EVENTS OPERATIONS:
    void handleButtonClick(const sf::Vector2i& mousePosition);
    void handleTextInput(const sf::Event& event);



    // PAGE SETUPS
    enum class Page {MainMenu, Training, Testing};
    Page currentPage;

    // TENSOR AND IMAGE OPERATIONS FOR VISULIZE
    sf::Image tensorToSFMLImage(const torch::Tensor& tensor, int scaleFactor);
    sf::RenderWindow window;
    sf::Font font;
    sf::Text title;
    sf::Text startTrainingText;
    sf::Text startTestingText;
    sf::Text testResultText;
    sf::Text learningRateText;
    sf::Text weightDecayText;
    sf::RectangleShape hyperparametersBorder;


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

    sf::Text dataProgressText;
    sf::Text trainingProgressText;
    sf::RectangleShape progressBar;
    sf::RectangleShape trainingProgressIndicator;
    sf::RectangleShape dataProgressIndicator;
    
    sf::RectangleShape configBoardButton;
    sf::Text configBoardText;

    sf::Sprite imageSprite;
    sf::Texture imageTexture;
    sf::Text trainingLabelText;
    
    sf::RectangleShape metricInfoBorder;
    sf::RectangleShape progressBarBorder;


    sf::RectangleShape visualizeModelButton;
    sf::Text visualizeModelButtonText;
    // TESTING 
    sf::Text testTitle;
    sf::Text userNumImageText;
    sf::Text predictionText;
    sf::Sprite testImageSprite;
    sf::Texture testImageTexture;


    sf::String userNumImageInput;
    sf::RectangleShape numImageInputBox;
    sf::Text numImageInputText;
    
    sf::Text imageNumberText;
    sf::RectangleShape testingFrame;

    sf::RectangleShape testButton;
    sf::Text testButtonText;

    
    void saveImageToFolder(const torch::Tensor& tensor, const std::string& filename);
    torch::Tensor loadImageFromFolder(const std::string& filename);
    std::string handleFilePath(sf::String number);
    sf::Image matToSFImage(const cv::Mat& mat);
    std::vector<float> tensorToVector(const torch::Tensor& tensor);
    void plotVectors(const std::vector<torch::Tensor>& grid_vectors);


    int epochs;
    int currentEpoch;
    int batchSize = 64;
    double currentLoss;
    double learningRate = 1e-3;
    double weightDecay = 1e-4;
    std::string dataPath;
    torch::Device device;
    std::vector<std::pair<torch::Tensor, torch::Tensor>> train_batches;
    std::vector<std::pair<torch::Tensor, torch::Tensor>> test_batches;
    std::string vectorToString(std::vector<int64_t>& vec);
    std::vector<torch::Tensor> grid_vectors;
    


};
