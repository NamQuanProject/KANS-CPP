#pragma once

#include <SFML/Graphics.hpp>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include "dataset.h"
#include "KAN.h"
#include "KANsLinear.h"
#include "train.h"
#include <memory>
#include <iostream>
#include <sstream>
#include "matplotlibcpp.h"
#include <algorithm> 

class App {
public:
    App();
    // UI OPERATIONS:
    void run();
    void processEvents();
    void render();

    // PAGE DRAWING:
    void drawMainMenu();
    void drawTrainingPage();
    void drawTestPage();
    void drawFunctionPage();

    // UI FUNCTIONS:
    void updateModelStructure();
    void trainModel();
    void testModel();
    void resetParameters();

    // EVENTS OPERATIONS:
    void handleButtonClick(const sf::Vector2i& mousePosition);
    void handleTextInput(const sf::Event& event);



    // PAGE SETUPS
    enum class Page {MainMenu, Training, Testing, FunctionTrain};
    Page currentPage;




    
    
    // TENSOR AND IMAGE OPERATIONS FOR VISULIZE
    sf::Image tensorToSFMLImage(const torch::Tensor& tensor, int scaleFactor);
    sf::RenderWindow window;
    sf::Font font;
    sf::Text title;
    sf::Text startTrainingText;
    sf::Text startTestingText;
    sf::Text functionTestingText;

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
    sf::RectangleShape functionTestingButton;
    
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
    
    sf::Text numTestImage;

    sf::String userNumImageInput;
    sf::RectangleShape numImageInputBox;
    sf::Text numImageInputText;
    
    sf::RectangleShape returnConfigButton;
    sf::Text returnConfigButtonText;

    sf::Text imageNumberText;
    sf::RectangleShape testingFrame;

    sf::RectangleShape testButton;
    sf::Text testButtonText;

    // FUNCTION PAGE:
    sf::Text functionTestTitle;
    sf::Text functionText;
    sf::RectangleShape changeFuntionButton;
    sf::Text changeFuntionButtonText;
    int functionIndex;
    double x;
    double y;

    std::vector<std::string> functions;
    std::vector<std::vector<int64_t>> functionsStructure;
    sf::Text functionStructureText;
    sf::Text functionMenuText;

    sf::Sprite modelStructureSprite;
    sf::Texture modelStructureTexture;
    
    sf::RectangleShape trainFunctionButton;
    sf::Text trainFunctionButtonText;

    sf::RectangleShape returnMainMenuButton;
    sf::Text returnMainMenuButtonText;
    sf::String xInput;
    sf::String yInput;
    sf::Text xText;
    sf::Text yText;
    sf::RectangleShape xBox;
    sf::RectangleShape yBox;
    int whichInput = -1;
    sf::Text textFrame;
    sf::Text xTitle;
    sf::Text yTitle;
    sf::Text testFunctionButtonText;
    sf::RectangleShape testFunctionButton;
    sf::Text functionPredictionText;
    sf::Text functionRealResultText;
    void predictFunction(float x, float y);




    std::vector<std::pair<std::vector<float>, std::vector<float>>> get_positions(const std::vector<int>& structure, const float initial_spacing, const float spacing_increment);
    void draw_square(float x, float y, float size, const std::vector<float>& vector);
    void trainFunction();
    std::vector<std::vector<std::vector<float>>> getModelGrid(KAN& kans);
    void plotModel(const std::vector<int>& structure, KAN& kan, const std::string& filename);
    void createAVIVideoFromImages(const std::string& outputVideoPath, int numFrames, double fps);
    void createMP4VideoFromImages(const std::string& outputVideoPath, int numFrames, double fps);
    std::vector<float> tensorToVector(const torch::Tensor& tensor);
    std::vector<std::vector<float>> batchToVectors(const torch::Tensor& batch);
    

    float convertTextToFloat(const sf::Text& text);

    
    void saveImageToFolder(const torch::Tensor& tensor, const std::string& filename);
    torch::Tensor loadImageFromFolder(const std::string& filename);
    std::string handleFilePath(sf::String number);
    sf::Image matToSFImage(const cv::Mat& mat);


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
