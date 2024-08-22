#include "UI.h"
#include <algorithm> 
namespace plt = matplotlibcpp;


/* INITIALIZE THE UI SYSTEMS*/
App::App()
    : window(sf::VideoMode(1200, 800), "Kolmogorov-Arnold Model Board"),
      epochs(5),
      currentEpoch(0),
      currentLoss(0.0),
      modelStructure({784, 64, 10}),
      device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
      dataPath("../data/train.csv")  {
    
    font.loadFromFile("../arial/ARIALBD.TTF");

    if (!backgroundTexture.loadFromFile("../background.jpg")) {
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
    title.setString("KANs MNIST Training Hyperparamers Settings");
    title.setCharacterSize(30);
    title.setFillColor(sf::Color::White);
    title.setPosition(300.f, 50.f);


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
    startTrainingButton.setSize(sf::Vector2f(210, 50));
    startTrainingButton.setPosition(200.f, 700.f);

    
    startTrainingText.setFont(font);
    startTrainingText.setFillColor(sf::Color::Blue);
    startTrainingText.setString("Start MNIST Training");
    startTrainingText.setCharacterSize(20);

    sf::FloatRect trainingTextRect = startTrainingText.getLocalBounds();
    startTrainingText.setOrigin(trainingTextRect.left + trainingTextRect.width / 2.0f, trainingTextRect.top + trainingTextRect.height / 2.0f);
    startTrainingText.setPosition(
        startTrainingButton.getPosition().x + startTrainingButton.getSize().x / 2.0f,
        startTrainingButton.getPosition().y + startTrainingButton.getSize().y / 2.0f
    ); 


    startTestingButton.setSize(sf::Vector2f(200, 50));
    startTestingButton.setPosition(880.f, 700.f);

    startTestingText.setFont(font);
    startTestingText.setFillColor(sf::Color::Blue);
    startTestingText.setString("Start MNIST Testing");
    startTestingText.setCharacterSize(20);

    sf::FloatRect testingTextRect = startTestingText.getLocalBounds();
    startTestingText.setOrigin(testingTextRect.left + testingTextRect.width / 2.0f, testingTextRect.top + testingTextRect.height / 2.0f);
    startTestingText.setPosition(
        startTestingButton.getPosition().x + startTestingButton.getSize().x / 2.0f,
        startTestingButton.getPosition().y + startTestingButton.getSize().y / 2.0f
    );

    functionTestingButton.setSize(sf::Vector2f(300, 50));
    functionTestingButton.setPosition(480.f, 700.f);

    functionTestingText.setFont(font);
    functionTestingText.setFillColor(sf::Color::Blue);
    functionTestingText.setString("Complex Function Training");
    functionTestingText.setCharacterSize(20);

    sf::FloatRect functionTestingTextRect = functionTestingText.getLocalBounds();
    functionTestingText.setOrigin(functionTestingTextRect.left + functionTestingTextRect.width / 2.0f, functionTestingTextRect.top + functionTestingTextRect.height / 2.0f);
    functionTestingText.setPosition(
        functionTestingButton.getPosition().x + functionTestingButton.getSize().x / 2.0f,
        functionTestingButton.getPosition().y + functionTestingButton.getSize().y / 2.0f
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
    configBoardButton.setPosition(650.f, 700.f);


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
    



    


    /*--------------------------------------------TESTING PARTS--------------------------------------------*/
    testTitle.setFont(font);
    testTitle.setString("KANs Testing Board");
    testTitle.setCharacterSize(30);
    testTitle.setFillColor(sf::Color::White);
    testTitle.setPosition(450.f, 100.f);

    imageNumberText.setFont(font);
    imageNumberText.setCharacterSize(20);
    imageNumberText.setString("Checking Image Number: ... ");
    imageNumberText.setPosition(700.f, 250.f);

    predictionText.setFont(font); 
    predictionText.setCharacterSize(20); 
    predictionText.setFillColor(sf::Color::White); 
    predictionText.setPosition(180.f, 700.f);

    numTestImage.setFont(font); 
    numTestImage.setCharacterSize(20); 
    numTestImage.setFillColor(sf::Color::White); 
    numTestImage.setPosition(920.f, 300.f);
    numTestImage.setString("/4199");

    numImageInputBox.setSize(sf::Vector2f(200, 30));
    numImageInputBox.setPosition(700.f, 300.f);
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

    


    returnConfigButton.setSize(sf::Vector2f(200, 50));
    returnConfigButton.setPosition(800.f, 700.f);
    returnConfigButton.setFillColor(sf::Color::White);

    returnConfigButtonText.setFont(font);
    returnConfigButtonText.setString("Config Board");
    returnConfigButtonText.setCharacterSize(20);
    returnConfigButtonText.setFillColor(sf::Color::Black);

    

    sf::FloatRect returnConfigButtonTextRect = returnConfigButtonText.getLocalBounds();
    returnConfigButtonText.setOrigin(returnConfigButtonTextRect .left + returnConfigButtonTextRect.width / 2.0f, returnConfigButtonTextRect.top + returnConfigButtonTextRect.height / 2.0f);
    returnConfigButtonText.setPosition(
        returnConfigButton.getPosition().x + returnConfigButton.getSize().x / 2.0f,
        returnConfigButton.getPosition().y + returnConfigButton.getSize().y / 2.0f
    );


    testButton.setSize(sf::Vector2f(200, 35));
    testButton.setPosition(700.f, 350.f);
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

    
    

    testingFrame.setSize(sf::Vector2f(450.f, 530.f));
    testingFrame.setPosition(680.f, 230.f);
    testingFrame.setFillColor(sf::Color::Transparent);
    testingFrame.setOutlineThickness(5.f);
    testingFrame.setOutlineColor(sf::Color::White);



    // SETUP CURRENT PAGE:
    currentPage = Page::MainMenu;
    
    /*-------------------------------------------FUNCTION TESTING------------------------------------------*/
    /*------------------------------------VISUALIZE THE MODEL STRUCTURE------------------------------------*/
    functionIndex = 0;
    functions = {
        "f(x) = exp(sin(Pi * x) + y^2)",
        "f(x) = x^2 + y^2",
        "f(x) = sin(x) * cos(y)",
        "f(x) = tanh(x + y)",
    };
    functionsStructure = {
        {2, 5, 1},
        {2, 5, 1},
        {2, 5, 1},
        {2, 5, 1},
    };

    functionMenuText.setFont(font);
    functionMenuText.setCharacterSize(20);
    functionMenuText.setFillColor(sf::Color::White);
    functionMenuText.setString("Functions:");
    functionMenuText.setPosition(700.f, 250.f);

    functionTestTitle.setFont(font);
    functionTestTitle.setString("Complex Functions Test");
    functionTestTitle.setCharacterSize(30);
    functionTestTitle.setFillColor(sf::Color::White);
    functionTestTitle.setPosition(450.f, 50.f);
    
    functionText.setFont(font);
    functionText.setCharacterSize(20);
    functionText.setString(functions[functionIndex]);
    functionText.setPosition(700.f, 300.f);

    functionStructureText.setFont(font);
    functionStructureText.setCharacterSize(20);
    functionStructureText.setString(vectorToString(functionsStructure[functionIndex]));
    functionStructureText.setPosition(700.f, 400.f);


    changeFuntionButton.setSize(sf::Vector2f(200, 30));
    changeFuntionButton.setPosition(700.f, 350.f);
    changeFuntionButton.setFillColor(sf::Color::White);

    changeFuntionButtonText.setFont(font);
    changeFuntionButtonText.setString("Change function");
    changeFuntionButtonText.setCharacterSize(20);
    changeFuntionButtonText.setFillColor(sf::Color::Black);
    sf::FloatRect changeFuntionButtonRect = changeFuntionButtonText.getLocalBounds();
    changeFuntionButtonText.setOrigin(changeFuntionButtonRect.left + changeFuntionButtonRect.width / 2.0f, changeFuntionButtonRect.top + changeFuntionButtonRect.height / 2.0f);
    changeFuntionButtonText.setPosition(
        changeFuntionButton.getPosition().x + changeFuntionButton.getSize().x / 2.0f,
        changeFuntionButton.getPosition().y + changeFuntionButton.getSize().y / 2.0f
    );


    trainFunctionButton.setSize(sf::Vector2f(200, 50));
    trainFunctionButton.setPosition(350.f, 700.f);
    trainFunctionButton.setFillColor(sf::Color::White);
    
    trainFunctionButtonText.setFont(font);
    trainFunctionButtonText.setString("Train f(x)");
    trainFunctionButtonText.setCharacterSize(20);
    trainFunctionButtonText.setFillColor(sf::Color::Black);

    sf::FloatRect trainFunctionButtonRect = trainFunctionButtonText.getLocalBounds();

    trainFunctionButtonText.setOrigin(trainFunctionButtonRect.left + trainFunctionButtonRect.width / 2.0f, trainFunctionButtonRect.top + trainFunctionButtonRect.height / 2.0f);
    trainFunctionButtonText.setPosition(
        trainFunctionButton.getPosition().x + trainFunctionButton.getSize().x / 2.0f,
        trainFunctionButton.getPosition().y + trainFunctionButton.getSize().y / 2.0f
    );


    returnMainMenuButton.setSize(sf::Vector2f(200, 50));
    returnMainMenuButton.setPosition(100.f, 700.f);
    returnMainMenuButton.setFillColor(sf::Color::White);
    
    returnMainMenuButtonText.setFont(font);
    returnMainMenuButtonText.setString("Config Board");
    returnMainMenuButtonText.setCharacterSize(20);
    returnMainMenuButtonText.setFillColor(sf::Color::Black);

    sf::FloatRect returnMainMenuButtonRect = returnMainMenuButtonText.getLocalBounds();

    returnMainMenuButtonText.setOrigin(returnMainMenuButtonRect.left + returnMainMenuButtonRect.width / 2.0f, returnMainMenuButtonRect.top + returnMainMenuButtonRect.height / 2.0f);
    returnMainMenuButtonText.setPosition(
        returnMainMenuButton.getPosition().x + returnMainMenuButton.getSize().x / 2.0f,
        returnMainMenuButton.getPosition().y + returnMainMenuButton.getSize().y / 2.0f
    );


    xTitle.setFont(font);
    xTitle.setString("X:");
    xTitle.setCharacterSize(30);
    xTitle.setFillColor(sf::Color::White);
    xTitle.setPosition(700.f, 450.f);

    xBox.setSize(sf::Vector2f(100, 50));
    xBox.setPosition(750.f, 450.f);
    returnMainMenuButton.setFillColor(sf::Color::White);
    
    xText.setFont(font);
    xText.setString("Input x");
    xText.setCharacterSize(20);
    xText.setFillColor(sf::Color::Black);

    sf::FloatRect xTextRect = xText.getLocalBounds();

    xText.setOrigin(xTextRect.left + xTextRect.width / 2.0f, xTextRect.top + xTextRect.height / 2.0f);
    xText.setPosition(
        xBox.getPosition().x + xBox.getSize().x / 2.0f,
        xBox.getPosition().y + xBox.getSize().y / 2.0f
    );


    yTitle.setFont(font);
    yTitle.setString("Y: ");
    yTitle.setCharacterSize(30);
    yTitle.setFillColor(sf::Color::White);
    yTitle.setPosition(700.f, 550.f);

    yBox.setSize(sf::Vector2f(100, 50));
    yBox.setPosition(750.f, 550.f);
    returnMainMenuButton.setFillColor(sf::Color::White);
    
    yText.setFont(font);
    yText.setString("Input y");
    yText.setCharacterSize(20);
    yText.setFillColor(sf::Color::Black);

    sf::FloatRect yTextRect = xText.getLocalBounds();

    yText.setOrigin(yTextRect.left + yTextRect.width / 2.0f, yTextRect.top + yTextRect.height / 2.0f);
    yText.setPosition(
        yBox.getPosition().x + yBox.getSize().x / 2.0f,
        yBox.getPosition().y + yBox.getSize().y / 2.0f
    );


    testFunctionButton.setSize(sf::Vector2f(120, 30));
    testFunctionButton.setPosition(700.f, 650.f);
    testFunctionButton.setFillColor(sf::Color::White);
    
    testFunctionButtonText.setFont(font);
    testFunctionButtonText.setString("Test");
    testFunctionButtonText.setCharacterSize(20);
    testFunctionButtonText.setFillColor(sf::Color::Black);

    sf::FloatRect testFunctionButtonRect = testFunctionButtonText.getLocalBounds();

    testFunctionButtonText.setOrigin(testFunctionButtonRect.left + testFunctionButtonRect.width / 2.0f, testFunctionButtonRect.top + testFunctionButtonRect.height / 2.0f);
    testFunctionButtonText.setPosition(
        testFunctionButton.getPosition().x + testFunctionButton.getSize().x / 2.0f,
        testFunctionButton.getPosition().y + testFunctionButton.getSize().y / 2.0f
    );


    functionPredictionText.setFont(font);
    functionPredictionText.setString("Prediction:");
    functionPredictionText.setCharacterSize(20);
    functionPredictionText.setFillColor(sf::Color::White);
    functionPredictionText.setPosition(870.f, 450.f);
    
    
    functionRealResultText.setFont(font);
    functionRealResultText.setString("Real Result:");
    functionRealResultText.setCharacterSize(20);
    functionRealResultText.setFillColor(sf::Color::White);
    functionRealResultText.setPosition(870.f, 550.f);


}



// SFML EVENTS SET UP AND RENDERING
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
        case Page::FunctionTrain:
            drawFunctionPage();
            break;
        
    }
    window.display();
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
    } else if (configBoardButton.getGlobalBounds().contains(mousePosition.x, mousePosition.y) && currentPage == Page::Testing) {
        resetParameters();
        currentPage = Page::MainMenu;
    
    } else if (returnConfigButton.getGlobalBounds().contains(mousePosition.x, mousePosition.y) && currentPage == Page::Testing){
        currentPage = Page::MainMenu;

    } else if (startTrainingButton.getGlobalBounds().contains(mousePosition.x, mousePosition.y) && currentPage == Page::MainMenu) {
        if (currentPage == Page::MainMenu) {
            currentPage = Page::Training;
            trainModel();
        }
    } else if (startTestingButton.getGlobalBounds().contains(mousePosition.x, mousePosition.y) && (currentPage == Page::Training || currentPage == Page::MainMenu)) {
        currentPage = Page::Testing;
    }
    else if (returnMainMenuButton.getGlobalBounds().contains(mousePosition.x, mousePosition.y) && currentPage == Page::FunctionTrain) {
        currentPage = Page::MainMenu;
    }
    else if (functionTestingButton.getGlobalBounds().contains(mousePosition.x, mousePosition.y) && currentPage == Page::MainMenu) {
        currentPage = Page::FunctionTrain;
    }
    else if (trainFunctionButton.getGlobalBounds().contains(mousePosition.x, mousePosition.y) && currentPage == Page::FunctionTrain) {
        trainFunction();
    }
    else if (numImageInputBox.getGlobalBounds().contains(mousePosition.x, mousePosition.y) && currentPage == Page::Testing) {
        if (numImageInputText.getString() == "Image num: ") numImageInputText.setString("|");
        
    }
    else if (xBox.getGlobalBounds().contains(mousePosition.x, mousePosition.y) && currentPage == Page::FunctionTrain) {
        whichInput = 0;
        xText.setString("|");
        xInput.clear();
    }
    else if (yBox.getGlobalBounds().contains(mousePosition.x, mousePosition.y) && currentPage == Page::FunctionTrain) {
        whichInput = 1;
        yText.setString("|");
        yInput.clear();
    }


    else if (testButton.getGlobalBounds().contains(mousePosition.x, mousePosition.y) && currentPage == Page::Testing) {
        if (userNumImageInput.toAnsiString() != "") {
            testModel();
            userNumImageInput.clear();
            numImageInputText.setString("Image num: ");
        }
        
    }
    else if (testFunctionButton.getGlobalBounds().contains(mousePosition.x, mousePosition.y) && currentPage == Page::FunctionTrain) {
        float x = convertTextToFloat(xText);
        float y = convertTextToFloat(yText);
        predictFunction(x,y);
    }
    else if (changeFuntionButton.getGlobalBounds().contains(mousePosition.x, mousePosition.y) && currentPage == Page::FunctionTrain) {
        functionIndex++;
        if (functionIndex >= functions.size()) {
            functionIndex = 0;
        }
        functionText.setString(functions[functionIndex]);
        functionStructureText.setString(vectorToString(functionsStructure[functionIndex]));
    }

    else {
        if (currentPage == Page::Testing) {
            numImageInputText.setString("Image num: ");
            userNumImageInput.clear();
        }
        if (currentPage == Page::MainMenu) {
            structureInputText.setString("Structure Input");
            userStructureInput.clear();

        }
        if (currentPage == Page::FunctionTrain) {
            whichInput = -1;
            xText.setString("Input x");
            yText.setString("Input y");
            xInput.clear();
            yInput.clear();
        }
        
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
        else if (currentPage == Page::FunctionTrain) {
            if (whichInput == 0) {
                if (event.text.unicode == '\b' && xInput.getSize() > 0) { 
                    xInput.erase(xInput.getSize() - 1, 1);
                    xText.setString(xInput.toAnsiString());
                
                } else if (event.text.unicode >= 32 && event.text.unicode <= 126) { 
                    xInput += static_cast<char>(event.text.unicode);
                    xText.setString(xInput.toAnsiString());
                }
            }
            else if (whichInput == 1){
                if (event.text.unicode == '\b' && yInput.getSize() > 0) { 
                    yInput.erase(yInput.getSize() - 1, 1);
                    yText.setString(yInput.toAnsiString());
                
                } else if (event.text.unicode >= 32 && event.text.unicode <= 126) { 
                    yInput += static_cast<char>(event.text.unicode);
                    yText.setString(yInput.toAnsiString());
                }
            }
            
        }
        
    }
}

void App::run() {
    while (window.isOpen()) {
        processEvents();
        render();
    }
}


// PAGE DRAWING FUNCTIONS:
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

    window.draw(functionTestingButton);
    window.draw(functionTestingText);
    
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
    window.draw(numTestImage);

    window.draw(returnConfigButton);
    window.draw(returnConfigButtonText);

    window.draw(testingFrame);
    window.draw(imageNumberText);

    window.draw(testImageSprite);

    
}

void App::drawFunctionPage() {
    window.draw(functionText);
    window.draw(changeFuntionButton);
    window.draw(changeFuntionButtonText);
    window.draw(functionStructureText);
    window.draw(trainFunctionButton);
    window.draw(trainFunctionButtonText);
    window.draw(functionTestTitle);
    window.draw(functionMenuText);

    window.draw(modelStructureSprite);
    window.draw(testingFrame);
    window.draw(returnMainMenuButton);
    window.draw(returnMainMenuButtonText);
    window.draw(yBox);
    window.draw(xBox);
    window.draw(xText);
    window.draw(yText);
    window.draw(xTitle);
    window.draw(yTitle);
    window.draw(testFunctionButton);
    window.draw(testFunctionButtonText);
    window.draw(functionPredictionText);
    window.draw(functionRealResultText);
}


// MAIN MENU FUNTIONS
void App::updateModelStructure() {
    modelStructure.clear(); 

    std::string inputString = userStructureInput.toAnsiString();

    std::istringstream iss(inputString);
    std::string token;
    while (std::getline(iss, token, ',')) {
        try {
            int value = std::stoi(token);
            modelStructure.push_back(value);
        } catch (const std::invalid_argument& e) {
            std::cerr << "Invalid input: " << token << std::endl;
        } catch (const std::out_of_range& e) {
            std::cerr << "Out of range input: " << token << std::endl;
        }
    }
    structureInputText.setString("Structure Input");
    vectorText.setString("Model Structure:  " + vectorToString(modelStructure));
    userStructureInput.clear();
}

void App::resetParameters() {
    epochs = 5;
    currentEpoch = 0;
    epochText.setString("Epochs:"  + std::to_string(epochs));
    currentEpochText.setString("Epochs: " + std::to_string(currentEpoch));
    vectorText.setString("Model Structure:  " + vectorToString(modelStructure));
    batchText.setString("Batch: 0");
    currentLoss = 0.0;
    currentLossText.setString("Current Loss: " + std::to_string(currentLoss));
    accuracyText.setString("Accuracy: 0.0%");
    dataProgressText.setString("Data process: 0%");
    trainingProgressIndicator.setSize(sf::Vector2f(0, 0));
    trainingProgressText.setString("Training Progress: 0%");
    
}


// TRAIN PAGE FUNCTIONS:

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
    cv::cvtColor(mat, rgbMat, cv::COLOR_GRAY2RGB);

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
        window.clear(); 
        render(); 

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

        // SAVING TEST IMAGES:
        // for (int i = 0; i < images.size(0); ++i) {
        //     std::string filename = "image_" + std::to_string(image_count++) + ".png";
        //     saveImageToFolder(images[i].cpu(), filename);
        // }
    }

    std::cout << "Test Loss: " << test_loss / test_batches.size() << std::endl;
    std::cout << "Test Accuracy: " << static_cast<double>(correct) / total * 100.0 << "%" << std::endl;
    accuracyText.setString("Accuracy: " + std::to_string(static_cast<double>(correct) / total * 100.0) + "%");
    torch::save(kan, "../model/KAN.pt");
    render();
}

// TEST PAGE FUNCTIONS
void App::testModel() {
    KAN kan(modelStructure);
    torch::load(kan, "../model/KAN.pt");
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
    std::string filepath = "../testImage/" + filename;
    if (!sfImage.saveToFile(filepath)) {
        std::cout << "Failed to save image to " << filepath << std::endl;
    } else {
        std::cout << "Image saved to " << filepath << std::endl;
    }
}

torch::Tensor App::loadImageFromFolder(const std::string& filename) {
    std::string filepath = "../testImage/" + filename;
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



// FUNCTION PAGE FUNCTIONS:
std::vector<std::pair<std::vector<float>, std::vector<float>>> App::get_positions(const std::vector<int>& structure, const float initial_spacing = 1.25, const float spacing_increment = 0.35) {
    std::vector<std::pair<std::vector<float>, std::vector<float>>> positions;
    std::vector<float> y_offsets;
    for (int i = 0; i < structure.size(); ++i) {
        y_offsets.push_back(1 - 2.0 * i / (structure.size() - 1));
    }
    
    for (int i = 0; i < structure.size(); ++i) {
        float spacing = (i % 2 == 0) ? (initial_spacing - i * spacing_increment + 1.0) : (initial_spacing - i * 2 * spacing_increment + 2.0);
        std::vector<float> x_positions;
        std::vector<float> y_positions(structure[i], y_offsets[i]);
        if (structure[i] == 1) {
            x_positions.push_back(0);
        } else {
            for (int j = 0; j < structure[i]; ++j) {
                x_positions.push_back(-spacing + j * 2 * spacing / (structure[i] - 1));
            }
        }
        positions.emplace_back(x_positions, y_positions);
    }
    
    return positions;
}

void App::draw_square(float x, float y, float size, const std::vector<float>& vector) {
    std::vector<double> square_x = {x - size / 2, x + size / 2, x + size / 2, x - size / 2, x - size / 2};
    std::vector<double> square_y = {y - size / 2, y - size / 2, y + size / 2, y + size / 2, y - size / 2};
    plt::plot(square_x, square_y, "k-");  

    
    if (!vector.empty()) {
        size_t num_points = vector.size();
        std::vector<double> x_positions(num_points);
        std::vector<double> y_positions(num_points);

        double max_abs_val = *std::max_element(vector.begin(), vector.end(), [](double a, double b) {
            return std::abs(a) < std::abs(b);
        });

        std::vector<double> normalized_vector(num_points);
        for (size_t i = 0; i < num_points; ++i) {
            normalized_vector[i] = vector[i] / max_abs_val;
        }

        for (size_t i = 0; i < num_points; ++i) {
            x_positions[i] = x - size / 2 + i * size / (num_points - 1);
            y_positions[i] = y + normalized_vector[i] * size / 2;
        }

        plt::plot(x_positions, y_positions, "b-");  
    }
}

std::vector<std::vector<std::vector<float>>> App::getModelGrid(KAN& kans) {
    /* (LAYERS, LAYERS NODES NUMBER , GRID SIZE AFTER CALCULATIONS) */
    std::vector<torch::Tensor> spline_vectors;
    for (auto layer : kans->layers) {
        int64_t size = layer->spline_weight.size(-1);
        torch::Tensor reshaped_tensor = layer->spline_weight.view({-1, size});
        spline_vectors.push_back(reshaped_tensor);
    }
    
    std::vector<std::vector<std::vector<float>>> batch_vectors;
    for (auto grid_vector: spline_vectors) {
        std::vector<std::vector<float>> batch_vector = batchToVectors(grid_vector);
        batch_vectors.push_back(batch_vector);
    }
    
    return batch_vectors;
}

void App::trainFunction() {
    std::vector<int> structure = {1, 5, 5, 10, 2};
    std::vector<int64_t> model_structure = {2 , 5 ,1};

    KAN kan(model_structure);
    kan->to(device);
    auto optimizer = torch::optim::LBFGS(kan->parameters(), torch::optim::LBFGSOptions(1).max_iter(100));
    int64_t epoch = 20;
    auto input = torch::rand({1024, 2});
    kan->train();
    for (int i = 0; i < epoch; ++i) {
        auto closure = [&]() -> torch::Tensor {
            optimizer.zero_grad();
            torch::Tensor output = kan->forward(input);
            torch::Tensor reg_loss = kan->regularization_loss(1, 0);

            auto x1 = input.index({torch::indexing::Slice(), 0});
            auto x2 = input.index({torch::indexing::Slice(), 1});

            const double pi = M_PI; 
            torch::Tensor target;

            if (functionIndex == 0) {
                target = torch::exp(torch::sin(pi * x1) + x2 * x2);
            } else if (functionIndex == 1) {
                target = x1 * x1 + x2 * x2;
            } else if (functionIndex == 2) {
                target = torch::sin(x1) * torch::cos(x2);
            } else if (functionIndex == 3) {
                target = torch::tanh(x1 + x2);
            } else {
                std::cerr << "Invalid function index" << std::endl;
                return torch::Tensor(); // Return an empty tensor to avoid errors
            }
            

            torch::Tensor loss = torch::nn::functional::mse_loss(output.squeeze(-1), target);

            torch::Tensor total_loss = loss + 1e-5 * reg_loss;

            total_loss.backward();
            return total_loss;
        };
        optimizer.step(closure);

        std::string filename = "../build/trainImages/epoch_" + std::to_string(i + 1) + ".png";
        plotModel(structure, kan, filename);

        if (!modelStructureTexture.loadFromFile(filename)) {
            std::cerr << "Error loading image: " << filename << std::endl;
            continue;
        }
        
        modelStructureSprite.setTexture(modelStructureTexture);
        float desiredSize = 550.0f;

        float scaleFactorX = desiredSize / modelStructureTexture.getSize().x;
        float scaleFactorY = desiredSize / modelStructureTexture.getSize().y;
        float scaleFactor = std::min(scaleFactorX, scaleFactorY);

        modelStructureSprite.setScale(scaleFactor, scaleFactor);
        modelStructureSprite.setPosition(50.f, 200.f);
        render();
    }
    kan->eval();
    auto new_input = torch::rand({5, 2});
    std::cout << "Input: " << new_input << std::endl;
    std::cout << "Output: " << kan->forward(new_input) << std::endl;

    int numFrames = epoch; 
    double fps = 20.0; 
    std::string outputVideoPath = "../build/training_visualization.avi";
    createAVIVideoFromImages(outputVideoPath, numFrames, fps);

    torch::save(kan, "../model/KANFUNCTION.pt");
}

void App::predictFunction(float x, float y) {
    std::vector<int64_t> model_structure = {2, 5, 1}; 

    KAN kan(model_structure);
    torch::load(kan, "../model/KANFUNCTION.pt");
    kan->to(device); 
    kan->eval();
    
    torch::Tensor input = torch::tensor({{x, y}}, torch::kFloat).to(device); 

    torch::Tensor output = kan->forward(input);

    torch::Tensor computed_result;
    const float pi = static_cast<float>(M_PI);
    torch::Tensor x_tensor = torch::tensor({x}, torch::kFloat).to(device);
    torch::Tensor y_tensor = torch::tensor({y}, torch::kFloat).to(device);

    switch (functionIndex) {
        case 0:
            computed_result = torch::exp(torch::sin(pi * x_tensor) + y_tensor * y_tensor);
            break;
        case 1:
            computed_result = x_tensor * x_tensor + y_tensor * y_tensor;
            break;
        case 2:
            computed_result = torch::sin(x_tensor) * torch::cos(y_tensor);
            break;
        case 3:
            computed_result = torch::tanh(x_tensor + y_tensor);
            break;
        default:
            std::cerr << "Invalid function index." << std::endl;
            return;
    }

    // Update the SFML texts with the results
    functionPredictionText.setString("Prediction: " + std::to_string(output.item<float>()));
    functionRealResultText.setString("Real Result: " + std::to_string(computed_result.item<float>()));
}

void App::plotModel(const std::vector<int>& structure, KAN& kan, const std::string& filename) {
    auto positions = get_positions(structure);
    auto grids = getModelGrid(kan);
    int layer = 0;
    for (size_t layer_idx = 0; layer_idx < positions.size(); ++layer_idx) {
        auto& [x_pos, y_pos] = positions[layer_idx];
        if (layer_idx % 2 == 0) {
            plt::plot(x_pos, y_pos, "ko");  
        } else {
            std::vector<float> example_vectors = {1, 5, 6 ,2 , 3 ,4 , 7, 8};
            
            for (size_t i = 0; i < x_pos.size(); ++i) {
                draw_square(x_pos[i], y_pos[i], 0.15, grids[1-layer][i]);
            }
            layer++;
        }
    }
    plt::xlim(-2.5, 2.5);
    plt::ylim(-2.5, 2.5);
    plt::axis("off");  
    plt::axis("equal");
    plt::save(filename);
    plt::clf(); 
}

std::vector<std::vector<float>> App::batchToVectors(const torch::Tensor& batch) {
    std::vector<std::vector<float>> batch_vectors;
    for (int i = 0; i < batch.size(0); ++i) {
        batch_vectors.push_back(tensorToVector(batch[i]));
    }
    return batch_vectors;
}

std::vector<float> App::tensorToVector(const torch::Tensor& tensor) {
    auto cpu_tensor = tensor.to(torch::kCPU).to(torch::kFloat32).contiguous();
    std::vector<float> vec(cpu_tensor.data_ptr<float>(), cpu_tensor.data_ptr<float>() + cpu_tensor.numel());
    return vec;
}

void App::createAVIVideoFromImages(const std::string& outputVideoPath, int numFrames, double fps) {
    std::vector<cv::Mat> frames;
    
    for (int i = 0; i < numFrames; ++i) {
        std::string filename = "../build/trainImages/epoch_" + std::to_string(i + 1) + ".png";
        cv::Mat frame = cv::imread(filename);
        if (frame.empty()) {
            std::cerr << "Could not read image: " << filename << std::endl;
            continue;
        }
        frames.push_back(frame);
    }

    if (frames.empty()) {
        std::cerr << "No frames were loaded." << std::endl;
        return;
    }

    cv::Size frameSize(frames[0].cols, frames[0].rows);

    cv::VideoWriter videoWriter(outputVideoPath, cv::VideoWriter::fourcc('M','J','P','G'), fps, frameSize);

    for (const auto& frame : frames) {
        videoWriter.write(frame);
    }

    videoWriter.release();
    std::cout << "Video saved as " << outputVideoPath << std::endl;
}

void App::createMP4VideoFromImages(const std::string& outputVideoPath, int numFrames, double fps) {
    std::vector<cv::Mat> frames;
    
    for (int i = 0; i < numFrames; ++i) {
        std::string filename = "../build/trainImages/epoch_" + std::to_string(i + 1) + ".png";
        cv::Mat frame = cv::imread(filename);
        if (frame.empty()) {
            std::cerr << "Could not read image: " << filename << std::endl;
            continue;
        }
        frames.push_back(frame);
    }

    if (frames.empty()) {
        std::cerr << "No frames were loaded." << std::endl;
        return;
    }

    cv::Size frameSize(frames[0].cols, frames[0].rows);

    cv::VideoWriter videoWriter(outputVideoPath, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, frameSize);

    if (!videoWriter.isOpened()) {
        std::cerr << "Could not open the output video file for write." << std::endl;
        return;
    }

    for (const auto& frame : frames) {
        videoWriter.write(frame);
    }

    videoWriter.release();
    std::cout << "Video saved as " << outputVideoPath << std::endl;
}


float App::convertTextToFloat(const sf::Text& text) {

    std::string str = text.getString().toAnsiString();

    try {
        float value = std::stod(str);
        return value;
    } catch (const std::invalid_argument& e) {
        std::cerr << "Invalid argument: " << e.what() << std::endl;
        return 0.0;
    } catch (const std::out_of_range& e) {
        std::cerr << "Out of range: " << e.what() << std::endl;
        return 0.0;
    }
}


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