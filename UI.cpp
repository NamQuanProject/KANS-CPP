#include "UI.h"
void app() {
 // Create the SFML window
    sf::RenderWindow window(sf::VideoMode(1200, 800), "SFML Image and Buttons");

    // Load your image using SFML's Texture and Sprite
    sf::Texture texture;
    if (!texture.loadFromFile("/Users/quannguyennam/Documents/Projects/KANS/TestImage.png")) {
        std::cerr << "Failed to load image." << std::endl;
        return;
    }
    sf::Sprite sprite(texture);

    // Determine the size of the square frame
    float frameSize = 400.f;
    sf::RectangleShape frame(sf::Vector2f(frameSize, frameSize));
    frame.setFillColor(sf::Color::Transparent);
    frame.setOutlineColor(sf::Color::Black);
    frame.setOutlineThickness(2.f);
    frame.setPosition(50.f, 50.f); // Adjust position as needed

    // Resize the sprite to fit within the square frame
    sf::FloatRect spriteBounds = sprite.getLocalBounds();
    float scaleFactor = frameSize / std::max(spriteBounds.width, spriteBounds.height);
    sprite.setScale(scaleFactor, scaleFactor);
    sprite.setPosition(frame.getPosition() + sf::Vector2f((frameSize - spriteBounds.width * scaleFactor) / 2.f , (frameSize - spriteBounds.height * scaleFactor) / 2.f));

    // Create buttons at the bottom
    sf::RectangleShape button1(sf::Vector2f(200.0f, 50.0f));
    button1.setFillColor(sf::Color::Green);
    button1.setPosition(100.0f, 700.0f);

    sf::RectangleShape button2(sf::Vector2f(200.0f, 50.0f));
    button2.setFillColor(sf::Color::Blue);
    button2.setPosition(400.0f, 700.0f);

    sf::RectangleShape button3(sf::Vector2f(200.0f, 50.0f));
    button3.setFillColor(sf::Color::Red);
    button3.setPosition(700.0f, 700.0f);

    // Main loop
    while (window.isOpen()) {
        // Process events
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }

            // Handle button clicks
            if (event.type == sf::Event::MouseButtonPressed) {
                sf::Vector2i mousePos = sf::Mouse::getPosition(window);

                if (button1.getGlobalBounds().contains(static_cast<sf::Vector2f>(mousePos))) {
                    std::cout << "Button 1 clicked!" << std::endl;
                    // Add action for button 1
                }
                else if (button2.getGlobalBounds().contains(static_cast<sf::Vector2f>(mousePos))) {
                    std::cout << "Button 2 clicked!" << std::endl;
                    // Add action for button 2
                }
                else if (button3.getGlobalBounds().contains(static_cast<sf::Vector2f>(mousePos))) {
                    std::cout << "Button 3 clicked!" << std::endl;
                    // Add action for button 3
                }
            }
        }

        // Clear the window
        window.clear(sf::Color::White);

        // Draw the image on the left
        window.draw(sprite);

        // Draw the buttons at the bottom
        window.draw(button1);
        window.draw(button2);
        window.draw(button3);

        // Display the window contents
        window.display();
    }
}