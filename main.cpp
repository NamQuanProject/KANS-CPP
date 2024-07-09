#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <SFML/Graphics.hpp>

#include <iostream>
#include "dataset.h"
#include "KAN.cpp"
#include "KANsLinear.cpp"
#include "train.cpp"

int main() {
    sf::RenderWindow window(sf::VideoMode(800, 600), "SFML Example");
    window.setFramerateLimit(60);


    // Main loop
    while (window.isOpen()) {
        // Process events
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
        }

        // Clear the window
        window.clear(sf::Color::Black);

        // Draw shapes
        window.draw(circle);
        window.draw(rectangle);

        // Display the window contents
        window.display();
    }

    return 0;
}
