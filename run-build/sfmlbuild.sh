#!/bin/bash

# Check if SFML directory already exists
if [ -d "SFML" ]; then
    echo "SFML directory already exists. Skipping clone."
else
    # Clone the SFML repository
    git clone https://github.com/SFML/SFML.git -b 2.5.x
    if [ $? -ne 0 ]; then
        echo "Failed to clone SFML repository."
        exit 1
    fi
fi

cd SFML

# Create a build directory and navigate into it
mkdir -p build
cd build

# Run CMake to configure the build
cmake ..
if [ $? -ne 0 ]; then
    echo "CMake configuration failed."
    exit 1
fi

# Build SFML using 7 available cores
make -j7
if [ $? -ne 0 ]; then
    echo "Failed to build SFML."
    exit 1
fi

echo "SFML built successfully."
