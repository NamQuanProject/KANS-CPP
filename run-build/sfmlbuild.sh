#!/bin/bash

# Check if SFML directory already exists
if [ -d "SFML" ]; then
    echo "SFML directory already exists. Skipping clone."
else
    # Clone the SFML repository
    git clone https://github.com/SFML/SFML.git -b 2.6.x
    if [ $? -ne 0 ]; then
        echo "Failed to clone SFML repository."
        echo "Attempting to download SFML source archive..."

        # Download SFML source archive
        wget https://www.sfml-dev.org/files/SFML-2.6.1-sources.zip
        if [ $? -ne 0 ]; then
            echo "Failed to download SFML source archive."
            exit 1
        fi

        # Unzip SFML source archive
        unzip SFML-2.6.1-sources.zip
        if [ $? -ne 0 ]; then
            echo "Failed to unzip SFML source archive."
            exit 1
        fi

        # Clean up downloaded archive
        rm SFML-2.6.1-sources.zip

        # Move unzipped directory to SFML
        mv SFML-2.6.1 SFML
        if [ $? -ne 0 ]; then
            echo "Failed to move SFML source directory."
            exit 1
        fi
    fi
fi

# Navigate into SFML directory
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
