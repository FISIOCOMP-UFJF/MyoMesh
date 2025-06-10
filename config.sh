#!/bin/bash
#set -x

dir="MyoMesh"

if [ $(basename "$PWD") = "$dir" ]; then

    if command -v git &> /dev/null; then
        echo "==================================================="
        echo "Cloning hexa-mesh-from-VTK repository..."
        echo "==================================================="
        git clone https://github.com/FilipeNamorato/hexa-mesh-from-VTK_vtk9.git

        if [ $? -eq 0 ]; then
            echo "Repository cloned successfully."
            cd hexa-mesh-from-VTK_vtk9
            cmake .
            if [ $? -eq 0 ]; then
                echo "CMake configuration successful."
                make
                if [ $? -eq 0 ]; then
                    echo "Project build successful."
                else
                    echo "Failed to build project."
                    exit 1
                fi
            else
                echo "Failed to configure project with CMake."
                exit 1
            fi
            cd ..
        else
            echo "Failed to clone repository."
            exit 1
        fi

        if [ ! -d "convertPly2STL/build" ]; then
            echo "==================================================="
            echo "Compiling convertPly2STL..."
            echo "==================================================="
            mkdir -p convertPly2STL/build
            cd convertPly2STL/build
            cmake ..
            if [ $? -ne 0 ]; then
                echo "Failed to configure convertPly2STL with CMake."
                exit 1
            fi
            make
            if [ $? -ne 0 ]; then
                echo "Failed to build convertPly2STL."
                exit 1
            fi
            cd ../..
        else
            echo "convertPly2STL already built. Skipping."
        fi

    else
        echo "Git is not installed. Unable to clone the repository."
        exit 1
    fi

else
    echo "You are not in the desired directory."
    echo "Please go to the ${dir} directory and run the script again."
    exit 1
fi
