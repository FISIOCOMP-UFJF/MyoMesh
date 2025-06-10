cmake_minimum_required(VERSION 3.12)
project(PlyToStl)

# Especifique só os componentes necessários
find_package(VTK REQUIRED COMPONENTS
    IOPLY
    IOGeometry
    FiltersCore
    FiltersGeneral
    FiltersSources
    FiltersGeometry
    CommonCore
    CommonDataModel
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(PlyToStl ply_to_stl.cpp)

target_link_libraries(PlyToStl PRIVATE ${VTK_LIBRARIES})
