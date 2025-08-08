#include <vtkSmartPointer.h>
#include <vtkPLYReader.h>
#include <vtkSmoothPolyDataFilter.h>
#include <vtkPolyDataNormals.h>
#include <vtkTriangleFilter.h>
#include <vtkSTLWriter.h>
#include <vtkCleanPolyData.h>


// só quando for scar
#include <vtkDelaunay3D.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridWriter.h>

#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib> // para std::stoi

int main(int argc, char *argv[])
{
    // 1) input.ply
    // 2) output.stl
    // Ex.: ./program input.ply output.stl 0
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                << " input.ply output.stl scar(0/1)"
                << " [smooth(0/1)=1] [relax=0.05] [iters=200]\n";
        return EXIT_FAILURE;
    }

    std::string inputFileName = argv[1];
    std::string outputFileName = argv[2];
    bool scarRawFlag = (std::stoi(argv[3]) != 0);

    // argumentos opcionais
    int    scarSmoothFlag = (argc > 4) ? std::stoi(argv[4]) : 0;
    double relaxation     = (argc > 5) ? std::stod(argv[5]) : 0.05;
    int    iterations     = (argc > 6) ? std::stoi(argv[6]) : 300;


    // Leitura do arquivo PLY
    vtkSmartPointer<vtkPLYReader> reader = vtkSmartPointer<vtkPLYReader>::New();
    reader->SetFileName(inputFileName.c_str());
    reader->Update();

    // Guardar a malha original antes de suavizar
    vtkSmartPointer<vtkPolyData> originalMesh = reader->GetOutput();
    originalMesh->Register(nullptr); // garante que não seja desalocado prematuramente
    
    vtkSmartPointer<vtkSmoothPolyDataFilter> smoother = vtkSmartPointer<vtkSmoothPolyDataFilter>::New();
    // Suavização da malha
    if (!scarRawFlag){
        smoother->SetInputData(originalMesh);
    }

    // Define o fator de suavização conforme flagScar
    if(scarSmoothFlag)
        smoother->SetRelaxationFactor(relaxation);
    else if(scarRawFlag)
        smoother->SetRelaxationFactor(0.0002);
    
    if(!scarRawFlag){
        smoother->SetNumberOfIterations(200);
        smoother->SetRelaxationFactor(0.02);
        smoother->BoundarySmoothingOff();
        smoother->BoundarySmoothingOff();
    }
    vtkSmartPointer<vtkPolyData> meshForNormals;

    if (!scarRawFlag){
        smoother->Update();
        meshForNormals = smoother->GetOutput();
    }else
        meshForNormals = originalMesh;

    // Gera as normais a partir da malha adequada
    vtkSmartPointer<vtkPolyDataNormals> normals =
        vtkSmartPointer<vtkPolyDataNormals>::New();
    normals->SetInputData(meshForNormals);
    normals->FlipNormalsOn();
    normals->Update();
    normals->FlipNormalsOn();
    normals->Update();

    // Triangula a malha
    vtkSmartPointer<vtkTriangleFilter> triangleFilter = vtkSmartPointer<vtkTriangleFilter>::New();
    triangleFilter->SetInputData(normals->GetOutput());
    triangleFilter->Update();

   // Limpa duplicatas
    vtkSmartPointer<vtkCleanPolyData> cleaner =
        vtkSmartPointer<vtkCleanPolyData>::New();
    cleaner->SetInputData(triangleFilter->GetOutput());
    cleaner->Update();

    // Malha final de superfície
    vtkSmartPointer<vtkPolyData> finalMesh = cleaner->GetOutput();

    // (Opcional) mapeamento de vértices
    if (originalMesh->GetNumberOfPoints() == finalMesh->GetNumberOfPoints())
    {
        std::ofstream mapFile("vertex_mapping.txt");
        if (mapFile.is_open())
        {
            vtkIdType N = originalMesh->GetNumberOfPoints();
            for (vtkIdType i = 0; i < N; ++i)
            {
                double o[3], s[3];
                originalMesh->GetPoint(i, o);
                finalMesh   ->GetPoint(i, s);
                mapFile << i << " "
                        << o[0] << " " << o[1] << " " << o[2] << " "
                        << s[0] << " " << s[1] << " " << s[2] << "\n";
            }
        }
        else
        {
            std::cerr << "Could not open 'vertex_mapping.txt'\n";
        }
    }


    // Escreve a malha final em formato STL
    vtkSmartPointer<vtkSTLWriter> writer = vtkSmartPointer<vtkSTLWriter>::New();
    writer->SetFileName(outputFileName.c_str());
    writer->SetInputData(finalMesh);
    writer->SetFileTypeToASCII();
    writer->Write();

    // Libera a referência extra do originalMesh
    originalMesh->Delete();

    return EXIT_SUCCESS;
}
