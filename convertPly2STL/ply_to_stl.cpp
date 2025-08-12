#include <vtkSmartPointer.h>
#include <vtkPLYReader.h>
#include <vtkSTLReader.h>
#include <vtkSmoothPolyDataFilter.h>
#include <vtkPolyDataNormals.h>
#include <vtkTriangleFilter.h>
#include <vtkSTLWriter.h>
#include <vtkCleanPolyData.h>

#include <vtkDelaunay3D.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridWriter.h>

#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>   // std::stoi, std::stod
#include <algorithm> // std::transform
#include <cctype>    // std::tolower

int main(int argc, char *argv[])
{
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " input.[ply|stl] output.stl scar(0/1)"
                  << " [smooth(0/1)=1] [relax=0.05] [iters=200] [scar_stage(1|2)=2]\n";
        return EXIT_FAILURE;
    }

    const std::string inputFileName  = argv[1];
    const std::string outputFileName = argv[2];
    const bool isScar = (std::stoi(argv[3]) != 0);

    // Defaults alinhados ao Usage
    int    doSmooth   = (argc > 4) ? std::stoi(argv[4]) : 1;      // 1 = apply smooth
    double relaxation = (argc > 5) ? std::stod(argv[5]) : 0.05;   // relaxation factor
    int    iterations = (argc > 6) ? std::stoi(argv[6]) : 200;    // iterations
    int    scarStage  = (argc > 7) ? std::stoi(argv[7]) : 2;      // 1=closure; 2=refinement

    // Stage 1 (closure) for fibrosis: smooth force and fixed relaxation=0.0002
    if (isScar && scarStage == 1) {
        doSmooth   = 1;
        relaxation = 0.0002;   
        
    }

    auto loadPolyData = [](const std::string& path) -> vtkSmartPointer<vtkPolyData> {
        std::string ext;
        if (auto pos = path.find_last_of('.'); pos != std::string::npos)
            ext = path.substr(pos);
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c){ return std::tolower(c); });

        if (ext == ".ply") {
            auto r = vtkSmartPointer<vtkPLYReader>::New();
            r->SetFileName(path.c_str());
            r->Update();
            return r->GetOutput();
        } else if (ext == ".stl") {
            auto r = vtkSmartPointer<vtkSTLReader>::New();
            r->SetFileName(path.c_str());
            r->Update();
            return r->GetOutput();
        } else {
            std::cerr << "Extensão não suportada: " << ext << " (use .ply ou .stl)\n";
            return nullptr;
        }
    };

    
    vtkSmartPointer<vtkPolyData> originalMesh = loadPolyData(inputFileName);
    if (!originalMesh || originalMesh->GetNumberOfPoints() == 0) {
        std::cerr << "Erro: malha vazia ou inválida em '" << inputFileName << "'.\n";
        return EXIT_FAILURE;
    }
    originalMesh->Register(nullptr); // evita desalocação antecipada

    vtkSmartPointer<vtkPolyData> meshForNormals = originalMesh;

    // Suavização (apenas se solicitado)
    vtkSmartPointer<vtkSmoothPolyDataFilter> smoother;
    if (doSmooth) {
        smoother = vtkSmartPointer<vtkSmoothPolyDataFilter>::New();
        smoother->SetInputData(originalMesh);
        smoother->SetNumberOfIterations(iterations);
        smoother->SetRelaxationFactor(relaxation);
        smoother->FeatureEdgeSmoothingOff();
        smoother->BoundarySmoothingOff();
        smoother->Update();
        meshForNormals = smoother->GetOutput();
    }

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

    // Limpeza
    auto cleaner = vtkSmartPointer<vtkCleanPolyData>::New();
    cleaner->SetInputData(triangleFilter->GetOutput());
    if (isScar && scarStage == 1) {
        cleaner->PointMergingOff(); // no fechamento, preserva contagem/índices
    } else {
        cleaner->PointMergingOn();  // no ajuste fino, permite merge
    }
    cleaner->Update();

    // Malha final de superfície
    vtkSmartPointer<vtkPolyData> finalMesh = cleaner->GetOutput();

    // (Opcional) mapeamento de vértices (só se contagem não mudou)
    if (originalMesh->GetNumberOfPoints() == finalMesh->GetNumberOfPoints()) {
        std::ofstream mapFile("vertex_mapping.txt");
        if (mapFile.is_open()) {
            vtkIdType N = originalMesh->GetNumberOfPoints();
            for (vtkIdType i = 0; i < N; ++i) {
                double o[3], s[3];
                originalMesh->GetPoint(i, o);
                finalMesh->GetPoint(i, s);
                mapFile << i << " "
                        << o[0] << " " << o[1] << " " << o[2] << " "
                        << s[0] << " " << s[1] << " " << s[2] << "\n";
            }
        } else {
            std::cerr << "Aviso: não foi possível abrir 'vertex_mapping.txt'.\n";
        }
    }

    // Escrita STL
    auto writer = vtkSmartPointer<vtkSTLWriter>::New();
    writer->SetFileName(outputFileName.c_str());
    writer->SetInputData(finalMesh);
    writer->SetFileTypeToASCII(); // use SetFileTypeToBinary() se preferir menor tamanho
    if (!writer->Write()) {
        std::cerr << "Erro ao escrever STL em '" << outputFileName << "'.\n";
        originalMesh->Delete();
        return EXIT_FAILURE;
    }

    // Libera referência extra
    originalMesh->Delete();
    return EXIT_SUCCESS;
}
