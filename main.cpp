#include <iostream>
#include <memory>
#include "Simulation.h"
#include "EquationOfState.h"
#include "Kernel.h"
#include "DissipationTerms.h"
#include "InitialConditions.h"
#include "Boundaries.h"

int main() {
    try {
        // Definir parámetros
        int N = 4000;          // Número de partículas
        double x_min = -1.0;
        double x_max = 1.0;
        BoundaryType boundaryType = BoundaryType::FIXED; 
        // Otras opciones: BoundaryType::FIXED, BoundaryType::OPEN

        // Crear el objeto de condiciones iniciales
        auto initialConditions = std::make_shared<InitialConditions>();

        // Aquí es donde escoges el tipo de condición inicial que quieres:
        initialConditions->setInitialConditionType(InitialConditionType::SOD);
        //initialConditions->setInitialConditionType(InitialConditionType::BLAST);
        //initialConditions->setInitialConditionType(InitialConditionType::SEDOV);
        

        double GAMMA = 1.4; // Valor típico para aire
        auto eos = std::make_shared<EquationOfState>(GAMMA);
        
        KernelType choice = KernelType::M6; // M3 M5 M6
        auto kernel = std::make_shared<Kernel>(choice);

        double alpha = 1.0;
        double beta = 1.0;
        auto dissipation = std::make_shared<DissipationTerms>(alpha, beta);

        double eta = 1.2;
        double tol = 1e-6;
        
        double fixed_h = 0.009;      
        bool use_fixed_h = false;

        double endTime = 0.21;

        // Crear la simulación con todos los objetos necesarios
        Simulation sim(kernel, eos, dissipation, initialConditions,
                       eta, tol, use_fixed_h, fixed_h,
                       N, x_min, x_max, boundaryType);

        sim.run(endTime);

        std::cout << "Simulación completada exitosamente." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error durante la simulación: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
