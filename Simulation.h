#ifndef SIMULATION_H
#define SIMULATION_H
#include <omp.h>
#include <iomanip> 

#include <vector>
#include <memory>
#include <string>
#include "Particle.h"
#include "DensityUpdater.h"
#include "EquationOfState.h"
#include "DissipationTerms.h"
#include "InitialConditions.h"
#include "Kernel.h"
#include "Boundaries.h"

#include <algorithm>    // Para std::find, std::min, std::remove_if
#include <iterator>

class Simulation {
public:
    Simulation(std::shared_ptr<Kernel> kern,
               std::shared_ptr<EquationOfState> eqs,
               std::shared_ptr<DissipationTerms> diss,
               std::shared_ptr<InitialConditions> initConds,
               double eta,
               double tol,
               bool use_fixed_h,
               double fixed_h,
               int N,                 // Número de partículas
               double x_min,
               double x_max,
               BoundaryType boundaryType);

    void run(double endTime);
    void restoreGhostParticles(); // Declaración del nuevo método


private:
    double time;
    std::vector<Particle> particles;
    std::shared_ptr<Kernel> kernel;
    std::shared_ptr<EquationOfState> eos;
    std::shared_ptr<DissipationTerms> dissipation;
    std::shared_ptr<InitialConditions> initialConditions;
    DensityUpdater densityUpdater; 
    Boundaries boundaries; 

    int N;
    double x_min;
    double x_max;

    void leapfrogStep(double timeStep);
    void modifiedKDKStep(double timeStep);
    void rungeKutta4Step(double timeStep);
    double calculateTimeStep() const;
    void writeOutputCSV(const std::string& filename) const;

    std::vector<Particle> getNeighbors(const Particle& particle) const;

};

#endif // SIMULATION_H
