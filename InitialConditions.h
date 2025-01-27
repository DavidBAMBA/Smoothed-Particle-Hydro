#ifndef INITIALCONDITIONS_H
#define INITIALCONDITIONS_H

#include <vector>
#include <memory>
#include "Particle.h"
#include "Kernel.h"
#include "EquationOfState.h"
#include "DensityUpdater.h"


enum class InitialConditionType {
    SOD,
    BLAST,
    SEDOV,
    TEST
};

class InitialConditions {
public:

    InitialConditions();
    void setInitialConditionType(InitialConditionType type);
    void initializeParticles(std::vector<Particle>& particles,
                             std::shared_ptr<Kernel> kernel,
                             std::shared_ptr<EquationOfState> eos,
                             int N, double x_min, double x_max);

private:
    InitialConditionType icType;
};

#endif // INITIALCONDITIONS_H
