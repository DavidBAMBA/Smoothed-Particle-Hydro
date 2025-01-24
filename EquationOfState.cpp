#include "EquationOfState.h"
#include <cmath>

EquationOfState::EquationOfState(double gammaVal) : GAMMA(gammaVal) {}

double EquationOfState::calculatePressure(double density, double specificInternalEnergy) const {
    return (GAMMA - 1.0) * density * specificInternalEnergy;
}

double EquationOfState::calculateSoundSpeed(double u, double pressure) const {
    return std::sqrt(GAMMA * (GAMMA - 1.0) * u);
}

double EquationOfState::getGamma() const {
    return GAMMA;
}
