#ifndef EQUATIONOFSTATE_H
#define EQUATIONOFSTATE_H

class EquationOfState {
public:
    EquationOfState(double gammaVal);
    
    double calculatePressure(double density, double specificInternalEnergy) const;
    double calculateSoundSpeed(double u, double pressure) const;
    double getGamma() const;

private:
    double GAMMA;
};

#endif // EQUATIONOFSTATE_H
