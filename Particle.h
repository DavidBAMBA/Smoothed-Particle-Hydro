// Particle.h
#ifndef PARTICLE_H
#define PARTICLE_H

#include <array>
#include <string>
#include <omp.h>
#include <memory>
#include <vector>

class EquationOfState;
class DissipationTerms;
class DensityUpdater;
class Kernel;

// Propiedades iniciales específicas para partículas fantasma
struct GhostProperties {
    std::array<double, 3> initialVelocity;
    double initialDensity;
    double initialPressure;
    double initialSpecificInternalEnergy;
};

class Particle {
public:

    bool isGhost;
    std::shared_ptr<GhostProperties> ghostProperties; // Propiedades compartidas entre partículas fantasma

    Particle(const std::array<double, 3>& position,
             const std::array<double, 3>& velocity,
             double mass,
             double specificInternalEnergy,
             bool ghost = false,
             std::shared_ptr<GhostProperties> ghostProps = nullptr);

    // Métodos para actualizar variables físicas
    void updateDensity(const std::vector<Particle>& neighbors, DensityUpdater& densityUpdater, const Kernel& kernel);
    void updatePressure(const EquationOfState& eos);
    void updateSoundSpeed(const EquationOfState& eos);
    void calculateAccelerationAndEnergyChangeRate(const std::vector<Particle>& neighbors,
                                                  const Kernel& kernel,
                                                  const DissipationTerms& dissipation,
                                                  const EquationOfState& equationOfState);
    void updateTotalSpecificEnergy();

    // Atributos públicos para facilitar el acceso directo
    double mass;
    double density;
    double pressure;
    double specificInternalEnergy;
    double totalSpecificEnergy;
    double h;
    double Omega;
    double soundSpeed;
    std::array<double, 3> position;
    std::array<double, 3> velocity;
    std::array<double, 3> acceleration;
    double energyChangeRate;
    mutable std::string conv_h;
        /**
     * @brief Sobrecarga del operador == para comparar partículas.
     * @param other  Otra partícula con la que comparar.
     * @return true si las partículas tienen la misma posición.
     */
    bool operator==(const Particle& other) const {
        return position == other.position;
    }
};

#endif // PARTICLE_H