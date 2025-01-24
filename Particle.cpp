// Particle.cpp
#include "Particle.h"
#include "DensityUpdater.h"
#include "EquationOfState.h"
#include "DissipationTerms.h"
#include "Kernel.h"
#include "MathUtils.h"
#include <cmath>
#include <iostream>
#include <string>
#include <omp.h>


// Constructor
Particle::Particle(const std::array<double, 3>& position,
                   const std::array<double, 3>& velocity,
                   double mass,
                   double specificInternalEnergy)
    : mass(mass),
      density(0.0),
      pressure(0.0),
      specificInternalEnergy(specificInternalEnergy),
      totalSpecificEnergy(0.0),
      h(0.0),
      Omega(1.0),
      soundSpeed(0.0),
      position(position),
      velocity(velocity),
      acceleration({0.0, 0.0, 0.0}),
      energyChangeRate(0.0),
      conv_h("con")
{
    updateTotalSpecificEnergy();
}

void Particle::updateDensity(const std::vector<Particle>& neighbors,
                            DensityUpdater& densityUpdater, 
                            const Kernel& kernel) {
    densityUpdater.updateDensity(*this, neighbors, kernel);
}

void Particle::updatePressure(const EquationOfState& eos) {
    pressure = eos.calculatePressure(density, specificInternalEnergy);
}

void Particle::updateSoundSpeed(const EquationOfState& eos) {
    soundSpeed = eos.calculateSoundSpeed(specificInternalEnergy, pressure);
}

void Particle::calculateAccelerationAndEnergyChangeRate(const std::vector<Particle>& neighbors,
                                                        const Kernel& kernel,
                                                        const DissipationTerms& dissipation,
                                                        const EquationOfState& equationOfState) {
    double dv_dt_x = 0.0, dv_dt_y = 0.0, dv_dt_z = 0.0;
    double du_dt = 0.0;

    #pragma omp parallel for reduction(+:dv_dt_x, dv_dt_y, dv_dt_z, du_dt)
    for (size_t i = 0; i < neighbors.size(); ++i) {
        const auto& neighbor = neighbors[i];

        if (&neighbor == this) continue;

        // Vectors
        std::array<double, 3> r_ab = {
            position[0] - neighbor.position[0],
            position[1] - neighbor.position[1],
            position[2] - neighbor.position[2]
        };
        double r_ab_norm = MathUtils::vectorNorm(r_ab);
        std::array<double, 3> v_ab = {
            velocity[0] - neighbor.velocity[0],
            velocity[1] - neighbor.velocity[1],
            velocity[2] - neighbor.velocity[2]
        };

        // Gradientes del kernel
        std::array<double, 3> gradW_a = kernel.gradW(r_ab, h);
        std::array<double, 3> gradW_b = kernel.gradW(r_ab, neighbor.h);

        double v_ab_dot_gradW = MathUtils::dotProduct(v_ab, gradW_a);

        //if (r_ab_norm > std::max(3.0*h, 3.0*neighbor.h)) continue; // Fuera del soporte del kernel

        // Viscosidad artificial PHANTOM
        auto [q_ab_a, q_ab_b] = dissipation.PriceLondato2010Viscosity(*this, neighbor, kernel, equationOfState);

        // Factores de presión y viscosidad
        double factor_a = (pressure + q_ab_a) / (Omega * density * density);
        double factor_b = (neighbor.pressure + q_ab_b) / (neighbor.Omega * neighbor.density * neighbor.density);

        // Aceleración
        dv_dt_x -= neighbor.mass * (factor_a * gradW_a[0] + factor_b * gradW_b[0]);
        dv_dt_y -= neighbor.mass * (factor_a * gradW_a[1] + factor_b * gradW_b[1]);
        dv_dt_z -= neighbor.mass * (factor_a * gradW_a[2] + factor_b * gradW_b[2]);


        // Contribución a la energía
        //double v_ab_dot_gradW = MathUtils::dotProduct(v_ab, gradW_a);
        double term_energy = pressure / (density * density * Omega) * v_ab_dot_gradW;

        du_dt += term_energy * neighbor.mass;

        // Término disipativo (Lambda_shock)
        double viscous_heating = dissipation.viscousShockHeating(*this, neighbor, kernel, equationOfState);
        double thermal_conductivity = dissipation.artificialThermalConductivity(*this, neighbor, kernel, equationOfState);
        du_dt += (viscous_heating + thermal_conductivity)* neighbor.mass;
        

/*         // Viscosidad artificial monagan 1997
        auto [PI_ab, Ome_ab] = dissipation.mon97_art_vis(*this, neighbor, kernel, equationOfState);
        // Factores de presión y viscosidad
        double factor_a = (pressure) / (Omega * density * density);
        double factor_b = (neighbor.pressure) / (neighbor.Omega * neighbor.density * neighbor.density);

        // Aceleración
        for (int i = 0; i < 3; ++i) {
            dv_dt[i] -= neighbor.mass * (factor_a * gradW_a[i] + factor_b * gradW_b[i] + 0.5*PI_ab*(gradW_a[i]/Omega+gradW_b[i]/neighbor.Omega));
        }
        // Contribución a la energía
        //double v_ab_dot_gradW = MathUtils::dotProduct(v_ab, gradW_a);
        double term_energy = pressure / (density * density * Omega) * v_ab_dot_gradW;
        du_dt += (term_energy + Ome_ab ) * neighbor.mass;
 */


    }

    acceleration = {dv_dt_x, dv_dt_y, dv_dt_z};
    energyChangeRate = du_dt;
}

void Particle::updateTotalSpecificEnergy() {
    double kineticEnergy = 0.5 * MathUtils::dotProduct(velocity, velocity);
    totalSpecificEnergy = specificInternalEnergy + kineticEnergy;
}
