// DissipationTerms.cpp
#include "DissipationTerms.h"
#include "MathUtils.h"
#include <algorithm>
#include <Kernel.h>
#include <EquationOfState.h> 

// Constructor
DissipationTerms::DissipationTerms(double alpha_value, double beta_value)
    : alphaAV(alpha_value), betaAV(beta_value) {}

// Implementación de la Viscosidad Artificial Monaghan 1997 (Riemann Solver)
std::pair<double, double> DissipationTerms::mon97_art_vis(const Particle& a, const Particle& b, const Kernel& kernel, const EquationOfState& equationOfState) const {
    // Diferencia de velocidad y posición
    std::array<double, 3> v_ab = {
        a.velocity[0] - b.velocity[0],
        a.velocity[1] - b.velocity[1],
        a.velocity[2] - b.velocity[2]
    };

    std::array<double, 3> r_ab = {
        a.position[0] - b.position[0],
        a.position[1] - b.position[1],
        a.position[2] - b.position[2]
    };

    std::array<double, 3> r_ab_hat = MathUtils::hatVector(r_ab);
    double q = MathUtils::vectorNorm(r_ab) / a.h;

    double v_ab_dot_r_ab_hat = MathUtils::dotProduct(v_ab, r_ab_hat);

    if (v_ab_dot_r_ab_hat >= 0.0) {
        return {0.0, 0.0}; // Partículas separándose
    }

    // Valores promedio
    double p_a = equationOfState.calculatePressure(a.density, a.specificInternalEnergy);
    double p_b = equationOfState.calculatePressure(b.density, b.specificInternalEnergy);
    double rho_mean = 0.5 * (a.density + b.density);
    // Calcula las velocidades del sonido para cada partícula
    double cs_a = equationOfState.calculateSoundSpeed(a.specificInternalEnergy , p_a);
    double cs_b = equationOfState.calculateSoundSpeed(b.specificInternalEnergy, p_b);

    // Velocidad de señal
    double v_sig = 0.5 * alphaAV * (cs_a + cs_b - betaAV * v_ab_dot_r_ab_hat );

    // Término PI_ab
    double PI_ab = - v_sig * v_ab_dot_r_ab_hat * (1.0 / rho_mean);

    // Término de energía
    double Ome_ab = 0.5 * PI_ab * v_ab_dot_r_ab_hat*kernel.F(q, a.h) / MathUtils::vectorNorm(r_ab);

    return {PI_ab*0.0, Ome_ab*0.0};
}

// Implementación de la Conductividad Artificial Price 2008
double DissipationTerms::price08_therm_cond(double p_i, double p_j, double rho_mean, double u_i, double u_j) const {
    // Parámetros
    double alpha_u = 1.0;

    // Velocidad de señal térmica
    double v_sig_u = std::sqrt(std::abs((p_i - p_j) / rho_mean));

    // Contribución a du/dt
    double dudt = alpha_u * v_sig_u * (u_i - u_j) / rho_mean;

    return dudt;
}

std::pair<double, double> DissipationTerms::PriceLondato2010Viscosity(const Particle& a, const Particle& b, const Kernel& kernel, const EquationOfState& equationOfState) const {
    // Diferencia de velocidad y posición
    std::array<double, 3> v_ab = {
        a.velocity[0] - b.velocity[0],
        a.velocity[1] - b.velocity[1],
        a.velocity[2] - b.velocity[2]
    };

    std::array<double, 3> r_ab = {
        a.position[0] - b.position[0],
        a.position[1] - b.position[1],
        a.position[2] - b.position[2]
    };

    std::array<double, 3> r_ab_hat = MathUtils::hatVector(r_ab);
    double v_ab_dot_r_ab_hat = MathUtils::dotProduct(v_ab, r_ab_hat);
    
    if (v_ab_dot_r_ab_hat >= 0.0) {
        return {0.0, 0.0}; // Partículas separándose
    }
    
    // Calcula las velocidades del sonido para cada partícula
    double p_a = equationOfState.calculatePressure(a.density, a.specificInternalEnergy);
    double p_b = equationOfState.calculatePressure(b.density, b.specificInternalEnergy);
    double cs_a = equationOfState.calculateSoundSpeed(a.specificInternalEnergy, p_a);
    double cs_b = equationOfState.calculateSoundSpeed(b.specificInternalEnergy, p_b);

    // Velocidad de señal
    double v_sig_a =  alphaAV * cs_a + betaAV * std::abs(v_ab_dot_r_ab_hat);
    double v_sig_b =  alphaAV * cs_b + betaAV * std::abs(v_ab_dot_r_ab_hat);

    double q_ab_a = -0.5 * a.density * v_sig_a * v_ab_dot_r_ab_hat;
    double q_ab_b = -0.5 * b.density * v_sig_b * v_ab_dot_r_ab_hat;

    
    return {q_ab_a, q_ab_b};
}

// Implementación de Viscous Shock Heating
double DissipationTerms::viscousShockHeating(const Particle& a, const Particle& b, const Kernel& kernel, const EquationOfState& equationOfState) const {
    // Diferencia de velocidad
    std::array<double, 3> v_ab = {
        a.velocity[0] - b.velocity[0],
        a.velocity[1] - b.velocity[1],
        a.velocity[2] - b.velocity[2]
    };

    // Vector de posición
    std::array<double, 3> r_ab = {
        a.position[0] - b.position[0],
        a.position[1] - b.position[1],
        a.position[2] - b.position[2]
    };

    double r_ab_norm = MathUtils::vectorNorm(r_ab);
    if (r_ab_norm == 0.0) return 0.0;

    std::array<double,3> r_hat = MathUtils::hatVector(r_ab);
    double v_ab_dot_r_ab_hat = MathUtils::dotProduct(v_ab, r_hat);

    // Calcula las velocidades del sonido para cada partícula
    double p_a = equationOfState.calculatePressure(a.density, a.specificInternalEnergy);
    double cs_a = equationOfState.calculateSoundSpeed(a.specificInternalEnergy, p_a);

    // Velocidad de señal
    double v_sig_a =  alphaAV * cs_a + betaAV * std::abs(v_ab_dot_r_ab_hat);

    double q_a = r_ab_norm / a.h;

    // Término Lambda_shock - Viscous Shock Heating
    double term1 = - (1.0 / (a.Omega * a.density)) * v_sig_a * 0.5 * std::pow(v_ab_dot_r_ab_hat, 2) * kernel.F(q_a, a.h);

    return term1;
}

// Implementación de Artificial Thermal Conductivity
double DissipationTerms::artificialThermalConductivity(const Particle& a, const Particle& b, const Kernel& kernel, const EquationOfState& equationOfState) const {
    // Vector de posición
    std::array<double, 3> r_ab = {
        a.position[0] - b.position[0],
        a.position[1] - b.position[1],
        a.position[2] - b.position[2]
    };

    double r_ab_norm = MathUtils::vectorNorm(r_ab);
    //std::array<double,3> r_ab_hat = MathUtils::hatVector(r_ab);

    double q_a = r_ab_norm / a.h;
    double q_b = r_ab_norm / b.h;

    // Compute signal velocity 
    double p_a = equationOfState.calculatePressure(a.density, a.specificInternalEnergy);
    double p_b = equationOfState.calculatePressure(b.density, b.specificInternalEnergy);
    double rho_mean = 0.5*(a.density+b.density);
    double v_sig = std::sqrt( std::abs(p_a-p_b) / rho_mean);

    // Diferencia de energía interna
    double u_diff = a.specificInternalEnergy - b.specificInternalEnergy;
    double F_ab_mean = 0.5 * (kernel.F(q_a, a.h) / (a.Omega * a.density) + kernel.F(q_b,b.h) / (b.Omega * b.density));
    
    // Término Lambda_shock - Artificial Thermal Conductivity
    double term2 =  v_sig * u_diff * F_ab_mean;
    
    return term2;
}

