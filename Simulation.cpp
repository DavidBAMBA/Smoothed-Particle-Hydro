// Simulation.cpp
#include "Simulation.h"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <filesystem>
#include "MathUtils.h"
#include <omp.h>
#include <iomanip>

#include <vector>       // Para std::vector
#include <algorithm>    // Para std::find, std::min, std::remove_if
#include <iterator>


Simulation::Simulation(std::shared_ptr<Kernel> kern,
                       std::shared_ptr<EquationOfState> eqs,
                       std::shared_ptr<DissipationTerms> diss,
                       std::shared_ptr<InitialConditions> initConds,
                       double eta,
                       double tol,
                       bool use_fixed_h,
                       double fixed_h,
                       int N,
                       double x_min,
                       double x_max,
                       BoundaryType boundaryType)
    : time(0.0),
      kernel(kern),
      eos(eqs),
      dissipation(diss),
      initialConditions(initConds),
      densityUpdater(eta, tol, use_fixed_h, fixed_h),
      boundaries(x_min, x_max, boundaryType),
      N(N),
      x_min(x_min),
      x_max(x_max) {}

std::vector<Particle> Simulation::getNeighbors(const Particle& particle) const {
    // Determinar el índice de la partícula actual
    auto it = std::find(particles.begin(), particles.end(), particle);
    if (it == particles.end()) {
        throw std::runtime_error("Partícula no encontrada en el sistema.");
    }
    
    // Índice de la partícula
    size_t index = std::distance(particles.begin(), it);
    
    // Límites para vecinos izquierdo y derecho
    size_t left_start = (index >= 500) ? index - 500 : 0; // Hasta 500 a la izquierda
    size_t right_end = std::min(index + 500 + 1, particles.size()); // Hasta 500 a la derecha

    // Extraer vecinos
    std::vector<Particle> neighbors(particles.begin() + left_start, particles.begin() + right_end);

    // Excluir la partícula actual
    neighbors.erase(std::remove_if(neighbors.begin(), neighbors.end(), 
                                   [&particle](const Particle& p) { return &p == &particle; }),
                    neighbors.end());

    return neighbors;
}

double Simulation::calculateTimeStep() const {
    double C_cour = 0.3;   // Coeficiente Courant
    double C_force = 0.25; // Coeficiente fuerza
    double alphaAV = 1.0;  // Alpha para la viscosidad artificial
    double betaAV = 2.0;   // Beta para la viscosidad artificial

    double min_dt = std::numeric_limits<double>::max();

    #pragma omp parallel for reduction(min:min_dt)
    for (size_t i = 0; i < particles.size(); ++i) {
        const auto& particle = particles[i];
        double v_sig_max = 0.0;

        // Calcular v_sig para todos los vecinos
        auto neighbors = getNeighbors(particle);
        for (const auto& neighbor : neighbors) {
            std::array<double, 3> v_ab = {
                particle.velocity[0] - neighbor.velocity[0],
                particle.velocity[1] - neighbor.velocity[1],
                particle.velocity[2] - neighbor.velocity[2]
            };

            // Diferencia de posición r_ab = r_a - r_b
            std::array<double, 3> r_ab = {
                particle.position[0] - neighbor.position[0],
                particle.position[1] - neighbor.position[1],
                particle.position[2] - neighbor.position[2]
            };

            // Norma de r_ab y vector unitario r_hat_ab
            std::array<double, 3> r_ab_hat = MathUtils::hatVector(r_ab);

            // Producto escalar v_ab . r_hat_ab
            double v_ab_dot_r_ab_hat = MathUtils::dotProduct(r_ab_hat, v_ab);

            // Velocidad de sonido de la partícula
            double p_a = particle.pressure; // Usar presión existente
            double cs_a = eos->calculateSoundSpeed(particle.specificInternalEnergy, p_a);

            // Cálculo de v_sig usando la fórmula de la imagen
            double v_sig = alphaAV * cs_a + betaAV * std::abs(v_ab_dot_r_ab_hat);

            // Actualizar v_sig máximo
            v_sig_max = std::max(v_sig_max, v_sig);
        }

        // Evitar división por cero
        if (v_sig_max <= 0.0) {
            continue;
        }

        double dt_cour = 0.0;
        if (v_sig_max > 0.0) {
            dt_cour = C_cour * (particle.h / v_sig_max);
        } else {
            dt_cour = std::numeric_limits<double>::max();
        }
        // Condición de fuerza (timestep por aceleración)
        double a_magnitude = 0.0;
        for (int i = 0; i < 3; ++i) {
            a_magnitude += std::pow(particle.acceleration[i], 2);
        }
        a_magnitude = std::sqrt(a_magnitude);
        double dt_force = (a_magnitude > 0.0) ? C_force * std::sqrt(particle.h / a_magnitude)
                                              : std::numeric_limits<double>::max();

        // Determinar el paso de tiempo mínimo para esta partícula
        double dt_particle = std::min(dt_cour, dt_force);
        #pragma omp critical
        {
            if (dt_particle < min_dt) {
                min_dt = dt_particle;
            }
        }
    }

    // Asegurar que el paso de tiempo no sea demasiado pequeño
    double min_allowed_dt = 1e-8;
    if (min_dt < min_allowed_dt) {
        min_dt = min_allowed_dt;
        #pragma omp critical
        {
            std::cerr << "Advertencia: Paso de tiempo ajustado al mínimo permitido: " << min_dt << std::endl;
        }
    }

    //std::cout << "Nuevo paso de tiempo calculado: " << min_dt << std::endl;
    return min_dt;
}

void Simulation::leapfrogStep(double timeStep) {
    // Paso 1: Actualizar velocidad a medio paso

    #pragma omp parallel for
    for (size_t i = 0; i < particles.size(); ++i) {
        for (int j = 0; j < 3; ++j) {
            particles[i].velocity[j] += 0.5 * timeStep * particles[i].acceleration[j];

        }
    }

    // Paso 2: Actualizar posición al tiempo n+1 usando velocidad a medio paso
    #pragma omp parallel for
    for (size_t i = 0; i < particles.size(); ++i) {
        for (int j = 0; j < 3; ++j) {
            particles[i].position[j] += timeStep * particles[i].velocity[j];
        }
    }

    // Paso 3: Actualizar densidad, presión y velocidad del sonido al tiempo n+1
    #pragma omp parallel for
    for (size_t i = 0; i < particles.size(); ++i) {
        auto neighbors = getNeighbors(particles[i]);
        particles[i].updateDensity(neighbors, densityUpdater, *kernel);
        particles[i].updatePressure(*eos);
        particles[i].updateSoundSpeed(*eos);
    }

    // Paso 4: Calcular aceleración y tasa de cambio de energía interna al tiempo n+1
    #pragma omp parallel for
    for (size_t i = 0; i < particles.size(); ++i) {
        auto neighbors = getNeighbors(particles[i]);
        particles[i].calculateAccelerationAndEnergyChangeRate(neighbors, *kernel, *dissipation, *eos);
    }

    // Paso 5: Actualizar velocidad a tiempo n+1 usando aceleración al tiempo n+1
    #pragma omp parallel for
    for (size_t i = 0; i < particles.size(); ++i) {
        for (int j = 0; j < 3; ++j) {
            particles[i].velocity[j] += 0.5 * timeStep * particles[i].acceleration[j];
        }
    }

    // Paso 6: Actualizar energía interna específica
    #pragma omp parallel for
    for (size_t i = 0; i < particles.size(); ++i) {
        particles[i].specificInternalEnergy += timeStep * particles[i].energyChangeRate;
    }


    // Actualizar energía específica total
    #pragma omp parallel for
    for (size_t i = 0; i < particles.size(); ++i) {
        particles[i].updateTotalSpecificEnergy();
    }

    // Aplicar condiciones de frontera
    boundaries.apply(particles);

    // Actualizar el tiempo
    time += timeStep;
}

void Simulation::rungeKutta4Step(double timeStep) {
    // Número de partículas
    size_t numParticles = particles.size();

    // Guardar el estado inicial de las partículas
    std::vector<std::array<double,3>> initialPositions(numParticles), initialVelocities(numParticles);
    std::vector<double> initialSpecificInternalEnergy(numParticles);

    for (size_t i = 0; i < numParticles; ++i) {
        initialPositions[i] = particles[i].position;
        initialVelocities[i] = particles[i].velocity;
        initialSpecificInternalEnergy[i] = particles[i].specificInternalEnergy;
    }

    // Vector de k para velocidades, posiciones y energía interna
    // k1, k2, k3, k4 para posición, velocidad y energía
    std::vector<std::array<double,3>> k1_pos(numParticles), k1_vel(numParticles);
    std::vector<std::array<double,3>> k2_pos(numParticles), k2_vel(numParticles);
    std::vector<std::array<double,3>> k3_pos(numParticles), k3_vel(numParticles);
    std::vector<std::array<double,3>> k4_pos(numParticles), k4_vel(numParticles);

    std::vector<double> k1_u(numParticles), k2_u(numParticles), k3_u(numParticles), k4_u(numParticles);

    // Paso 1: Evaluar derivadas en el estado inicial (k1)
    // Primero actualizar densidad, presión, velocidad del sonido, aceleración y tasa de energía en el estado actual.
    #pragma omp parallel for
    for (size_t i = 0; i < numParticles; ++i) {
        auto neighbors = getNeighbors(particles[i]);
        particles[i].updateDensity(neighbors, densityUpdater, *kernel);
        particles[i].updatePressure(*eos);
        particles[i].updateSoundSpeed(*eos);
    }

    #pragma omp parallel for
    for (size_t i = 0; i < numParticles; ++i) {
        auto neighbors = getNeighbors(particles[i]);
        particles[i].calculateAccelerationAndEnergyChangeRate(neighbors, *kernel, *dissipation, *eos);
    }

    // Calcular k1
    #pragma omp parallel for
    for (size_t i = 0; i < numParticles; ++i) {
        // dx/dt = v
        for (int j = 0; j < 3; ++j) {
            k1_pos[i][j] = particles[i].velocity[j] * timeStep;
            k1_vel[i][j] = particles[i].acceleration[j] * timeStep;
        }
        // du/dt = energyChangeRate
        k1_u[i] = particles[i].energyChangeRate * timeStep;
    }

    // Paso 2: Actualizar el estado a la mitad del paso con k1/2 para calcular k2
    #pragma omp parallel for
    for (size_t i = 0; i < numParticles; ++i) {
        for (int j = 0; j < 3; ++j) {
            particles[i].position[j] = initialPositions[i][j] + 0.5 * k1_pos[i][j];
            particles[i].velocity[j] = initialVelocities[i][j] + 0.5 * k1_vel[i][j];
        }
        particles[i].specificInternalEnergy = initialSpecificInternalEnergy[i] + 0.5 * k1_u[i];
    }

    // Recalcular condiciones para k2
    #pragma omp parallel for
    for (size_t i = 0; i < numParticles; ++i) {
        auto neighbors = getNeighbors(particles[i]);
        particles[i].updateDensity(neighbors, densityUpdater, *kernel);
        particles[i].updatePressure(*eos);
        particles[i].updateSoundSpeed(*eos);
    }

    #pragma omp parallel for
    for (size_t i = 0; i < numParticles; ++i) {
        auto neighbors = getNeighbors(particles[i]);
        particles[i].calculateAccelerationAndEnergyChangeRate(neighbors, *kernel, *dissipation, *eos);
    }

    // Calcular k2
    #pragma omp parallel for
    for (size_t i = 0; i < numParticles; ++i) {
        for (int j = 0; j < 3; ++j) {
            k2_pos[i][j] = particles[i].velocity[j] * timeStep;
            k2_vel[i][j] = particles[i].acceleration[j] * timeStep;
        }
        k2_u[i] = particles[i].energyChangeRate * timeStep;
    }

    // Paso 3: Actualizar el estado a la mitad del paso con k2/2 para calcular k3 (desde el estado inicial)
    #pragma omp parallel for
    for (size_t i = 0; i < numParticles; ++i) {
        for (int j = 0; j < 3; ++j) {
            particles[i].position[j] = initialPositions[i][j] + 0.5 * k2_pos[i][j];
            particles[i].velocity[j] = initialVelocities[i][j] + 0.5 * k2_vel[i][j];
        }
        particles[i].specificInternalEnergy = initialSpecificInternalEnergy[i] + 0.5 * k2_u[i];
    }

    // Recalcular condiciones para k3
    #pragma omp parallel for
    for (size_t i = 0; i < numParticles; ++i) {
        auto neighbors = getNeighbors(particles[i]);
        particles[i].updateDensity(neighbors, densityUpdater, *kernel);
        particles[i].updatePressure(*eos);
        particles[i].updateSoundSpeed(*eos);
    }

    #pragma omp parallel for
    for (size_t i = 0; i < numParticles; ++i) {
        auto neighbors = getNeighbors(particles[i]);
        particles[i].calculateAccelerationAndEnergyChangeRate(neighbors, *kernel, *dissipation, *eos);
    }

    // Calcular k3
    #pragma omp parallel for
    for (size_t i = 0; i < numParticles; ++i) {
        for (int j = 0; j < 3; ++j) {
            k3_pos[i][j] = particles[i].velocity[j] * timeStep;
            k3_vel[i][j] = particles[i].acceleration[j] * timeStep;
        }
        k3_u[i] = particles[i].energyChangeRate * timeStep;
    }

    // Paso 4: Actualizar el estado con k3 completo (para calcular k4)
    #pragma omp parallel for
    for (size_t i = 0; i < numParticles; ++i) {
        for (int j = 0; j < 3; ++j) {
            particles[i].position[j] = initialPositions[i][j] + k3_pos[i][j];
            particles[i].velocity[j] = initialVelocities[i][j] + k3_vel[i][j];
        }
        particles[i].specificInternalEnergy = initialSpecificInternalEnergy[i] + k3_u[i];
    }

    // Recalcular condiciones para k4
    #pragma omp parallel for
    for (size_t i = 0; i < numParticles; ++i) {
        auto neighbors = getNeighbors(particles[i]);
        particles[i].updateDensity(neighbors, densityUpdater, *kernel);
        particles[i].updatePressure(*eos);
        particles[i].updateSoundSpeed(*eos);
    }

    #pragma omp parallel for
    for (size_t i = 0; i < numParticles; ++i) {
        auto neighbors = getNeighbors(particles[i]);
        particles[i].calculateAccelerationAndEnergyChangeRate(neighbors, *kernel, *dissipation, *eos);
    }

    // Calcular k4
    #pragma omp parallel for
    for (size_t i = 0; i < numParticles; ++i) {
        for (int j = 0; j < 3; ++j) {
            k4_pos[i][j] = particles[i].velocity[j] * timeStep;
            k4_vel[i][j] = particles[i].acceleration[j] * timeStep;
        }
        k4_u[i] = particles[i].energyChangeRate * timeStep;
    }

    // Combinar los resultados
    #pragma omp parallel for
    for (size_t i = 0; i < numParticles; ++i) {
        for (int j = 0; j < 3; ++j) {
            particles[i].position[j] = initialPositions[i][j] + (k1_pos[i][j] + 2.0*k2_pos[i][j] + 2.0*k3_pos[i][j] + k4_pos[i][j]) / 6.0;
            particles[i].velocity[j] = initialVelocities[i][j] + (k1_vel[i][j] + 2.0*k2_vel[i][j] + 2.0*k3_vel[i][j] + k4_vel[i][j]) / 6.0;
        }
        particles[i].specificInternalEnergy = initialSpecificInternalEnergy[i] + (k1_u[i] + 2.0*k2_u[i] + 2.0*k3_u[i] + k4_u[i]) / 6.0;
    }

    // Actualizar la energía específica total
    #pragma omp parallel for
    for (size_t i = 0; i < numParticles; ++i) {
        particles[i].updateTotalSpecificEnergy();
    }

    // Aplicar condiciones de frontera
    boundaries.apply(particles);

    // Actualizar el tiempo de la simulación
    time += timeStep;
}

void Simulation::run(double endTime) {
    // Inicializar las partículas
    initialConditions->initializeParticles(particles, kernel, eos, N, x_min, x_max);

    int step = 0;
    double time = 0.0;

    // Crear el directorio "outputs" si no existe
    std::filesystem::create_directory("outputs");

    while (time < endTime) {
        double newTimeStep = calculateTimeStep(); // Usar el miembro eos

        // Escribir los datos en un archivo CSV cada 100 pasos
        if (step % 300 == 0) {
            std::string filename = "output_step_" + std::to_string(step) + ".csv";
            writeOutputCSV(filename);
        }

        // Llamar al integrador leapfrog
        //leapfrogStep(newTimeStep);
        rungeKutta4Step(newTimeStep);
        std::cout << "---------------------------------------------------------------------------\n"
                  << " \t STEP: " << step << "\t dt: "<< newTimeStep << "\t time: "<< time <<"\n"
                  << "---------------------------------------------------------------------------\n";
        time += newTimeStep;
        step++;
    }

    // Guardar los resultados finales
    writeOutputCSV("output_final.csv");
    std::cout << "Simulación completada exitosamente." << std::endl;
}

void Simulation::writeOutputCSV(const std::string& filename) const {
    // Crear el directorio "outputs" si no existe
    std::filesystem::create_directory("outputs");

    // Construir la ruta completa del archivo
    std::string fullPath = "outputs/" + filename;

    // Abrir archivo en la carpeta "outputs"
    std::ofstream file(fullPath);
    if (!file) {
        throw std::runtime_error("No se pudo abrir el archivo para escritura en: " + fullPath);
    }
    file << std::fixed << std::setprecision(8);

    // Escribir encabezados
    file << "t\tx\ty\tz\tvx\tvy\tvz\tP\tu\trho\th\tmass\tOmega\tconv\n";

    for (const auto& particle : particles) {
        file << time << "\t"
            << particle.position[0] << "\t"
            << particle.position[1] << "\t"
            << particle.position[2] << "\t"
            << particle.velocity[0] << "\t"
            << particle.velocity[1] << "\t"
            << particle.velocity[2] << "\t"
            << particle.pressure << "\t"
            << particle.specificInternalEnergy << "\t"
            << particle.density << "\t"
            << particle.h << "\t"
            << particle.mass << "\t"
            << particle.Omega << "\t"
            << particle.conv_h << "\n";

    }

    file.close();
    std::cout << "Datos escritos en " << fullPath << std::endl;
}

