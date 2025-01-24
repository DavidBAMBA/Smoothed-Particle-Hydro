#include "InitialConditions.h"
#include <cmath>
#include "DensityUpdater.h"
#include <iostream>

InitialConditions::InitialConditions() : icType(InitialConditionType::SOD) {}

void InitialConditions::setInitialConditionType(InitialConditionType type) {
    icType = type;
}

void InitialConditions::initializeParticles(std::vector<Particle>& particles,
                                            std::shared_ptr<Kernel> kernel,
                                            std::shared_ptr<EquationOfState> eos,
                                            int N,
                                            double x_min,
                                            double x_max) {
    double GAMMA = eos->getGamma();
    DensityUpdater densityUpdater(1.2, 1e-4, false, 0.1);

    switch (icType) {
// Ejemplo de distribución 1D para SOD usando repartición proporcional a la masa
    case InitialConditionType::SOD:
        {
            // Datos del problema
            double density_left  = 1.0;
            double density_right = 0.125;
            double pressure_left  = 1.0;
            double pressure_right = 0.1;
            // Se asume que x_min y x_max están definidos

            // La discontinuidad se coloca, por ejemplo, de forma proporcional (en este caso, en el centro)
            double x_discontinuity = 0.5 * (x_min + x_max);
            
            // Longitudes de cada región (en 1D, es simplemente la longitud)
            double L_left  = x_discontinuity - x_min;
            double L_right = x_max - x_discontinuity;
            
            // Calcula la masa total en cada región (masa = densidad * longitud)
            double mass_total_left  = density_left  * L_left;
            double mass_total_right = density_right * L_right;
            
            // Número total de partículas (por ejemplo, 1000)
            //int N = 1000;
            
            // Asignar el número de partículas de cada lado de manera proporcional a su masa total
            int N_left  = static_cast<int>( N * mass_total_left  / (mass_total_left  + mass_total_right) );
            int N_right = N - N_left;
            
            // Espaciamiento en cada región (se coloca cada partícula en el centro de su celda)
            double dx_left  = L_left  / N_left;
            double dx_right = L_right / N_right;
            
            // Se calcula una masa única para cada partícula (la masa total del sistema dividido entre N)
            double mass_total = mass_total_left + mass_total_right;
            double mass_per_particle = mass_total / N;
            
            
            // Distribución en la región izquierda
            for (int i = 0; i < N_left; ++i) {
                // Posición centrada en la celda
                double x = x_min + (i + 0.5) * dx_left;
                std::array<double, 3> position = { x, 0.0, 0.0 };
                std::array<double, 3> velocity = { 0.0, 0.0, 0.0 };
                
                double pressure = pressure_left;
                double density  = density_left;
                double specificInternalEnergy = pressure / ((GAMMA - 1.0) * density);
                
                // Construye la partícula (se asume que el constructor recibe (posición, velocidad, masa, energía interna))
                Particle particle(position, velocity, mass_per_particle, specificInternalEnergy);
                particle.density = density;
                
                // Calcular el parámetro de suavizado; aquí usamos un factor (por ejemplo 1.2) por similitud con el ejemplo original.
                //particle.h = 1.2 * (mass_per_particle / density_left); 
                particle.h = 0.0005;//
                particle.Omega = densityUpdater.calculateOmega(particle, particles, particle.h, particle.density, *kernel);
                particle.pressure = pressure_left;
                particle.updatePressure(*eos);
                //particle.updateSoundSpeed(*eos);
                //particle.updateDensity(particle[i]);
                
                particles.push_back(particle);
            }
            
            // Distribución en la región derecha
            for (int i = 0; i < N_right; ++i) {
                double x = x_discontinuity + (i + 0.5) * dx_right;
                std::array<double, 3> position = { x, 0.0, 0.0 };
                std::array<double, 3> velocity = { 0.0, 0.0, 0.0 };
                
                double pressure = pressure_right;
                double density  = density_right;
                double specificInternalEnergy = pressure / ((GAMMA - 1.0) * density);
                
                Particle particle(position, velocity, mass_per_particle, specificInternalEnergy);
                particle.density = density;
                
                particle.h = 0.005;//1.2 * (mass_per_particle / density_right);
                //particle.h = 1.2 * (mass_per_particle / density_left); //0.0005;//
                particle.Omega = densityUpdater.calculateOmega(particle, particles, particle.h, particle.density, *kernel);
                particle.pressure = pressure_right;

                
                particle.updatePressure(*eos);
                //particle.updateSoundSpeed(*eos);
                
                particles.push_back(particle);
            }
        }
    break;


    case InitialConditionType::BLAST:
        {
            // ---- BLAST WAVE ----
            double density_left = 1.0;
            double density_right = 1.0;

            double x_discontinuity = 0.5 * (x_min + x_max);

            double volume_left = x_discontinuity - x_min;
            double volume_right = x_max - x_discontinuity;

            double mass_left  = density_left * volume_left;
            double mass_right = density_right * volume_right;

            int N_left = static_cast<int>(N * mass_left / (mass_left + mass_right));
            int N_right = N - N_left;

            double mass_per_particle = (mass_left + mass_right) / N;

            double dx_left = volume_left / N_left;
            double dx_right = volume_right / N_right;

            // Izquierda
            for (int i = 0; i < N_left; ++i) {
                double x = x_min + (i + 0.5) * dx_left;
                std::array<double, 3> position = {x, 0.0, 0.0};
                std::array<double, 3> velocity = {0.0, 0.0, 0.0};

                double pressure = 1000.0;
                double density = density_left;
                double specificInternalEnergy = pressure / ((GAMMA - 1.0) * density);

                Particle particle(position, velocity, mass_per_particle, specificInternalEnergy);

                particle.density = density;
                particle.h = 0.009;
                particle.Omega = densityUpdater.calculateOmega(particle, particles, particle.h, particle.density, *kernel);


                particle.updatePressure(*eos);
                particle.updateSoundSpeed(*eos);

                particles.push_back(particle);
            }

            // Derecha
            for (int i = 0; i < N_right; ++i) {
                double x = x_discontinuity + (i + 0.5) * dx_right;
                std::array<double, 3> position = {x, 0.0, 0.0};
                std::array<double, 3> velocity = {0.0, 0.0, 0.0};

                double pressure = 0.1;
                double density = density_right;
                double specificInternalEnergy = pressure / ((GAMMA - 1.0) * density);

                Particle particle(position, velocity, mass_per_particle, specificInternalEnergy);

                particle.density = density;
                particle.h = 0.09;
                particle.Omega = densityUpdater.calculateOmega(particle, particles, particle.h, particle.density, *kernel);


                particle.updatePressure(*eos);
                particle.updateSoundSpeed(*eos);

                particles.push_back(particle);
            }
        }
        break;

    case InitialConditionType::SEDOV:
        {
            // ---- SEDOV BLAST WAVE ----
            double density = 1.0;  
            double L = x_max - x_min;
            density = 1.0 / L; // ajustar densidad para masa total = 1
            double mass_per_particle = 1.0 / N;
            double dx = L / N;

            int Nx_center = 5;
            if (Nx_center >= N) Nx_center = N/2;
            if (Nx_center < 1) Nx_center = 1;

            int i_center = N/2;
            int i_start = i_center - Nx_center/2;
            if (i_start < 0) i_start = 0;
            int i_end = i_start + Nx_center;
            if (i_end > N) i_end = N;

            double E0 = 1.0e2;
            double u0 = E0 / (Nx_center * mass_per_particle * (GAMMA - 1.0));

            double p_out = 1e-5;
            double u_out = p_out / ((GAMMA - 1.0)*density);

            for (int i = 0; i < N; ++i) {
                double x = x_min + (i + 0.5)*dx;
                std::array<double, 3> position = {x, 0.0, 0.0};
                std::array<double, 3> velocity = {0.0, 0.0, 0.0};

                double u;
                if (i >= i_start && i < i_end) {
                    u = u0;
                } else {
                    u = u_out;
                }

                double p = (GAMMA - 1.0)*density*u;
                Particle particle(position, velocity, mass_per_particle, u);

                double eta = 1.2;
                double h = eta * std::sqrt(mass_per_particle / density);

                particle.density = density;
                particle.h = h;
                particle.pressure = p;

                particle.updatePressure(*eos);
                particle.updateSoundSpeed(*eos);

                particles.push_back(particle);
            }
        }
        break;

            case InitialConditionType::TEST:
    {
        // ---- TEST ----
        // Propiedades uniformes para todas las partículas reales
        double uniform_density = 1.0;
        double uniform_pressure = 1.0;
        double uniform_velocity = 0.0;
        double mass_per_particle = (x_max - x_min) * uniform_density / (N + 2*N*0.1);
        double h = 1.2 * (mass_per_particle/ uniform_density); // Longitud de suavizado constante

        // Espaciado uniforme
        double dx = (x_max - x_min) / N;
        double perturbation = dx * 0.01 * (rand() / double(RAND_MAX) - 0.5);


       /*  // Propiedades fantasma
        auto ghostPropsLeft = std::make_shared<GhostProperties>(
            GhostProperties{{uniform_velocity, 0.0, 0.0}, uniform_density, uniform_pressure, uniform_pressure / ((GAMMA - 1.0) * uniform_density)});
        auto ghostPropsRight = std::make_shared<GhostProperties>(
            GhostProperties{{uniform_velocity, 0.0, 0.0}, uniform_density, uniform_pressure, uniform_pressure / ((GAMMA - 1.0) * uniform_density)});

        // Número de partículas fantasma
        int numGhostsLeft = static_cast<int>(std::ceil(N*0.1));
        int numGhostsRight = static_cast<int>(std::ceil(N*0.1));

        // Generar partículas fantasma a la izquierda
        for (int g = 0; g < numGhostsLeft; ++g) {
            double x = x_min - (g + 1) * dx + perturbation;
            std::array<double, 3> position = {x, 0.0, 0.0};
            particles.emplace_back(position, ghostPropsLeft->initialVelocity, mass_per_particle, 
                                   ghostPropsLeft->initialSpecificInternalEnergy, true, ghostPropsLeft);
            particles.back().h = h;
        } */

        // Generar partículas reales en el dominio
        for (int i = 0; i < N; ++i) {
            double x = x_min + (i + 0.5) * dx + perturbation;
            std::array<double, 3> position = {x, 0.0, 0.0};
            std::array<double, 3> velocity = {uniform_velocity, 0.0, 0.0};

            Particle particle(position, velocity, mass_per_particle, uniform_pressure / ((GAMMA - 1.0) * uniform_density));
            particle.density = uniform_density;
            particle.pressure = uniform_pressure;
            particle.h = h;

            particle.updatePressure(*eos);
            particle.updateSoundSpeed(*eos);

            particles.push_back(particle);
        }

        /* // Generar partículas fantasma a la derecha
        for (int g = 0; g < numGhostsRight; ++g) {
            double x = x_max + (g + 1) * dx + perturbation;
            std::array<double, 3> position = {x, 0.0, 0.0};
            particles.emplace_back(position, ghostPropsRight->initialVelocity, mass_per_particle, 
                                   ghostPropsRight->initialSpecificInternalEnergy, true, ghostPropsRight);
            particles.back().h = h;
        } */
    }
    break;

    }
}
