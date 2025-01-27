#include "DensityUpdater.h"
#include <iostream>
#include <algorithm>
#include "MathUtils.h"
#include "Kernel.h"
#include <omp.h>

// ========================
// Constructor
// ========================
DensityUpdater::DensityUpdater(double eta, double tol, bool use_fixed_h, double fixed_h)
    : eta_(eta), tol_(tol), use_fixed_h_(use_fixed_h), fixed_h_(fixed_h) {}

// ========================
// updateDensity: Calcula (y actualiza) la densidad de la partícula
// ========================
void DensityUpdater::updateDensity(Particle& particle, const std::vector<Particle>& neighbors, const Kernel& kernel) {
    if (use_fixed_h_) {
        particle.h = fixed_h_;
    } else {
        double h_guess = (particle.h == 0.0) 
            ? eta_ * (particle.mass / ((particle.density > 0.0) ? particle.density : 1.0))
            : particle.h;
        //std::cout << "[updateDensity] h_guess: " << h_guess << std::endl;
        // Usa findConvergedH para obtener un h refinado
        double h_found = findConvergedH(particle, neighbors, h_guess, kernel);
        if (h_found > 0.0) {
            particle.h = h_found;
        } else {
            std::cerr << "[updateDensity] No converge el método iterativo. Se mantiene el valor de h." << std::endl;
        }
    }
    particle.density = calculateDensity(particle, neighbors, particle.h, kernel);
    particle.Omega = calculateOmega(particle, neighbors,particle.h, particle.density, kernel);
}

// ========================
// calculateDensity: Calcula la densidad SPH a partir de las contribuciones de los vecinos
// ========================
double DensityUpdater::calculateDensity(const Particle& particle, const std::vector<Particle>& neighbors, double h, const Kernel& kernel) const {
    double rho = 0.0;
    #pragma omp parallel for reduction(+:rho)
    for (size_t i = 0; i < neighbors.size(); ++i) {
        const auto& neighbor = neighbors[i];
        std::array<double, 3> r_ab = {
            particle.position[0] - neighbor.position[0],
            particle.position[1] - neighbor.position[1],
            particle.position[2] - neighbor.position[2]
        };
        double r_ab_norm = MathUtils::vectorNorm(r_ab);
        double W_val = kernel.W(r_ab_norm, h);
        rho += neighbor.mass * W_val;
    }
    return rho;
}

// ========================
// calculateOmega: Calcula Omega (factor de corrección)
// ========================
double DensityUpdater::calculateOmega(const Particle& particle, const std::vector<Particle>& neighbors, double h, double rho, const Kernel& kernel) const {
    double sum_m_dWdh = 0.0;

    //#pragma omp parallel for reduction(+:sum_m_dWdh)
    for (const auto& neighbor : neighbors) {
        std::array<double, 3> r_ab = {
            particle.position[0] - neighbor.position[0],
            particle.position[1] - neighbor.position[1],
            particle.position[2] - neighbor.position[2]
        };
        double r_ab_norm = MathUtils::vectorNorm(r_ab);
        if (r_ab_norm > 3.0 * h)
            continue;
        double dWdh_val = kernel.dWdh(r_ab_norm, h);
        if (std::isfinite(dWdh_val))
            sum_m_dWdh += neighbor.mass * dWdh_val;
    }
    sum_m_dWdh *= h / rho;
    double omega = 1.0 + sum_m_dWdh;
    return std::max(omega, 1e-7);
}

// ========================
// computeFunctionAndDerivative: Centraliza el cálculo de f(h) y df/dh
// f(h) = (eta*m/h) - rho_sph(h)
// ========================
std::pair<double, double> DensityUpdater::computeFunctionAndDerivative(const Particle& particle, const std::vector<Particle>& neighbors, double h, const Kernel& kernel) const {
    double rho_sph = calculateDensity(particle, neighbors, h, kernel);
    double h_inv = 1.0 / h;
    double rho_h = eta_ * particle.mass * h_inv;
    double f_h = rho_h - rho_sph;
    double omega_val = calculateOmega(particle, neighbors, h, rho_h, kernel);
    double df = -rho_h * omega_val * h_inv;
    return {f_h, df};
}

// ========================
// newtonRaphson: Método de Newton–Raphson (utiliza computeFunctionAndDerivative)
// ========================
double DensityUpdater::newtonRaphson(const Particle& particle, const std::vector<Particle>& neighbors, double h_guess, const Kernel& kernel) const {
    double h_old = h_guess;
    const int max_iter = 50;
    int newtonIter = 0;
    for (int i = 0; i < max_iter; ++i) {
        newtonIter++;
        auto [f, df] = computeFunctionAndDerivative(particle, neighbors, h_old, kernel);
        //std::cout << "[Newton] Iteración " << newtonIter << ", h = " << h_old  << ", f = " << f << ", df = " << df << std::endl;
        if (std::fabs(df) < 1e-20) {
            std::cerr << "[Newton] Derivada nula." << std::endl;
            break;
        }
        double h_new = h_old - f / df;
        // Restringir el salto para evitar cambios bruscos: usar h_guess como referencia
        double ratio_min = 0.9, ratio_max = 1.1;
        h_new = std::max(ratio_min * h_guess, std::min(h_new, ratio_max * h_guess));
        if (!std::isfinite(h_new) || h_new <= 0.0) {
            std::cerr << "[Newton] h_new no finito o negativo." << std::endl;
            break;
        }
        double rel_change = std::fabs((h_new - h_old) / h_old);
        if (rel_change < tol_) {
            //std::cout << "[Newton] Convergencia en iteración " << newtonIter << std::endl;
            return h_new;
        }
        h_old = h_new;
    }
    return 0.0; // No convergió
}

// ========================
// bisectionMethod: Función auxiliar de bisección para reducir el intervalo donde f(h) cambia de signo
// ========================
double DensityUpdater::bisectionMethod(const Particle& particle, const std::vector<Particle>& neighbors, 
                                         double h_left, double h_right, const Kernel& kernel) const {
    const int max_bisection = 50;
    int bisectionIter = 0;
    double h_mid = 0.5 * (h_left + h_right);
    auto eval_f = [&](double hh) -> double {
        double rho_sph = calculateDensity(particle, neighbors, hh, kernel);
        double rho_h = eta_ * particle.mass / hh;
        return rho_h - rho_sph;
    };
    double f_left = eval_f(h_left);
    double f_right = eval_f(h_right);
    double f_mid = eval_f(h_mid);
    for (bisectionIter = 0; bisectionIter < max_bisection; ++bisectionIter) {
        //std::cout << "[Bisección] Iteración " << bisectionIter+1 << ", h_mid = " << h_mid << ", f_mid = " << f_mid << std::endl;
        if (std::fabs(f_mid) < tol_) {
            //std::cout << "[Bisección] Convergencia en iteración " << bisectionIter+1 << std::endl;
            break;
        }
        if (f_left * f_mid < 0.0) {
            h_right = h_mid;
            f_right = f_mid;
        } else {
            h_left = h_mid;
            f_left = f_mid;
        }
        h_mid = 0.5 * (h_left + h_right);
        f_mid = eval_f(h_mid);
        double intervalSize = h_right - h_left;
        if (intervalSize < tol_ * h_mid) {
            //std::cout << "[Bisección] Intervalo reducido en iteración " << bisectionIter+1 << std::endl;
            break;
        }
    }
    return h_mid;
}

// ========================
// findConvergedH: Función híbrida que realiza bracketing, luego bisección (usando bisectionMethod)
// y finalmente refina la raíz llamando a newtonRaphson.
// ========================
double DensityUpdater::findConvergedH(const Particle& particle, const std::vector<Particle>& neighbors, double h_guess, const Kernel& kernel) const {
    //std::cout << "[findConvergedH] h_guess: " << h_guess << std::endl;
    // Limite máximo para h (para evitar expansión excesiva)
    double h_max = 10 * h_guess;
    
    auto eval_f = [&](double hh) -> double {
        double rho_sph = calculateDensity(particle, neighbors, hh, kernel);
        double rho_h = eta_ * particle.mass / hh;
        return rho_h - rho_sph;
    };

    // 1) Bracketing: Buscar un intervalo [h_left, h_right] en el que f(h) cambie de signo.
    double h_left = std::max(h_guess, 1e-14);
    double f_left = eval_f(h_left);
    if (std::fabs(f_left) < tol_) {
        //std::cout << "[Bracketing] Convergencia directa con h = " << h_left << std::endl;
        return h_left;
    }
    const int max_bracket_steps = 20;
    // Expandir hacia abajo
    for (int i = 0; i < max_bracket_steps; ++i) {
        double testH = 0.5 * h_left;
        if (testH < 1e-14) break;
        double f_test = eval_f(testH);
        if (f_test * f_left < 0.0) {
            h_left = testH;
            f_left = f_test;
            //std::cout << "[Bracketing] Cambio de signo hacia abajo en iteración " << i+1         << ", h_left = " << h_left << std::endl;
            break;
        } else {
            h_left = testH;
            f_left = f_test;
        }
    }
    // Expandir hacia arriba
    double h_right = std::max(h_guess, 1e-14);
    double f_right = eval_f(h_right);
    for (int i = 0; i < max_bracket_steps; ++i) {
        double testH = 2.0 * h_right;
        if (testH > h_max) {
            h_right = h_max;
            f_right = eval_f(h_right);
            //std::cout << "[Bracketing] h_right alcanzó el máximo permitido: " << h_right << std::endl;
            break;
        }
        double f_test = eval_f(testH);
        if (f_test * f_right < 0.0) {
            h_right = testH;
            f_right = f_test;
            //std::cout << "[Bracketing] Cambio de signo hacia arriba en iteración " << i+1          << ", h_right = " << h_right << std::endl;
            break;
        } else {
            h_right = testH;
            f_right = f_test;
        }
    }
    if (f_left * f_right > 0.0) {
        std::cerr << "[Bracketing] No se detecta cambio de signo en f(h) en el rango permitido [h_guess, h_max]." << std::endl;
        return 0.0;
    }
    //std::cout << "[Bracketing] Intervalo final: h_left = " << h_left << ", h_right = " << h_right << std::endl;
    
    // 2) Bisección (se usa bisectionMethod definida aparte)
    double h_mid = bisectionMethod(particle, neighbors, h_left, h_right, kernel);
    //std::cout << "[findConvergedH] h_mid (después de bisección) = " << h_mid << std::endl;
    
    // 3) Newton–Raphson: Refinar la raíz usando el método centralizado.
    //std::cout << "[findConvergedH] Usando h_mid obtenido de bisección: " << h_mid << std::endl;
    double h_newton = newtonRaphson(particle, neighbors, h_mid, kernel);
    //std::cout << "[findConvergedH] Convergencia final con Newton: h = " << h_newton << std::endl;
    
    if (h_newton > 0.0) {
        particle.conv_h = "newton";
    } else {
        particle.conv_h = "no-converged";
    }

    return h_newton;
}
