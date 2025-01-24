#include "Boundaries.h"
#include <algorithm> // Añadir esta línea


Boundaries::Boundaries(double x_min, double x_max, BoundaryType type)
    : x_min_(x_min), x_max_(x_max), type_(type) {}

void Boundaries::apply(std::vector<Particle>& particles) {
    switch (type_) {
        case BoundaryType::PERIODIC:
            applyPeriodic(particles);
            break;
        case BoundaryType::FIXED:
            applyFixed(particles);
            break;
        case BoundaryType::OPEN:
            applyOpen(particles);
            break;
    }
}

void Boundaries::applyPeriodic(std::vector<Particle>& particles) {
    for (auto& p : particles) {
        if (p.position[0] < x_min_) {
            p.position[0] = x_max_ - (x_min_ - p.position[0]);
        } else if (p.position[0] > x_max_) {
            p.position[0] = x_min_ + (p.position[0] - x_max_);
        }
    }
}

void Boundaries::applyFixed(std::vector<Particle>& particles) {
    for (auto& p : particles) {
        if (p.position[0] < x_min_) {
            p.position[0] = x_min_;
            p.velocity[0] = 0.0;
        } else if (p.position[0] > x_max_) {
            p.position[0] = x_max_;
            p.velocity[0] = 0.0;
        }
    }
}

void Boundaries::applyOpen(std::vector<Particle>& particles) {
    // En condiciones abiertas, si una partícula sale del rango,
    // puede eliminarse o simplemente no actualizarse.
    // Aquí las removeremos del vector.
    particles.erase(std::remove_if(particles.begin(), particles.end(),
        [&](const Particle& p) {
            return (p.position[0] < x_min_ || p.position[0] > x_max_);
        }),
        particles.end());
}
