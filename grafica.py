import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import newton

# ======================================================================
# Funciones de solución exacta
# ======================================================================
def exact_riemann_solution(rhoL, uL, pL, rhoR, uR, pR, gamma=1.4, x0=0.0, t=0.2):
    """
    Solución exacta del problema de Riemann para condiciones iniciales arbitrarias.
    Devuelve un diccionario con todos los parámetros relevantes de la solución.
    """
    # Constantes del gas
    gm = gamma
    gm1 = gm - 1.0
    gm_p1 = gm + 1.0

    # Velocidades de sonido iniciales
    cL = np.sqrt(gm * pL / rhoL)
    cR = np.sqrt(gm * pR / rhoR)

    # Función para ecuación de p_star
    def pressure_star_eq(p_star):
        # Onda izquierda
        if p_star > pL:
            AL = 2.0 / (gm_p1 * rhoL)
            BL = gm1/gm_p1 * pL
            fL = (p_star - pL) * np.sqrt(AL/(p_star + BL))
        else:
            fL = 2*cL/gm1 * ((p_star/pL)**((gm1)/(2*gm)) - 1)
        
        # Onda derecha
        if p_star > pR:
            AR = 2.0 / (gm_p1 * rhoR)
            BR = gm1/gm_p1 * pR
            fR = (p_star - pR) * np.sqrt(AR/(p_star + BR))
        else:
            fR = 2*cR/gm1 * ((p_star/pR)**((gm1)/(2*gm)) - 1)
        
        return fL + fR + (uR - uL)

    # Resolver para p_star
    p_guess = 0.5*(pL + pR)
    p_star = newton(pressure_star_eq, p_guess)

    # Calcular u_star
    if p_star > pL:
        AL = 2.0/(gm_p1 * rhoL)
        BL = gm1/gm_p1 * pL
        u_star = uL + (p_star - pL)*np.sqrt(AL/(p_star + BL))
    else:
        u_star = uL + 2*cL/gm1*(1 - (p_star/pL)**((gm1)/(2*gm)))

    # Calcular densidades estrella
    if p_star > pL:
        rho_starL = rhoL*((p_star/pL) + (gm1)/gm_p1)/((gm1)/gm_p1*(p_star/pL) + 1)
    else:
        rho_starL = rhoL*(p_star/pL)**(1/gm)
    
    if p_star > pR:
        rho_starR = rhoR*((p_star/pR) + (gm1)/gm_p1)/((gm1)/gm_p1*(p_star/pR) + 1)
    else:
        rho_starR = rhoR*(p_star/pR)**(1/gm)

    # Calcular velocidades de las ondas
    wave_speeds = {}
    if p_star <= pL:
        c_starL = cL*(p_star/pL)**((gm1)/(2*gm))
        wave_speeds['x_head'] = x0 - cL*t
        wave_speeds['x_tail'] = x0 + (u_star - c_starL)*t
    else:
        shock_speed_L = uL + cL*np.sqrt((gm_p1/(2*gm))*(p_star/pL) + gm1/(2*gm))
        wave_speeds['x_shock_L'] = x0 + shock_speed_L*t

    wave_speeds['x_contact'] = x0 + u_star*t

    if p_star <= pR:
        c_starR = cR*(p_star/pR)**((gm1)/(2*gm))
        wave_speeds['x_tail_R'] = x0 + (u_star + c_starR)*t
    else:
        shock_speed_R = uR + cR*np.sqrt((gm_p1/(2*gm))*(p_star/pR) + gm1/(2*gm))
        wave_speeds['x_shock_R'] = x0 + shock_speed_R*t

    return {
        'p_star': p_star,
        'u_star': u_star,
        'rho_starL': rho_starL,
        'rho_starR': rho_starR,
        'cL': cL,
        'cR': cR,
        **wave_speeds
    }

def general_shock_tube_solution(x, t, rhoL, uL, pL, rhoR, uR, pR, gamma=1.4, x0=0.0):
    """
    Solución analítica general para cualquier condición inicial.
    """
    sol = exact_riemann_solution(rhoL, uL, pL, rhoR, uR, pR, gamma, x0, t)
    
    # Extraer parámetros
    p_star = sol['p_star']
    u_star = sol['u_star']
    rho_starL = sol['rho_starL']
    rho_starR = sol['rho_starR']
    cL = sol['cL']
    cR = sol['cR']
    
    # Determinar posiciones de las ondas
    if 'x_shock_L' in sol:
        x_head = sol['x_shock_L']
        x_tail = sol['x_shock_L']
    else:
        x_head = sol['x_head']
        x_tail = sol['x_tail']
    
    x_contact = sol['x_contact']
    
    if 'x_shock_R' in sol:
        x_shock_R = sol['x_shock_R']
    else:
        x_shock_R = sol['x_tail_R']

    # Inicializar arrays
    rho = np.zeros_like(x)
    u = np.zeros_like(x)
    p = np.zeros_like(x)

    # Asignar estados
    for i, xi in enumerate(x):
        if xi < x_head:
            rho[i] = rhoL
            u[i] = uL
            p[i] = pL
        elif xi < x_tail:
            if p_star <= pL:
                c = (xi - x0)/t
                u[i] = 2/(gamma+1)*(c + (gamma-1)/2*uL + cL)
                c_local = cL - 0.5*(gamma-1)*u[i]
                rho[i] = rhoL*(c_local/cL)**(2/(gamma-1))
                p[i] = pL*(rho[i]/rhoL)**gamma
        elif xi < x_contact:
            rho[i] = rho_starL
            u[i] = u_star
            p[i] = p_star
        elif xi < x_shock_R:
            rho[i] = rho_starR
            u[i] = u_star
            p[i] = p_star
        else:
            rho[i] = rhoR
            u[i] = uR
            p[i] = pR

    e_int = p/((gamma-1.0)*rho)
    return rho, u, p, e_int

# ======================================================================
# Visualización de datos
# ======================================================================
def plot_data(filename):
    """Genera gráficos comparando los datos SPH con la solución analítica."""
    # Leer datos SPH
    data = pd.read_csv(filename, sep='\t')
    
    # Parámetros de la simulación
    time = data['t'].iloc[0]
    step = os.path.splitext(os.path.basename(filename))[0].split('_')[-1]
    
    # Condiciones iniciales personalizadas
    rhoL, uL, pL = 1.0, 0.0, 1.0
    rhoR, uR, pR = 0.125, 0.0, 0.1
    gamma = 1.4  # Coincidir con la simulación

    # Calcular solución analítica
    x_analytic = np.linspace(-1.0, 1.0, 1000)
    rho_an, u_an, p_an, eint_an = general_shock_tube_solution(
        x_analytic, time, rhoL, uL, pL, rhoR, uR, pR, gamma
    )

    # Separar partículas reales y fantasma
    real = data[data['IsGhost'] == 0]
    ghosts = data[data['IsGhost'] == 1]

    # Crear figura
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Gráfico 1: Velocidad
    axs[0,0].scatter(real['x'], real['vx'], s=2, c='r', label='SPH Real')
    axs[0,0].scatter(ghosts['x'], ghosts['vx'], s=2, c='k', label='SPH Ghost')
    axs[0,0].plot(x_analytic, u_an, 'k-', lw=2, label='Analítico')
    axs[0,0].set_title(f'Velocidad (t={time:.4f})')
    axs[0,0].set_xlim(-0.7, 0.7)
    axs[0,0].legend()

    # Gráfico 2: Presión
    axs[0,1].scatter(real['x'], real['P'], s=2, c='r')
    axs[0,1].scatter(ghosts['x'], ghosts['P'], s=2, c='k')
    axs[0,1].plot(x_analytic, p_an, 'k-', lw=2)
    axs[0,1].set_title('Presión')
    axs[0,1].set_xlim(-0.7, 0.7)

    # Gráfico 3: Energía Interna
    axs[1,0].scatter(real['x'], real['u'], s=2, c='r')
    axs[1,0].scatter(ghosts['x'], ghosts['u'], s=2, c='k')
    axs[1,0].plot(x_analytic, eint_an, 'k-', lw=2)
    axs[1,0].set_title('Energía Interna')
    axs[1,0].set_xlim(-0.7, 0.7)

    # Gráfico 4: Densidad
    axs[1,1].scatter(real['x'], real['rho'], s=2, c='r')
    axs[1,1].scatter(ghosts['x'], ghosts['rho'], s=2, c='k')
    axs[1,1].plot(x_analytic, rho_an, 'k-', lw=2)
    axs[1,1].set_title('Densidad')
    axs[1,1].set_xlim(-0.7, 0.7)

    plt.tight_layout()
    plt.savefig(f'plot_step_{step}.png')
    plt.close()

# ======================================================================
# Función principal
# ======================================================================
def main():
    # Procesar todos los archivos de salida
    csv_files = sorted(glob.glob('outputs/output_step_*.csv'), 
                      key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    # Crear directorio para gráficos
    os.makedirs('plots', exist_ok=True)
    os.chdir('plots')
    
    # Generar gráficos
    for file in csv_files:
        print(f'Procesando: {file}')
        plot_data(os.path.join('..', file))
    
    print("Todos los gráficos generados en la carpeta 'plots'")

if __name__ == "__main__":
    main()