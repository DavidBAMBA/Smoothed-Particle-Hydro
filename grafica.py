import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def sod_exact_solution(x, t, gamma=1.4, x0=0.0):
    """
    Devuelve la solución analítica (rho, u, p, e_int) del Sod Shock Tube
    en las posiciones x y tiempo t. 
    La discontinuidad inicial se ubica en x0 (por defecto 0.0),
    y el dominio típico es [-1, 1].
    
    Estados (Sod clásico):
        Izquierda: rho=1.0, u=0.0, p=1.0
        Derecha:   rho=0.125, u=0.0, p=0.1
    """
    
    # --- 1) Estados iniciales ---
    rhoL, uL, pL = 1.0, 0.0, 1.0
    rhoR, uR, pR = 0.125, 0.0, 0.1
    
    # --- 2) Datos de la solución estrella (caso clásico) ---
    p_star = 0.30313
    u_star = 0.92745
    
    # Densidades estrella
    rho_starL = rhoL * (p_star/pL)**(1.0/gamma)
    gm = gamma
    factor_num = (p_star/pR + (gm-1.0)/(gm+1.0))
    factor_den = ((gm-1.0)/(gm+1.0)*p_star/pR + 1.0)
    rho_starR = rhoR * (factor_num/factor_den)
    
    # --- 3) Velocidades de sonido ---
    cL = np.sqrt(gamma * pL / rhoL) 
    c_starL = cL * (p_star/pL)**((gamma-1.)/(2.*gamma))
    
    # --- 4) Posiciones de las ondas ---
    # Rarefacción izquierda
    x_head   = x0 - cL * t
    x_tail   = x0 + (u_star - c_starL)* t
    
    # Contacto
    x_contact = x0 + u_star * t
    
    # Choque a la derecha (se conoce ~1.752 en Sod clásico)
    x_shock = x0 + 1.752 * t

    # --- 5) Vectores de salida ---
    rho_exact = np.zeros_like(x)
    u_exact   = np.zeros_like(x)
    p_exact   = np.zeros_like(x)
    
    # --- 6) Asignar solución por regiones ---
    for i, xx in enumerate(x):
        
        if xx < x_head:
            # Estado izquierdo
            rho_exact[i] = rhoL
            u_exact[i]   = uL
            p_exact[i]   = pL
        
        elif xx < x_tail:
            # Onda de rarefacción izquierda
            xi = (xx - x0)/t
            u_exact[i] = 2.0/(gamma+1.0)*(cL + xi)
            c_local = cL - 0.5*(gamma-1.0)*u_exact[i]
            rho_exact[i] = rhoL * (c_local/cL)**(2.0/(gamma-1.0))
            p_exact[i] = pL * (rho_exact[i]/rhoL)**gamma
        
        elif xx < x_contact:
            # Región estrella izquierda
            rho_exact[i] = rho_starL
            u_exact[i]   = u_star
            p_exact[i]   = p_star
        
        elif xx < x_shock:
            # Región estrella derecha
            rho_exact[i] = rho_starR
            u_exact[i]   = u_star
            p_exact[i]   = p_star
        
        else:
            # Estado derecho no perturbado
            rho_exact[i] = rhoR
            u_exact[i]   = uR
            p_exact[i]   = pR
    
    # Energía interna específica
    e_int_exact = p_exact / ((gamma - 1.0)*rho_exact)
    
    return rho_exact, u_exact, p_exact, e_int_exact


def blast_wave_exact_solution(x, t, gamma=1.4, x0=0.0):
    """
    Devuelve la solución analítica (rho, u, p, e_int) de una onda de explosión
    en las posiciones x y tiempo t. 
    La discontinuidad inicial se ubica en x0 (por defecto 0.0),
    y el dominio típico es [-1, 1].

    Estados iniciales (blast wave):
        Izquierda: rho=1.0, p=1000.0
        Derecha:   rho=1.0, p=0.1
    """

    # --- 1) Estados iniciales ---
    rhoL, uL, pL = 1.0, 0.0, 1000.0
    rhoR, uR, pR = 1.0, 0.0, 0.1
    
    # --- 2) Datos de la solución estrella ---
    # Se determinan utilizando las condiciones iniciales (shock-tube exact solver)
    p_star = 460.894  # Presión en la región estrella (determinada analíticamente o numéricamente)
    u_star = 19.5975  # Velocidad de la región estrella
    
    # Densidades estrella
    rho_starL = rhoL * (p_star / pL)**(1.0 / gamma)
    gm = gamma
    factor_num = (p_star / pR + (gm - 1.0) / (gm + 1.0))
    factor_den = ((gm - 1.0) / (gm + 1.0) * p_star / pR + 1.0)
    rho_starR = rhoR * (factor_num / factor_den)
    
    # --- 3) Velocidades de sonido ---
    cL = np.sqrt(gamma * pL / rhoL)
    c_starL = cL * (p_star / pL)**((gamma - 1.0) / (2.0 * gamma))
    cR = np.sqrt(gamma * pR / rhoR)
    
    # --- 4) Posiciones de las ondas ---
    # Rarefacción izquierda
    x_head = x0 - cL * t
    x_tail = x0 + (u_star - c_starL) * t
    
    # Contacto
    x_contact = x0 + u_star * t
    
    # Choque a la derecha
    shock_speed_right = np.sqrt((gamma + 1.0) * p_star / (2.0 * gamma * rhoR) + (gamma - 1.0) / (2.0 * gamma) * cR**2)
    x_shock = x0 + shock_speed_right * t

    # --- 5) Vectores de salida ---
    rho_exact = np.zeros_like(x)
    u_exact = np.zeros_like(x)
    p_exact = np.zeros_like(x)
    
    # --- 6) Asignar solución por regiones ---
    for i, xx in enumerate(x):
        
        if xx < x_head:
            # Estado izquierdo
            rho_exact[i] = rhoL
            u_exact[i] = uL
            p_exact[i] = pL
        
        elif xx < x_tail:
            # Onda de rarefacción izquierda
            xi = (xx - x0) / t
            u_exact[i] = 2.0 / (gamma + 1.0) * (cL + xi)
            c_local = cL - 0.5 * (gamma - 1.0) * u_exact[i]
            rho_exact[i] = rhoL * (c_local / cL)**(2.0 / (gamma - 1.0))
            p_exact[i] = pL * (rho_exact[i] / rhoL)**gamma
        
        elif xx < x_contact:
            # Región estrella izquierda
            rho_exact[i] = rho_starL
            u_exact[i] = u_star
            p_exact[i] = p_star
        
        elif xx < x_shock:
            # Región estrella derecha
            rho_exact[i] = rho_starR
            u_exact[i] = u_star
            p_exact[i] = p_star
        
        else:
            # Estado derecho
            rho_exact[i] = rhoR
            u_exact[i] = uR
            p_exact[i] = pR
    
    # Energía interna específica
    e_int_exact = p_exact / ((gamma - 1.0) * rho_exact)
    
    return rho_exact, u_exact, p_exact, e_int_exact


def plot_data(filename):
    # Leer datos
    data = pd.read_csv(filename, sep="\t")
    
    # Extraer tiempo y paso
    time = data['t'].iloc[0]
    step = os.path.splitext(os.path.basename(filename))[0].split('_')[-1]

    # Crear figura
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # --- Ajustar dominio analítico [-1, 1] ---
    x_analytic = np.linspace(-1.0, 1.0, 500)
    rho_an, u_an, p_an, eint_an = sod_exact_solution(x_analytic, time, gamma=1.4, x0=0.0)
    #rho_an, u_an, p_an, eint_an = blast_wave_exact_solution(x_analytic, time, gamma=5.0/3.0, x0=0.0)

    # Gráfico 1: Velocidad vx vs x
    axs[0, 0].scatter(data['x'], data['vx'], s=5, c='blue', label='SPH')
    axs[0, 0].plot(x_analytic, u_an, 'k-', label='Analítico')
    axs[0, 0].set_title(f'Velocidad vs Posición\nPaso {step}, Tiempo {time:.4f}')
    axs[0, 0].set_xlabel('x')
    axs[0, 0].set_ylabel('vx')
    axs[0, 0].set_xlim(-.5, .5)
    axs[0, 0].legend()

    # Gráfico 2: Presión vs x
    axs[0, 1].scatter(data['x'], data['P'], s=5, c='red', label='SPH')
    axs[0, 1].plot(x_analytic, p_an, 'k-', label='Analítico')
    axs[0, 1].set_title(f'Presión vs Posición\nPaso {step}, Tiempo {time:.4f}')
    axs[0, 1].set_xlabel('x')
    axs[0, 1].set_ylabel('Presión')
    axs[0, 1].set_xlim(-0.5, 0.5)
    axs[0, 1].legend()

    # Gráfico 3: Energía Interna vs x
    axs[1, 0].scatter(data['x'], data['u'], s=5, c='green', label='SPH')
    axs[1, 0].plot(x_analytic, eint_an, 'k-', label='Analítico')
    axs[1, 0].set_title(f'Energía Interna vs Posición\nPaso {step}, Tiempo {time:.4f}')
    axs[1, 0].set_xlabel('x')
    axs[1, 0].set_ylabel('Energía Interna')
    axs[1, 0].set_xlim(-.5, .5)
    axs[1, 0].legend()

    # Gráfico 4: Densidad vs x
    axs[1, 1].scatter(data['x'], data['rho'], s=5, c='purple', label='SPH')
    axs[1, 1].plot(x_analytic, rho_an, 'k-', label='Analítico')
    axs[1, 1].set_title(f'Densidad vs Posición\nPaso {step}, Tiempo {time:.4f}')
    axs[1, 1].set_xlabel('x')
    axs[1, 1].set_ylabel('Densidad')
    axs[1, 1].set_xlim(-.5, .5)
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig(f'results_step_{step}.png')
    plt.close()


def main():
    csv_files = glob.glob('outputs/output_step_*.csv')
    csv_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))
    
    os.makedirs('plots', exist_ok=True)
    os.chdir('plots')
    
    for filename in csv_files:
        print(f'Procesando {filename}...')
        plot_data(os.path.join('..', filename))
    
    print('Gráficos generados exitosamente en la carpeta "plots".')


if __name__ == '__main__':
    main()
