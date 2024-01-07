import math
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#from plotly.subplots import make_subplots
"""Programma pendolo doppio senza attrito o forze esterne armoniche; Tentativo mio di runge kutta applicato"""
"""  
O1' = ω1
O2' = ω2
ω1'=  	-g (2 m1 + m2) np.sin O1 - m2 g np.sin(O1 - 2 O2) - 2 np.sin(O1 - O2) m2 (ω22 l2 + ω12 L1 np.cos(O1 - O2))
L1 (2 m1 + m2 - m2 np.cos(2 O1 - 2 O2))
ω2'=  	2 np.sin(O1-O2) (ω12 L1 (m1 + m2) + g(m1 + m2) np.cos O1 + ω22 l2 m2 np.cos(O1 - O2))
l2 (2 m1 + m2 - m2 np.cos(2 O1 - 2 O2)) 
"""
def acc(O1, O2, dO1, dO2, g, l1, l2, m1, m2):
 a_1= -g*(2*m1 + m2)*np.sin(O1) - m2*g*np.sin(O1 - 2*O2) - 2*np.sin(O1 - O2)*m2*(dO2**2 * l2 + dO1**2*l1*np.cos(O1 - O2))/(l1*(2*m1 + m2 - m2*np.cos(2*O1 - 2*O2)))
 a_2=  2*np.sin(O1-O2)*(dO1**2 * l1*(m1 + m2) + g*(m1 + m2)*np.cos(O1) + dO2**2 * l2*m2*np.cos(O1 - O2))/(l2*(2*m1 + m2 - m2*np.cos(2*O1 - 2*O2))) 
 return a_1, a_2

def metodo_euleriano(O1, O2, dO1, dO2, dt, g, l1, l2, m1, m2):
    a_1, a_2 = acc(O1, O2, dO1, dO2, g, l1, l2, m1, m2)
    dO1_nuovo = dO1 + a_1 * dt
    dO2_nuovo = dO2 + a_2 * dt
    O1_nuovo = O1 + dO1_nuovo * dt
    O2_nuovo = O2 + dO2_nuovo * dt
    return O1_nuovo, O2_nuovo, dO1_nuovo, dO2_nuovo, a_1, a_2
def kt4(O1, O2, dO1, dO2, dt, g, l1, l2, m1, m2):
    a1_k1, a2_k1 = acc(O1, O2, dO1, dO2, g, l1, l2, m1, m2)
    k1_1W = a1_k1 * dt
    k1_2W = a2_k1 * dt
    k1_1O = dO1 * dt
    k1_2O = dO2 * dt

    # Calcolo k2
    O1_k2 = O1 + k1_1O * 0.5
    O2_k2 = O2 + k1_2O * 0.5
    dO1_k2 = dO1 + k1_1W * 0.5
    dO2_k2 = dO2 + k1_2W * 0.5
    a1_k2, a2_k2 = acc(O1_k2, O2_k2, dO1_k2, dO2_k2, g, l1, l2, m1, m2)
    k2_1W = a1_k2 * dt
    k2_2W = a2_k2 * dt
    k2_1O = dO1_k2 *dt
    k2_2O = dO2_k2 *dt

    # Calcolo k3
    O1_k3 = O1 + k2_1O * 0.5
    O2_k3 = O2 + k2_2O * 0.5
    dO1_k3 = dO1 + k2_1W * 0.5
    dO2_k3 = dO2 + k2_2W * 0.5
    a1_k3, a2_k3 = acc(O1_k3, O2_k3, dO1_k3, dO2_k3, g, l1, l2, m1, m2)
    k3_1W = a1_k3 * dt
    k3_2W = a2_k3 * dt
    k3_1O = dO1_k3 * dt
    k3_2O = dO2_k3 * dt

    # Calcolo k4
    O1_k4 = O1 + k3_1O
    O2_k4 = O2 + k3_2O
    dO1_k4 = dO1 + k3_1W
    dO2_k4 = dO2 + k3_2W
    a1_k4, a2_k4 = acc(O1_k4, O2_k4, dO1_k4, dO2_k4, g, l1, l2, m1, m2)
    k4_1W = a1_k4 * dt
    k4_2W = a2_k4 * dt
    k4_1O = dO1_k4 *dt
    k4_2O = dO2_k4 *dt

    # Aggiornamento finale
    W1_new = dO1 + (k1_1W + 2 * k2_1W + 2 * k3_1W + k4_1W) / 6
    W2_new = dO2 + (k1_2W + 2 * k2_2W + 2 * k3_2W + k4_2W) / 6
    O1_new = O1 + (k1_1O + 2 * k2_1O + 2 * k3_1O + k4_1O) / 6
    O2_new = O2 + (k1_2O + 2 * k2_2O + 2 * k3_2O + k4_2O) / 6

    return O1_new, O2_new, W1_new, W2_new

m1, m2 = 0.4, 1.6  
g,l1,l2,O1_deg,O2_deg=9.8127, 0.8, 0.3, float(input('angolo massa 1 = ')), float(input('Angolo massa 2 = ')) #starter e costanti
O1=np.deg2rad(O1_deg)#angoli starter massa 1 
O2=np.deg2rad(O2_deg)#angoli starter massa 2
O1_eul,O2_eul=O1, O2 #start Eulero stessi values
t, dt = 0, 0.01 # time start, interval
dO1, dO2 = 0, 0   #velocita angolare iniziale masse
dO1_eul,dO2_eul=dO1, dO2

#inizio loop; come salvo?
import matplotlib.pyplot as plt
import matplotlib.animation as animation
lista_O1_euler, lista_O2_euler = [], []
lista_O1_rk4,lista_O2_rk4=[],[]
lista_omega1_kt4, lista_omega2_kt4=[],[]
lista_accelerazione_alfa1_kt4,lista_accelerazione_alfa2_kt4=[],[]
lista_ascissa_tempo=[]
while t <= 50:
    lista_ascissa_tempo.append(t)
    O1, O2, dO1, dO2 = kt4(O1, O2, dO1, dO2, dt, g, l1, l2, m1, m2)
    O1_eul,O2_eul,dO1_eul,dO2_eul,a_1_eul,a_2_eul=metodo_euleriano(O1_eul, O2_eul, dO1_eul, dO2_eul, dt, g, l1, l2, m1, m2)
    
    lista_O1_euler.append(O1_eul);lista_O2_euler.append(O2_eul)
    lista_O1_rk4.append(O1); lista_O2_rk4.append(O2)
    lista_omega1_kt4.append(dO1);lista_omega2_kt4.append(dO2)
    #lista_accelerazione_alfa1_kt4.append(a_1);lista_acceler_kt4lista_accelerazione_alfa2_kt4azione_alfa2.append(a_2)
    t+=dt

######################################################################################################################
#animazione inizio
def calcola_posizioni(lista_O1_rk4, lista_O2_rk4, l1, l2):
    #calcola posizione delle masse dei pendoli, dalle liste, così può fare il disegno
    #e animarle, con matplot lib
    x1 = l1 * np.sin(lista_O1_rk4)
    y1 = -l1 * np.cos(lista_O1_rk4)
    x2 = x1 + l2 * np.sin(lista_O2_rk4)
    y2 = y1 - l2 * np.cos(lista_O2_rk4)
    return x1, y1, x2, y2
max_range = (l1 + l2) * 1.1  # La massima estensione del pendolo, con un piccolo margine

fig_euler, ax_euler = plt.subplots()
line_euler, = ax_euler.plot([], [], 'o-', lw=2, alpha=0.6, label='Euler')
ax_euler.set_xlim(-max_range, max_range)
ax_euler.set_ylim(-max_range, max_range)
ax_euler.legend()


# Figura per il metodo RK4
fig_rk4, ax_rk4 = plt.subplots()
line_rk4, = ax_rk4.plot([], [], 'o-', lw=2, alpha=0.6, color='orange', label='RK4')
ax_rk4.set_xlim(-max_range, max_range)
ax_rk4.set_ylim(-max_range, max_range)
ax_rk4.legend()
def init_euler():
    line_euler.set_data([], [])
    #trail_euler.set_data([], [])
    return line_euler, 

# Funzione di inizializzazione per RK4
def init_rk4():
    line_rk4.set_data([], [])
    #trail_rk4.set_data([], [])
    return line_rk4, 
# Funzione di animazione per Eulero
def animate_euler(i):
    x1, y1, x2, y2 = calcola_posizioni(lista_O1_euler[i], lista_O2_euler[i], l1, l2)
    line_euler.set_data([0, x1, x2], [0, y1, y2])
    return line_euler,

# Funzione di animazione per RK4
def animate_rk4(i):
    x1, y1, x2, y2 = calcola_posizioni(lista_O1_rk4[i], lista_O2_rk4[i], l1, l2)
    line_rk4.set_data([0, x1, x2], [0, y1, y2])
    return line_rk4,

# Creazione delle animazioni
ani_euler = animation.FuncAnimation(fig_euler, animate_euler, init_func=init_euler, frames=len(lista_ascissa_tempo), interval=20)
ani_rk4 = animation.FuncAnimation(fig_rk4, animate_rk4, init_func=init_rk4, frames=len(lista_ascissa_tempo), interval=20)


plt.show()
plt.show()
"""
fig, ax = plt.subplots()
line, = ax.plot([], [], 'o-', lw=2, alpha=0.6)
trail, = ax.plot([], [], '-', lw=1, alpha=0.5*dt)

max_range = (l1 + l2)  # La massima estensione del pendolo
ax.set_xlim(-max_range, max_range)
ax.set_ylim(-max_range, max_range)
# Funzione di inizializzazione
def init():
    line.set_data([], [])
    trail.set_data([], [])
    return line,trail
x_trail, y_trail = [], []
# Funzione di animazione
def animate(i):
    x1, y1, x2, y2 = calcola_posizioni(lista_O1_euler[i], lista_O2_euler[i])
    line.set_data([0, x1, x2], [0, y1, y2])
    x_trail.append(x2)
    y_trail.append(y2)
    trail.set_data(x_trail, y_trail)
    return line, trail

ani = animation.FuncAnimation(fig, animate, frames=len(lista_ascissa_tempo), init_func=init, blit=True, interval=20)
plt.show()
"""
#grafico rk4
plt.figure()
plt.plot(lista_ascissa_tempo, lista_O1_rk4, label='O1 RK4')
plt.plot(lista_ascissa_tempo, lista_O2_rk4, label='O2 RK4')
plt.xlabel('Tempo (s)')
plt.ylabel('Angolo (rad)')
plt.title('Angoli in Funzione del Tempo (RK4)')
plt.legend()
plt.show()
#grafico Euler
plt.figure()
plt.plot(lista_ascissa_tempo, lista_O1_euler, label='O1 euler')
plt.plot(lista_ascissa_tempo, lista_O2_euler, label='O2 euler')
plt.xlabel('Tempo (s)')
plt.ylabel('Angolo (rad)')
plt.title('Angoli in Funzione del Tempo (eulero)')
plt.legend()
plt.show()