#Programma Pendulum easy_senza attrito o altro con iterazione
#thks DOT-phy: sono Riccardo Venturi e ho preso i riferimenti da lui
#Newtoniano
#comandi graphico
from math import sin, radians
from vpython import graph, gcurve, color, rate  # Ensure you have the right imports
g1=graph(xtitle="t",ytitle="eta",width=500,height=250)
f1=gcurve(color=color.red, dot=True)
f_H=gcurve(color=color.green,dot=True)
#f2=gcurve(color=color.blue)
#f3=gcurve(color=color.yellow)
###Calcolato con logiche NEWtoniane###
"""ho preso le forze sulla massa e fatto il cambio in coordinate polari, avrò
   r_vector(→)=r*r_versor(^); d(r_→)/dt = r_dot * r^ + r*eta_dot*eta^; 
   dd(r_vector)/(dt**2) = r^*(r_ddot -2r*eta_dot**2) + eta^*(2eta_dot*r_dot + eta_ddot*r) """
#uso le equazioni per il metodo iterattivo, secondi i #↓dati↓ iniziali cammbiano le eq
#considero corda inelastica e quindi rddot=0 e rdot=0: 

g=9.812 #... m/s**2
m=0.5 # kg
eta=float(input("angolo scelto in ° "))
eta_rad=radians(eta)
etaDot=0 #velocità angolare iniziale rad/sec !! posso fare input
R=0.4 #raggio m

t0=0 # sec tempo start
dt=0.001 # timing; arco temporale

if eta == 0:
    print("angle is 0, no movement bro1!")
else:
    while t0 <= 3:
        rate(1000) #rate() function in VPython is crucial for controlling 
        #the speed of animations and simulations. It determines how many 
        #iterations of a loop are executed per second, thus controlling the time flow in the simulation
        etaDdot = -(g/R)*sin(eta_rad) #1 accelerazione angolar
        etaDot += etaDdot*dt #2 velocità
        eta_rad=eta_rad+etaDot*dt#3 angolo finale 
        t0+=dt
        f1.plot(t0,eta_rad)       
        #f2.plot(t0,etaDot)
        #f3.plot(t0,etaDdot)
 
"""Posso farlo con Lagrange e il path di E etc...: prendo coordinate, cartesiane per esempio,K=cinetica 0.5*m*v**2;U=pot m*g*h
   K=0.5*m*[(x**2+y**2)=R**2*etadot**2];U=-mg(R-Rcos(eta)); faccio -dE/deta, d/dt(dE/d(etadot)) e ottengo formula di èrima """

"""Hamiltonian now and compare; start P_(impulso)_dot=-gsin(eta)Rm → P2=P1+P_dot*dt 
# ottengo Pdot→P→ eta_dot=P/mR**2 → eta2=eta1+eta_dot*dt"""

eta_finale = eta_rad
etaDot_finale = etaDot
P = etaDot_finale * m * R**2  # Calcolo dell'impulso iniziale basato sulla velocità angolare finale del metodo Newtoniano

# Metodo Hamiltoniano
while t0 < 6:
    rate(100)
    pdot = -m * g * R * sin(eta_finale)
    P += pdot * dt
    etaDot = P / (m * R**2)
    eta_finale += etaDot * dt
    t0 += dt
    f_H.plot(t0, eta_finale)