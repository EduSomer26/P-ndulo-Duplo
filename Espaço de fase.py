import matplotlib.pyplot as plt
import sympy as sym
import numpy as np
from scipy.integrate import odeint
from matplotlib import animation
from matplotlib.animation import PillowWriter

m1,m2,l1,l2,g,t  = sym.symbols('m1 m2 l1 l2 g t')          #definindo os símbolos
the1,the2 = sym.symbols(r'theta1 theta2',cls=sym.Function) #colocando os ângulos em função do tempo

'''------------------------Definindo as derivadas de theta-------------------------------'''
the1 = the1(t)
the2 = the2(t)
the1_d = sym.diff(the1,t)
the2_d = sym.diff(the2,t)
the1_dd = sym.diff(the1_d,t)
the2_dd = sym.diff(the2_d,t)

'''------------------------Definindo as variáveis-------------------------------'''
x1 = l1*sym.sin(the1)
y1 = -l1*sym.cos(the1)
x2 = l2*sym.sin(the2) + x1
y2 = -l2*sym.cos(the2) + y1

'''------------------------Definindo a Lagrangeana-------------------------------'''
T1 = 1/2*m1*(sym.diff(x1,t)**2 + sym.diff(y1,t)**2) #Energia cinética
U1 = y1*m1*g                                        #Energia potencial

T2 = 1/2*m2*(sym.diff(x2,t)**2 + sym.diff(y2,t)**2) #Energia cinética
U2 = y2*m2*g                                        #Energia potencial

T = T1 + T2
U = U1 + U2

L = T - U                                           #L é a Lagrangeana



EqL1 = (sym.diff(sym.diff(L,the1_d),t)-sym.diff(L,the1)).simplify() #equação theta 1
EqL2 = (sym.diff(sym.diff(L,the2_d),t)-sym.diff(L,the2)).simplify() #equação theta 2

sol = sym.solve([EqL1,EqL2],(the1_dd,the2_dd))                      #resolvendo para a aceleração angular

v1_f = sym.lambdify(the1_d,the1_d)
v2_f = sym.lambdify(the2_d,the2_d)

dv1_f = sym.lambdify((t,g,m1,m2,l1,l2,the1,the2,the1_d,the2_d),sol[the1_dd])
dv2_f = sym.lambdify((t,g,m1,m2,l1,l2,the1,the2,the1_d,the2_d),sol[the2_dd])

def dSdt(S,t,g,m1,m2,l1,l2):
    the1, the2, v1, v2 = S
    return[
        v1_f(v1),
        v2_f(v2),
        dv1_f(t,g,m1,m2,l1,l2,the1,the2,v1,v2),
        dv2_f(t,g,m1,m2,l1,l2,the1,the2,v1,v2),
    ]

t_f = 50#s
n_passo = 1000
t = np.linspace(0,t_f,n_passo)

'''------------------------Constantes-------------------------------'''
m1 = 1 # Kg
m2 = m1 # Kg
l1 = 1 # m
l2 = l1 # m
g = 9.81 # m/s2

'''------------------------Conds. iniciais-------------------------------'''
dthe1_10 = 0  # rad/s
dthe2_10 = 0  # rad/s


the1_10 = np.pi # rad
the2_10 = 5*np.pi/6 # rad

resposta1 = odeint(dSdt,y0=[the1_10,the2_10,dthe1_10,dthe2_10],t=t,args=(g,m1,m2,l1,l2))

the1_1t = resposta1.T[0]
the2_1t = resposta1.T[1]
dthe1_1t = resposta1.T[2]
dthe2_1t = resposta1.T[3]


T = (1/2) * (m1 + m2) * l1**2 * dthe1_1t**2 + (1/2) * m2 * l2**2 * dthe2_1t**2 + m2 * l1 * l2 * dthe1_1t * dthe2_1t * np.cos(the1_1t - the2_1t)
U = -(m1 + m2) * g * l1 * np.cos(the1_1t) - m2 * g * l2 * np.cos(the2_1t)
E = (T + U)

'''#trajetórias
plt.plot(t,the1_1t,label='Trajetória \u03B81')
plt.plot(t,the2_1t,label='Trajetória \u03B82')
plt.xlabel('Tempo (s)',fontsize = 20)
plt.ylabel('\u03B8 (rad)',fontsize = 20)
plt.legend()
plt.show()

#E TOTAL
plt.plot(t,E, label='Energia total')
plt.xlabel('Tempo (s)',fontsize = 20)
plt.ylabel('Energia Total (J)',fontsize = 20)
plt.ylim(-35,15)
plt.legend()
plt.show()

#energias
plt.plot(t,U, label='Energia Potencial')
plt.plot(t,T, label='Energia Cinética')
plt.xlabel('Tempo (s)',fontsize = 20)
plt.ylabel('Energia (J)',fontsize = 20)
plt.legend()
plt.show()

#espaços de fase
fig, axs = plt.subplots(2)
axs[0].plot(the1_1t, dthe1_1t,label='espaço de fase \u03B81')
axs[1].plot(the2_1t, dthe2_1t,label='espaço de fase \u03B82')
axs[0].set_xlabel('\u03B8 (rad)', fontsize = 18)
axs[0].set_ylabel('\u03C9(rad/s)', fontsize = 18)
axs[0].legend()
axs[1].set_xlabel('\u03B8 (rad)', fontsize = 18)
axs[1].set_ylabel('\u03C9(rad/s)', fontsize = 18)
axs[1].legend()
plt.show()
'''















def pos(t, the1, the2, l1, l2):
    x1 = -l1 * np.sin(the1)
    y1 = -l1 * np.cos(the1)
    x2 = -l2 * np.sin(the2) + x1
    y2 = -l2 * np.cos(the2) + y1
    return [
        x1, y1, x2, y2
    ]


x11, y11, x12, y12 = pos(t, the1_1t, the2_1t, l1, l2)


def animate(i):
    ln1.set_data([0, x11[i], x12[i]], [0, y11[i], y12[i]])

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.set_facecolor('k')
ax.get_xaxis().set_ticks([])  # Para tirar o eixo x
ax.get_yaxis().set_ticks([])

ln1, = plt.plot([], [], 'yo--', lw=2, markersize=8)

ax.set_ylim(-4, 4)
ax.set_xlim(-4, 4)
ani = animation.FuncAnimation(fig, animate, frames=n_passo, interval=5)
ani.save('pen.gif', writer='pillow', fps=len(t[t < 1]))  # FPS deve ser o número de intervalos em 1 segundo

plt.show()


