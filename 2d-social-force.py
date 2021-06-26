import sympy as smp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, odeint
from numba import njit

def flatten(vectors):
    return np.hstack((vectors[:, 0], vectors[:, 1]))

def twoDify(vectorx, vectory):
    return np.concatenate((vectorx.reshape(vectorx.shape[0], 1), vectory.reshape(vectory.shape[0], 1)), axis=1)

def V(r, v, vd, i, dt=0.1):
    # s = 0.5 # step length/width of pedestrian other pedestrian j
    ed = (vd - v) / np.linalg.norm((vd - v))

    s = np.linalg.norm(vd, axis=1) * dt
    b = 0.5*np.sqrt((np.linalg.norm(r, axis=1) + np.linalg.norm(r - (s * ed.T).T, axis=1))**2 - s**2) # eq (4) helbing
    b[i] = 0.0
    V0 = 2.1
    sigma = 0.3
    return V0 * np.exp(-b / sigma) # eq 13

def U(rB):
    U0 = 10.
    R = 0.2
    return U0 * np.exp(-np.linalg.norm(rB) / R)

def gradient(F, r, delta=1e-3, *args): # idea: use np.gradient instead
    dx = np.array([delta, 0.0])
    dy = np.array([0.0, delta])

    f = F(r, *args)
    dvdx = (F(r + dx, *args) - f) / delta
    dvdy = (F(r + dy, *args) - f) / delta

    return np.stack((dvdx, dvdy))

def field_of_view(f, r, c):
    twophi = 135.0
    return ((-f.T*r).sum(1) >= np.linalg.norm(-f) * np.cos(twophi / 2)) * (1 - c) + c

ts = [0.0]
def social(z, v, vd, tr, m, N, width, dt): # assuming F function is equal
    global ts

    c = 0.5
    forces = np.zeros((N, D))

    for i in range(N):

        motive_F = m[i] * (vd[i] - v[i]) / tr[i] # motive_F is a 2d vector

        ped_f = -gradient(V, z[i] - z, 1e-4, v, vd, i, dt)

        ped_F = (field_of_view(ped_f, vd[i] / np.linalg.norm(vd[i]), c) * ped_f).T

        left_wall = np.array([z[i][0], width/2])
        right_wall = np.array([z[i][0], -width/2])

        left_wall_F = -gradient(U, left_wall - z[i], 1e-4)
        right_wall_F = -gradient(U, right_wall - z[i], 1e-4)

        # forces[i] = tr[i]*(v[i] - pvmax[i]) - v[i]*social_force #returns a 2d vector

        F = ped_F.sum(0) + left_wall_F + right_wall_F + motive_F
        forces[i] = F

    return forces.flatten('F')

def ode(t, S, vd, tr, m, N, width):
    global ts
    ts.append(t)
    zxs = S[: N]
    zys = S[N: 2*N]
    vxs = S[2*N: 3*N]
    vys = S[3*N: 4*N]

    zs = twoDify(zxs, zys)
    vs = twoDify(vxs, vys)

    dt = ts[-1] - ts[-2]

    return np.hstack((vxs, vys, social(zs, vs, vd, tr, m, N, width, dt)))
# %%
np.random.seed(123)
N = 25
D = 2

width = 25.

z = np.stack((np.random.uniform(0, 0.5, N), np.random.uniform(low=-5, high=5, size=(N))), axis=1)
v = np.random.normal(loc=1.34, scale=0.26, size=(N, D))
m = np.ones(N) # np.random.normal(loc=76.9, scale=10., size=N)
tr = 0.5*np.ones(N) # np.random.normal(loc=0.5, scale=0.1, size=N)
vd = np.stack((np.random.uniform(0, 0.75)*np.ones(N, dtype=np.float64), np.zeros(N, dtype=np.float64)), axis=1)

initial = np.hstack((flatten(z), flatten(v)))

#%% `odeint` usage
time = np.linspace(0, 10, 101)
sol = odeint(ode, y0=initial, t=time, args=(vd, tr, m, N, width), tfirst=True)
sol = sol.T

# %%
sol = solve_ivp(ode, y0=initial, t_span=(1e-2, 50), args=(vd, tr, m, N, width), rtol=1e-2, atol=1e-4)
time = sol.t
sol = sol.y
# %%
zxs = sol[:N]
zys = sol[N: 2 * N]
vxs = sol[2 * N: 3 * N]
vys = sol[3 * N: 4 * N]

# %%
for i in range(len(time)):
    plt.scatter(zxs[:, i], zys[:, i])

# %%
import matplotlib.animation as animation

velocity_color = plt.cm.get_cmap('cool')
pedestrian_color = plt.cm.get_cmap('hsv', N)

fig, axs = plt.subplots(1, figsize=(12, 2), dpi=160)

def animate(frame):
    axs.cla()
    axs.set_title(f'time: {round(time[frame], 2)}')
    axs.set_xlim(zxs[:, frame].min() - 3, zxs[:, frame].max() + 3)
    # axs[0].set_xlim(0, loop)

    axs.set_ylim(zys.min() - 0.1, zys.max() + 0.1)

    vs = np.stack((vxs[:, frame], vys[:, frame]), axis=1)
    max_speed = np.linalg.norm(vs, axis=1).max()

    for i in range(N):
        axs.scatter(zxs[i, frame], zys[i, frame], color=pedestrian_color(i), s=100)

        axs.arrow(zxs[i, frame], zys[i, frame], vxs[i, frame], vys[i, frame], color=velocity_color(np.linalg.norm(vs[i]) / max_speed))

    # axs[1].cla()
    # axs[1].set_title('bridge velocity')
    # axs[1].plot(time[:frame], dbridge[:frame])

anim = animation.FuncAnimation(fig, animate, frames=len(time))

anim.save('figs/2d-social.mp4')
# %% interpolation
import scipy.interpolate as interpolate

zxs.size
interpolation = interpolate.UnivariateSpline(zxs, zys)

interpolate.BivariateSpline
