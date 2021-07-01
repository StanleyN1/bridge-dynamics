import sympy as smp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, odeint
from numba import njit, jit

@njit
def flatten(vectors):
    return np.hstack((vectors[:, 0], vectors[:, 1]))

@njit
def twoDify(vectorx, vectory):
    return np.concatenate((vectorx.reshape(vectorx.shape[0], 1), vectory.reshape(vectory.shape[0], 1)), axis=1)

@njit
def norm_axis_1(vector):
    norm = np.zeros(vector.shape[0])
    for i in range(vector.shape[0]):
        norm[i] = np.linalg.norm(vector[i])
    return norm

@njit
def V(r, ed, v, i, dt=0.1):
    # s = 0.5 # step length/width of pedestrian other pedestrian j

    s = norm_axis_1(v) * dt
    b = 0.5*np.sqrt((norm_axis_1(r) + norm_axis_1(r - (s * ed.T).T))**2 - s**2) # eq (4) helbing
    b[i] = 0.0
    V0 = 2.1
    sigma = 0.3
    return V0 * np.exp(-b / sigma) # eq 13

@njit
def U(rB, ed, v, i, dt=0.1):
    U0 = 10
    R = 0.2
    return U0 * np.exp(-np.linalg.norm(rB) / R)
    # if np.linalg.norm(rB) <= 1.0:
    #     return U0 * np.exp(-2*np.linalg.norm(rB) / R)
    # else:
    #     return U0 * np.exp(np.linalg.norm(rB) / R)

def gradient(F, r, delta, ed, v, i, dt=0.1): # idea: use np.gradient instead
    dx = np.array([delta, 0.0])
    dy = np.array([0.0, delta])


    f = F(r, ed, v, i, dt)
    dvdx = (F(r + dx, ed, v, i, dt) - f) / delta
    dvdy = (F(r + dy, ed, v, i, dt) - f) / delta

    return np.vstack((dvdx, dvdx)) # np.vstack((dvdx, dvdy))

@njit
def field_of_view(f, r, c):
    twophi = 135.0
    return ((-f.T*r).sum(1) >= np.linalg.norm(-f) * np.cos(twophi / 2)) * (1 - c) + c

ts = [0.0]
def social(z, rd, v, vd, tr, m, N, width, dt): # assuming F function is equal
    c = 0.5
    forces = np.zeros((N, D))
    delta = 1e-2
    ed = (rd - z) / np.linalg.norm(rd - z)
    for i in range(N):
        motive_F = m[i] * (vd[i]*ed[i] - v[i]) / tr[i] # motive_F is a 2d vector

        ped_f = -gradient(V, z[i] - z, delta, ed, v, i, dt)

        ped_F = (field_of_view(ped_f, ed, c) * ped_f).T

        left_wall = np.array([z[i][0], width/2])
        right_wall = np.array([z[i][0], -width/2])

        left_wall_F = -gradient(U, left_wall - z[i], delta, ed, v, i, dt)
        right_wall_F = -gradient(U, right_wall - z[i], delta, ed, v, i, dt)

        # forces[i] = tr[i]*(v[i] - pvmax[i]) - v[i]*social_force #returns a 2d vector

        F = ped_F.sum(0) + left_wall_F.reshape(-1) + right_wall_F.reshape(-1) + motive_F

        forces[i] = F

    return forces.flatten('F')

def ode(t, S, rd, vd, tr, m, N, width):
    global ts
    ts.append(t)
    zxs = S[: N]
    zys = S[N: 2*N]
    vxs = S[2*N: 3*N]
    vys = S[3*N: 4*N]

    zs = twoDify(zxs, zys)
    vs = twoDify(vxs, vys)

    dt = ts[-1] - ts[-2]

    return np.hstack((vxs, vys, social(zs, rd, vs, vd, tr, m, N, width, dt)))
# %%
np.random.seed(123)
N_forward = 60
N_backward = 0
N = N_forward + N_backward
D = 2

end = 10.
width = 15.


pedestrian_y = width // 2 - 1
pedestrian_x = end - 1

z_forward = np.stack((np.random.uniform(0, pedestrian_x, N_forward), np.random.uniform(low=-pedestrian_y, high=pedestrian_y, size=N_forward)), axis=1)
z_backward = np.stack((np.random.uniform(end - pedestrian_x, end, N_backward), np.random.uniform(low=-pedestrian_y, high=pedestrian_y, size=N_backward)), axis=1)

z = np.vstack((z_forward, z_backward))

v_forward = np.stack((1.5*np.ones(N_forward), 0.0*np.ones(N_forward)), axis=1) # np.random.normal(loc=1.34, scale=0.26, size=(N_forward, D))
v_backward = np.stack((-1.5*np.ones(N_backward), 0.0*np.ones(N_backward)), axis=1) # np.random.normal(loc=-1.34, scale=0.26, size=(N_backward, D))

v = np.vstack((v_forward, v_backward))

# rd_forward = np.stack((20.*np.ones(N_forward), 0.0*np.ones(N_forward)), axis=1)
# rd_backward = np.stack((0.*np.ones(N_backward), 0.0*np.ones(N_backward)), axis=1)
rd_forward = np.stack((20.*np.ones(N_forward), z_forward[:, 1]), axis=1)
rd_backward = np.stack((0.*np.ones(N_backward), z_backward[:, 1]), axis=1)

rd = np.vstack((rd_forward, rd_backward))

vd = 3.*np.ones(N)
m = np.ones(N) # np.random.normal(loc=76.9, scale=10., size=N)
tr = 0.5*np.ones(N) # np.random.normal(loc=0.5, scale=0.1, size=N)


initial = np.hstack((flatten(z), flatten(v)))

#%% `odeint` usage
# time = np.linspace(0, 10, 101)
# sol = odeint(ode, y0=initial, t=time, args=(vd, tr, m, N, width), tfirst=True)
# sol = sol.T

# %%
tf = 50

sol = solve_ivp(ode, y0=initial, t_span=(1e-2, tf), args=(rd, vd, tr, m, N, width), rtol=1e-1, atol=1e-2, dense_output=True)
time = sol.t
S = sol.y
# %%
zxs = S[:N]
zys = S[N: 2 * N]
vxs = S[2 * N: 3 * N]
vys = S[3 * N: 4 * N]

# %%
from matplotlib.cm import get_cmap

color = get_cmap('hsv', N)

plt.axhline(width/2, color='black')
plt.axhline(-width/2, color='black')
for i in range(len(time[0:100])):
    for j in range(N):
        plt.scatter(zxs[j, -1], zys[j, -1], color=color(j))

# %%
import matplotlib.animation as animation
import matplotlib.colors as colors

velocity_color = plt.cm.get_cmap('cool')
pedestrian_color = plt.cm.get_cmap('seismic')

fig, axs = plt.subplots(1, figsize=(12, 2), dpi=160)


norm = colors.TwoSlopeNorm(vmin=-1.5, vcenter=0., vmax=1.5)

def animate(frame):
    axs.cla()
    axs.set_title(f'time: {round(time[frame], 2)}')
    axs.set_xlim(-1., 21.)
    # axs.set_xlim(zxs[:, frame].min() - 3, zxs[:, frame].max() + 3)
    # axs[0].set_xlim(0, loop)

    # axs.set_ylim(zys.min() - 0.1, zys.max() + 0.1)
    axs.set_ylim(-10, 10)
    plt.axhline(width/2, color='black')
    plt.axhline(-width/2, color='black')

    vs = np.stack((vxs[:, frame], vys[:, frame]), axis=1)
    max_speed = np.linalg.norm(vs, axis=1).max()

    for i in range(N):
        axs.scatter(zxs[i, frame], zys[i, frame], color=pedestrian_color(norm(vxs[i, frame])), s=100)

        # axs.arrow(zxs[i, frame], zys[i, frame], vxs[i, frame], vys[i, frame], color=velocity_color(np.linalg.norm(vs[i]) / max_speed))

    # axs[1].cla()
    # axs[1].set_title('bridge velocity')
    # axs[1].plot(time[:frame], dbridge[:frame])

anim = animation.FuncAnimation(fig, animate, frames=len(time))

anim.save('figs/social/2d-social-N=120-all-forward4.mp4')
