import sympy as smp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, odeint
import numba as nb
from numba import njit, jit, vectorize
import tqdm
from func_timeout import func_set_timeout

@njit
def H3(y, dy, o, a, l, p):
    return -(o)**2 * (y - p*(1 - 2 * (y < 0))) + l*(dy**2 + (o)**2 * (a**2 - (y - p*(1 - 2 * (y < 0)))**2))*dy

@njit
def H1(y, p, o):
    return o**2 * (p - y)

    # self.Hi = lambda y, dy, fv: (fv*o)**2 * (y - p*self.vsgn(y)) - l*(dy**2 - (fv*o)**2 * (y - p*self.vsgn(y))**2 + (fv*o)**2 * a**2)*dy
bridge_hz = 1.03
O = (bridge_hz*2*np.pi)

h = O * 0.04


@njit
def ddx(x, dx, H, r, h, O):
    return 1 / (1 - r.sum()) * ((H*r).sum() - h*dx - (O*O)*x)


@njit
def carroll(x):
    return 0.35*x*x*x - 1.59*x*x + 2.93*x

@njit
def f_max(v, vmax):
    return carroll(v) / carroll(vmax)
        # return v / vmax
        # return np.ones(len(v))

@njit
def social(z, v, vd, tr, N): # assuming F function is equal
    forces = np.zeros(N)
    for i in range(N):
        social_force = (2.0*((z - z[i]) > 0)*np.exp(-5.0*(z - z[i]))).sum() # SOCIAL FORCE FUNCTION self.F(z[i], pz).sum()
        forces[i] = tr[i]*(vd[i] - v[i]) - v[i]*social_force

    return forces

@njit
def F(zi, zj):
    return 2.0*((zj - zi) > 0)*np.exp(-5.0*(zj - zi))

@njit
def dean(v, height):
    return 1.3502 * np.sqrt(v) / height

'''
integrates wrt the whole system
incorporates bridge dynamics

Mx'' + 2hx' + Omega^2 x = -sum_i=1^N m_i y_i''
y_i'' + H(y_i, y_i', t) = -x''

where H comes from model 3
'''
@njit
def ode_bridge_full_optimized(t, S, loop, control):

    x = S[0, None]
    dx = S[1, None]
    peds = S[2:][:N] # lateral pos
    dpeds = S[2:][N:2 * N] # lateral vel

    zs = S[2:][2 * N:3 * N] # forward pos
    vs = S[2:][3 * N:4 * N] # forward vel

    if loop is not None:
        for i in range(N):
            if zs[i] > loop:
                zs[i] = 0.00
                vs[i] = 0.01 # need to change their entering velocity
    if control:
        for i in range(N):
            if peds[i] > p:
                dpeds[i] -= 0.5
                # print(f'controlled: {i}')
            elif peds[i] < -p:
                dpeds[i] += 0.5
                # print(f'controlled: {i}')

    Ls = 2*vs / carroll(vs)

    o = np.sqrt(9.81/Ls)  # self.v_to_o(vs, self.get('L')) # np.sqrt(9.81/self.get('L')) / 2 # 1.04 #  #

    # self.x = x
    # self.dx = dx
    # for i in range(self.N): # update person position and velocity
    #     self.people[i].z = zs[i]
    #     self.people[i].v = vs[i]
    #     self.people[i].y = peds[i]
    #     self.people[i].dy = dpeds[i]

    M = 113e3

    # try: # proportionally scale
    #     dpeds = dpeds * self.f_max(np.abs(vs), np.abs(self.history[-2][2:][3 * self.N:4 * self.N]))
    # except:
    #     pass

    r = m / (M + m)

    scale = (f_max(vs, vd)) # np.ones_like(o)#
    # scale = np.sqrt(vs / self.get('vmax'))

    H = H3(peds, dpeds, o*scale, a, l, p)

    # H = self.H1(peds, o*self.f_max(vs, self.get('vmax')), self.p)

    ddx = ddx(x, dx, H, r, h, O)

    return np.hstack(
        [dx, ddx,
        dpeds,
        -ddx - H,
        # np.ones_like(vs), np.ones_like(vs)])
        vs, social(zs, vs, vd, tr, N)])

@njit
def model_func(t, S, p, o, r, N, h, O, vmaxs, alphas):

    bridge = np.array([S[0]])
    dbridge = np.array([S[1]])
    y = S[2:][:N] # lateral pos
    dy = S[2:][N:2 * N] # lateral vel

    z = S[2:][2 * N:3 * N] # forward pos
    v = S[2:][3 * N:4 * N] # forward vel

    # scale = np.ones_like(o) # self.f_max(vs, self.get('vmax')) # np.ones_like(o) # ()

    H = H1(y, p, o)

    ddbrige = ddx(bridge, dbridge, H, r, h, O)

    return np.hstack((
        dbridge, ddbrige,
        dy, -ddbrige - H,
        # np.zeros_like(v), np.zeros_like(v)))
        v, social(z, v, vmaxs, alphas, N) # social force
     ))

@func_set_timeout(480)
def simulate_model_1_2(x, dx, people, params, N, t_f, model, ratio=None, sigma=1.):

    m = params[:, 0]
    L = params[:, 1]
    vd = params[:, 2]
    tr = params[:, 3]
    r = params[:, 4]

    o = np.sqrt(9.81 / L)

    M = 113e3
    r = m / (M + m)

    # forward_speed = forward_speed * np.ones(self.N)
    regular_leg = L / 1.34
    height = (regular_leg + 0.3091) / 0.7028

    v = people[:, 3].flatten()

    # fp = 1.3502 * (v ** 0.5) / height# 0.9*np.ones(N)# carroll(vd) / 2
    # fp = fp*np.ones(self.N)
    # fp = np.cbrt(v / (Ls * (1.352 / 1.34))**2)
    fp = carroll(v) #, height)

    bridge_hz = 1.03
    O = (bridge_hz*2*np.pi)
    if ratio:
        O /= ratio*O / (2*fp*np.pi)
        O = O.mean()
    h = O * 0.04


    t_0 = 0.
    t_s_next = np.random.uniform(0, (0.5/fp).max(), N) # 0.5/fp is step period

    bmin = np.random.randn(N)*0.002+0.0157 # bmin initialization
    bmin *= 1.0-2.0*(np.random.rand(N)>0.5) # choose foot

    p = bmin*(1.0-np.tanh(0.25*np.sqrt(9.8/L) / fp))

    y = np.zeros(N)
    dy = p * o * np.tanh(o * 0.5/fp)

    people[:, 0] = y.reshape(-1, 1)
    people[:, 1] = dy.reshape(-1, 1)

    state = np.hstack((x, dx, people.ravel('F')))

    times = []
    sols = []

    while t_0 < t_f:
        # Ls = np.sqrt(v / (fp**3) / (1.35/1.34)**2)
        # o = np.sqrt(9.81 / L)
        sol = solve_ivp(model_func, y0=state, t_span=(0, t_s_next.min() - t_0), args=(p, o, r, N, h, O, vd, tr), method='RK45')

        times.append(sol.t + t_0)
        sols.append(sol.y)

        state = sol.y[:, -1]
        x, dx, y, dy, z, v = parse(state, N)

        t_0 = t_s_next.min()
        idx_foot_down = np.where(t_s_next <= t_0 + 1e-10)[0]
        # fp = self.carroll(v[idx_foot_down]) / 2

        # plt.scatter(z, y)
        # plt.show()

        fp[idx_foot_down] = carroll(v[idx_foot_down]) # , height[idx_foot_down])# 1.3502 * np.sqrt(v[idx_foot_down]) / height[idx_foot_down] # carroll(v) / 2 #
        # print(v)

        o[idx_foot_down] = 1.5 * fp[idx_foot_down] # crude but necessary implemntation
        # fp = np.cbrt(v / (Ls * (1.352 / 1.34))**2)
        # fp = self.carroll(np.sqrt(v**2 + fp**3*Ls**2*(1.352/1.34))) / 2
        # plt.plot(sol.t, sol.y[2:][self.N:2*self.N].T);

        p[idx_foot_down] = y[idx_foot_down] + dy[idx_foot_down] / o[idx_foot_down] + bmin[idx_foot_down]
        bmin[idx_foot_down] *= -1

        if model == 1:
            t_s_next[idx_foot_down] += 0.5 / fp[idx_foot_down] # adding by the period (model 1)

        if model == 2:

            step_width = bmin[idx_foot_down]/(1 - np.tanh(o[idx_foot_down] / 4.0 / fp[idx_foot_down]))

            # step_width = 2.01 * y + 0.444 * dy
            # step_length = -0.52*np.sign(bmin) * y - 0.34*np.sign(bmin) *dy + 0.23*v
            step_length = v[idx_foot_down] / fp[idx_foot_down] #carroll(v[idx_foot_down])
            # step_width = 0.046 # unperturbed
            # step_length = 0.36 # can be updated based on social forward motion

            t_s_next[idx_foot_down] += 0.5 / fp[idx_foot_down]
            t_s_next[idx_foot_down] += (0.5 / fp[idx_foot_down]) * (step_width**2 - (y[idx_foot_down] - p[idx_foot_down])**2) / (4*step_length**2) # adding by the period (model 2)

    return [times, sols]


'''
IMPLEMENTATION NOT DONE
'''
def simulate_pedestrians(Ns, ts, loop, control):
    # Ns[i] = number of pedestrians to start walking at time ts[i]

    N = Ns.sum()
    people, params = generate_pedestrians(N, 1)


     #state = people.ravel('F')[:Ns[0]]

    times = []
    sols = []
    for i in range(1, len(ts)):
        # print(self.ode_bridge_full_optimized(0, state, loop, control).shape)
        sol = solve_ivp(ode_bridge_full_optimized, t_span=(ts[i-1], ts[i]), y0=state, args=(loop, control), method='RK23')
        if not sol.success:
            sol = solve_ivp(ode_bridge_full_optimized, t_span=(ts[i-1], ts[i]), y0=state, args=(loop, control), method='LSODA')

        times.append(sol.t)
        sols.append(sol.y)

        new_pedestrians = generate_pedestrians(Ns[i], 1)

        new_y = people[:, 0].ravel('F')
        new_dy = people[:, 1].ravel('F')
        new_z = people[:, 2].ravel('F')
        new_v = people[:, 3].ravel('F')

        bridge, dbridge, peds, dpeds, zs, vs = parse(sol.y[:, -1], N)

        state = np.hstack([bridge, dbridge, peds, new_y, dpeds, new_dy, zs, new_z, vs, new_v])
        N += Ns[i]
        if new_pedestrians != [[]] and new_pedestrians:
            people += new_pedestrians

    return [times, sols]


def plot_parameters(params):
    param_names = ['m', 'L', 'vd', 'tr', 'r']
    fig, ax = plt.subplots(1, len(params), figsize=(16, 4))
    for i in range(len(params)):
        ax[i].hist(params)
        ax[i].set_title(param_names[i])

    plt.tight_layout()

def plot_freqs(self):
    o = np.sqrt(9.81 / self.get('L'))

    plt.hist((self.O) / (0.25*o/np.log(self.p / self.a + np.sqrt((self.p / self.a)**2 - 1))))
    plt.xlabel('$\Omega / \omega$')
    plt.title('ratio of bridge to pedestrian frequency')

def plot_time_data(time, data):
    x, dx, y, dy, z, v = parse(data, (data.shape[0] - 2) // 4)
    fig, axs = plt.subplots(3, 2, figsize=(8, 10), sharex=True)

    axs[0][0].plot(time, x.T)
    axs[0][0].set_title('bridge position')
    axs[0][1].plot(time, dx.T)
    axs[0][1].set_title('bridge velocity')

    axs[1][0].plot(time, y.T)
    axs[1][0].set_title('lateral position')
    axs[1][1].plot(time, dy.T)
    axs[1][1].set_title('lateral velocity')

    axs[2][0].plot(time, z.T)
    axs[2][0].set_title('forward position')
    axs[2][1].plot(time, v.T)
    axs[2][1].set_title('forward velocity')

    plt.tight_layout()

def plot_phase(data):
    x, dx, y, dy, z, v = parse(data, (data.shape[0] - 2) // 4)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].plot(x, dx)
    axs[0].set_title('bridge position vs velocity')

    axs[1].plot(y, dy)
    axs[1].set_title('lateral position vs velocity')


@njit
def parse(S, N):
    bridge, dbridge = S[0:2]
    peds = S[2:][:N] # lateral pos
    dpeds = S[2:][N:2 * N] # lateral vel

    zs = S[2:][2 * N:3 * N] # forward pos
    vs = S[2:][3 * N:4 * N] # forward vel

    return bridge, dbridge, peds, dpeds, zs, vs

def plot_state(S, N):
    bridge, dbridge, peds, dpeds, zs, vs = parse(S, N)

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))

    ax[0].hist(peds)
    ax[0].set_title('lateral positions')
    ax[1].hist(dpeds)
    ax[1].set_title('lateral velocities')
    ax[2].hist(zs)
    ax[2].set_title('forward positions')
    ax[3].hist(vs)
    ax[3].set_title('forward velocities')

    plt.tight_layout()

def get_last_peak(time, bridge):
    return np.abs(bridge[-len(time) // 50:]).max()

def rk4(f, t, x, h):

    k1 = h * (f(t, x))
    k2 = h * (f((t+h/2), (x+k1/2)))
    k3 = h * (f((t+h/2), (x+k2/2)))
    k4 = h * (f((t+h), (x+k3)))
    k = (k1+2*k2+2*k3+k4)/6
    x = x + k

    return x


# %%
def generate_pedestrians(N, D, sigma=1.):
    y0 = np.random.uniform(-0.2, 0.2, size=(N, D))
    dy0 = np.random.uniform(-0.2, 0.2, size=(N, D))
    z0 = np.random.uniform(low=0, high=10, size=(N, D))
    v0 = np.random.normal(loc=1.4, scale=0.2, size=(N, D))
    vd = 1.3 * v0
    m = np.random.normal(loc=76.9, scale=10., size=N)
    tr = np.random.normal(loc=0.5, scale=0.01, size=N)
    r = np.random.normal(loc=0.35, scale=0.01, size=N)
    L = np.random.normal(1.17, 0.092*sigma, size=N)
    people = np.stack((y0, dy0, z0, v0), axis=1)
    params = np.stack((m, L, vd.flatten(), tr, r), axis=1)

    return people, params
# %%

%matplotlib inline
np.random.seed(123)

D = 1
N = 200
sigma = 1

people, params = generate_pedestrians(N, 1, sigma)

model2 = simulate_model_1_2(x=0.0, dx=0.01, people=people, params=params, N=N, t_f=50, model=2, ratio=3., sigma=1.0)

time = np.hstack(model2[0])
data = np.hstack(model2[1])
plot_time_data(time, data)
plt.savefig('figs/coupling/dean-formula.pdf')

# plot_freqs()
# %%
x, dx, y, dy, z, v = parse(data, 1)
plt.plot(time, y.T)
(0.9*1.17/1.34/1.352)**2*0.9
# %%
vs = np.linspace(0, 3., 1001)
L = 1.17

regular_leg = L / 1.34
height = (regular_leg + 0.3091) / 0.7028

d = 1.3502 * np.sqrt(vs) / height
c = carroll(vs)

o = np.sqrt(9.81 / L)

o / carroll(1.5)

o
dean(1.5, height) * 3

plt.axhline(o)
plt.plot(vs, 3*d, label='dean')
plt.plot(vs, 1.5*c, label='carroll')
plt.legend()
plt.ylabel('lateral frequency (1/s)')
plt.xlabel('velocity (m/s)')
plt.savefig('coupling-formulas.pdf')
# %%
sigmas = np.array([0.05, 0.1, 0.25, 1.])
dxs = np.array([0.0, 0.04, 0.08])
ratios = np.linspace(0.2, 3.2, 16)
Ns = np.arange(150, 235, 5)
# Ns = np.arange(1, 15, 5)

# %%

maxs = np.zeros((len(dxs), len(sigmas), len(ratios), len(Ns)))

for i, dx in enumerate(dxs):
    for j, sigma in enumerate(sigmas):
        for k, ratio in tqdm.tqdm(enumerate(ratios)):
            t = 60
            for l, N in enumerate(Ns):
                np.random.seed(1234)
                people, params = generate_pedestrians(N, 1, sigma)
                try:
                    model2 = simulate_model_1_2(x=0.0, dx=dx, people=people, params=params, N=N, t_f=t, model=2, ratio=ratio, sigma=sigma)
                    time = np.hstack(model2[0])
                    data = np.hstack(model2[1])
                    # np.save(f'data/violinplot/data-ratio={round(ratio, 2)}-N={N}.npy', np.vstack([time, data]))

                except:
                    try:
                        model2 = simulate_model_1_2(x=0.0, dx=dx, people=people, params=params, N=N, t_f=t-25, model=2, ratio=ratio, sigma=sigma)
                        time = np.hstack(model2[0])
                        data = np.hstack(model2[1])

                        # np.save(f'data/violinplot/data-ratio={round(ratio, 2)}-N={N}.npy', np.vstack([time, data]))
                    except:
                        continue

                maxs[i][j][k][l] = np.abs(data[0][-len(time) // 50:]).max()
    np.save('data/violinplot2/peaks.npy', maxs)

 # %%
# ((fp)*get('L')/1.34/1.352)**2*(fp)
Ns = np.arange(1, 200, 5)
# datas = []
# times = []
for N in tqdm.tqdm(Ns):
    np.random.seed(123)
    sim = Simulation(N, D, z0, v0, y, dy, m0, L0, vmax, alpha, F, x, dx, ratio=None)

    test = simulate_model_1_2(t_f=80, fp=0.9, step_width=step_width, forward_speed=forward_speed, model=2)
    data = np.hstack(test[1])
    time = np.hstack([np.hstack(test[0])])
    datas.append(data)
    times.append(time)

# %%
maxs = []

for ratio in ratios:
    max_peaks = []
    for N in Ns:
        data = np.load(f'data/violinplot/data-ratio={round(ratio, 2)}-N={N}.npy')
        time = data[0]
        bridge, dbridge, peds, dpeds, zs, vs = parse(data[1:], N)
        max_peak = np.abs(bridge[-len(time) // 50:]).max()
        max_peaks.append(max_peak)
    maxs.append(max_peaks)



    # plt.plot(Ns, maxs)
max_data = np.load('data/violinplot/final.npy')
max_data_2 = np.array(maxs)
np.save('data/violinplot/final2.npy', max_data_2)
# np.save(f'data/violinplot/final.npy', max_data)
# %%
from sortedcontainers import SortedDict

plt.rcParams["figure.figsize"] = (8, 6)

ratios_final = np.sort(np.concatenate([ratios, ratios2]))
d = dict(zip(np.around(ratios, 2), max_data)) | dict(zip(np.around(ratios2, 2), max_data_2))

s = SortedDict(d)

data = np.array(s.values())

for i in range(len(Ns)):
    plt.plot(ratios_final, data[:, i], label=Ns[i])

plt.xlabel('$\Omega / \omega$')
plt.ylabel('$A_x$')
plt.title('bridge amplitude and ratio')
plt.xlim(0.25, 2)
plt.legend()
plt.savefig('figs/brige-ratio-zoomed.pdf')
# %%
data = np.load(f'data/violinplot/data-ratio={1.}-N={230}.npy')

data[1:][1][0]

plot_time_data(data[0], data[1:])

# %%
for i in range(len(test[0])):
    bridge, dbridge, peds, dpeds, zs, vs = parse(test[1][i], N)
    plt.plot(test[0][i], bridge.T)
    # plt.xlim([0, 1])
    # plt.ylim([-.02, .02])
# %%
#y0 = lambda: np.random.uniform(-0.1, 0.1)
#dy0 = lambda: np.random.uniform(-0.1, 0.1)

# o = 0.73
o = np.sqrt(9.81 / get('L'))
bounds = (0.5/(o))*np.log(p / a + np.sqrt((p / a)**2 - 1)) / 2*np.pi
bound = bounds.max()
t = np.random.uniform(-bound, bound, N)
signs = np.random.choice([-1, 1], N)



o = np.sqrt(9.81 / get('L'))

y0 = (p - a*np.cosh(o * t)) * signs
dy0 = -a * np.sinh(o * t) * signs

plt.scatter(y0, dy0)
# %%

plot_freqs()
# %%
# initial = np.hstack([[x, dx], [y0() for i in range(N)], [dy0() for i in range(N)], [z0() for i in range(N)], [v0() for i in range(N)]])
initial = np.hstack([[x, dx], y0, dy0, [z0() for i in range(N)], [v0() for i in range(N)]])
loop = None # 100
control = True
# plot_state(initial, N)

sol = solve_ivp(ode_bridge_full_optimized, t_span=(0, 50), y0=initial, args=(loop, control), method='RK45')
# %%
# sol, d = odeint(ode_bridge_full, y0=initial, t=time, full_output = 1, tfirst=True, args=(loop, ))
num_of_runs = 30
times = []
sols = []
t0, tf = 0, 15
step = 30
for i in tqdm.tqdm(range(num_of_runs)):
    sol = solve_ivp(ode_bridge_full_optimized, t_span=(t0, tf), y0=initial, args=(loop, control), method='RK45')
    times.append(sol.t + step*(i + 1))
    sols.append(sol.y)
    t0 = sol.t.min()
    tf = t0 + step
    initial = sol.y[:, -1]

    # print(initial[0], initial[1])
    plt.plot(np.hstack(times), np.hstack([data[0] for data in sols])); # bridge position
    plt.title('bridge position')
    plt.show()

 # %%
for i in range(1, num_of_runs):
    times[i] += i*step

time = np.hstack(times)
S = np.hstack(sols)

# %%
# sol = odeint(ode_bridge_full_optimized, y0=initial, t=np.linspace(0, 100, 10001), args=(loop, control), tfirst=True)
 # %%
time = sol.t
S = sol.y

bridge, dbridge = S[0:2]
peds = S[2:][:N] # lateral pos
dpeds = S[2:][N:2 * N] # lateral vel

zs = S[2:][2 * N:3 * N] # forward pos
vs = S[2:][3 * N:4 * N] # forward vel

# # %%
# %matplotlib qt
#
# fig = plt.figure(figsize=(12, 6))
# ax = plt.axes(projection='3d')
# # plt.plot(time, p - peds.transpose()*sgn(peds.transpose()));
# ax.set_xlabel('time')
# ax.set_ylabel('y')
# ax.set_zlabel('v')
#
# for i in range(N):
#     ax.plot3D(time, peds[i].T, vs[i].T)

# %%
import matplotlib.colors as colors

cmap = plt.cm.get_cmap('Reds')
norm = colors.Normalize(0, max([vmax() for _ in range(100)]))


for i in range(len(peds)):
    plt.plot(time, peds[i], color=cmap(norm(vs[i].mean())))

# plt.ylim(-2, 2)
plt.xlabel('time')
plt.ylabel('y')
plt.title('lateral position');

# %% lateral positions
plt.plot(time, peds.T);
# %% lateral velocities
# plt.plot(np.cbrt(vs.transpose() / get('L')**2)*2);

plt.plot(time, dpeds.T);
plt.title('lateral velocity')

# plt.savefig('interesting synchrony.pdf')

# %% phase diagram of lateral direction

plt.plot(peds.T, dpeds.T);
plt.title('lateral position vs velocity')

# %%
plt.plot(time[:], zs.T[:]); # forward position
plt.title('forward position')
# %%
plt.plot(time, vs.T); # forward position
plt.title('forward velocity')
# %% frequencies
# (np.sqrt(9.81/ get('L')) / vs.T**0.58).std()
#
#
# carroll(vs[:, 3]) / carroll(get('vmax'))
#
# np.sqrt(9.81/ get('L')) * vs.T / get('vmax')
#
# plt.plot(time, v_to_o(vs.T, get('L')));

# %%
plt.plot(time, (bridge.T)); # bridge position
plt.title('bridge position')
# plt.savefig('figs/bridge ratio=0.25.pdf')
# %%
plt.plot(time, dbridge.T); # bridge position
plt.title('bridge velocity')
 # %%
plt.plot(time, np.gradient(dbridge, time)); # bridge position
plt.title('bridge acceleration')
# %%
plt.plot(bridge.T, dbridge.T); # bridge position
plt.title('bridge position vs velocity')

# %%
%matplotlib qt

fig = plt.figure(figsize=(12, 6))
ax = plt.axes(projection='3d')
# plt.plot(time, p - peds.transpos e()*sgn(peds.transpose()));
ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
ax.set_zlabel('$\dot{x}$')

ax.plot3D(time, bridge, dbridge)
# %%
%matplotlib qt

fig = plt.figure(figsize=(12, 6))
ax = plt.axes(projection='3d')
# plt.plot(time, p - peds.transpose()*sgn(peds.transpose()));
ax.set_xlabel('$t$')
ax.set_ylabel('$y$')
ax.set_zlabel('$v$')

for i in range(N):
    ax.plot3D(time, peds[i], vs[i])
# %%
%matplotlib qt

fig = plt.figure(figsize=(12, 6))
ax = plt.axes(projection='3d')
# plt.plot(time, p - peds.transpose()*sgn(peds.transpose()));
ax.set_xlabel('$x$')
ax.set_ylabel('$\dot{x}$')
ax.set_zlabel('$\ddot{x}$')

ax.plot3D(bridge, dbridge, np.gradient(dbridge, time))
# %%
np.random.seed(123)
N = 1
D = 1

def F(zi, zj):
    kappa = 2.0
    c = 5.0
    heaviside = lambda x: 1 if x > 0 else 0
    return kappa*heaviside(zj - zi)*np.exp(-c*(zj - zi))

z0 = lambda: np.random.uniform(low=0, high=1)
v0 = lambda: np.random.normal(loc=0.8, scale=0.1)

y = lambda: np.random.uniform(-0.1, 0.1)
dy = lambda: np.random.uniform(-0.1, 0.1)
x = 0.000
dx = 0.001 # 5

sigma = 1
m0 = lambda: np.random.normal(76.9, 10*sigma) # 70
L0 = lambda: 1.17 # lambda: np.random.normal(1.17, 0.092*sigma)

vmax = lambda: np.random.normal(1.5, 0.125) # 1.5 # np.random.uniform(1.3, 1.8) # 1.5
alpha = lambda: -1*(0.1*np.random.uniform(0, 1) + 0.025*np.random.uniform(0, 5)) # -0.1

sim2 = Simulation(N, D, z0, v0, y, dy, m0, L0, vmax, alpha, F, x, dx, ratio=1.)
sim2.plot_parameters()

# %%
Ns = np.hstack([np.random.randint(0, 3, 50)*np.ones(50, dtype=np.int64), np.zeros(50, dtype=np.int64)]) # np.ones(200, dtype=np.int64) #
total = np.cumsum(Ns)[-1]
ts = np.arange(100, dtype=np.float64)
loop = None
control = None

test = sim2.simulate_pedestrians(Ns, ts, loop, control)

# %%
for i in range(len(test[0])):
    bridge, dbridge, peds, dpeds, zs, vs = parse(test[1][i], Ns[i])
    plt.plot(test[0][i], bridge.T)



# %%
# f = lambda v, vmax: (0.35*v*v*v - 1.59*v*v + 2.93*v) / (0.35*vmax*vmax*vmax - 1.59*vmax*vmax + 2.93*vmax)
#
# test = np.array([f_max(vs[:, i], get('vmax')) for i in range(len(time))])
#
# for i in range(len(time)):
#     plt.plot(vs[:, i], f_max(vs[:, i], get('vmax')))


 # %%
import matplotlib.animation as animation

bridge, dbridge, peds, dpeds, zs, vs = parse(data, N)

fig, axs = plt.subplots(1, 2, figsize=(12, 2), dpi=160, gridspec_kw={'width_ratios': [3, 1]})

vector_color = plt.cm.get_cmap('cool')
def animate(frame):
    frame = frame * 4
    axs[0].cla()
    axs[0].set_title(f'time: {round(time[frame], 2)}')
    # axs[0].set_xlim(zs[:, frame].min() - 3, zs[:, frame].max() + 3)

    axs[0].set_xlim(0, loop)
    axs[0].set_ylim(peds.min() - 0.1, peds.max() + 0.1)
    plot_dots_from_data(frame, zs, vs, peds, dpeds, axs[0])

    axs[1].cla()
    axs[1].set_title('bridge velocity')
    axs[1].plot(time[:frame], dbridge[:frame])

anim = animation.FuncAnimation(fig, animate, frames=len(time) - 1600)

anim.save('figs/model2.mp4')
# %% WARNING: took over 6 gb of harddrive space to save all data
D = 1

def H(y, dy, t=None):
    if y >= 0:
        return o**2 * (y - p) - l*(dy**2 - v**2 * (y - p)**2 + v**2 * a**2)*dy
    else:
        return o**2 * (y + p) - l*(dy**2 - v**2 * (y + p)**2 + v**2 * a**2)*dy

def F(zi, zj):
    kappa = 2.0
    c = 5.0

    heaviside = lambda x: 1 if x > 0 else 0
    return kappa*heaviside(zj - zi)*np.exp(-c*(zj - zi))

z0 = lambda: np.random.uniform(low=0, high=5)
v0 = lambda: np.random.normal(loc=0.8, scale=0.01)
y = 0.008
dy = 0.0001
x = 0.000
dx = 0.0005

# sigma = 0.05
# m0 = lambda: np.random.normal(76.9, 10*sigma) # 70
# L0 = lambda: np.random.normal(1.17, 0.092*sigma)
y0 = lambda: np.random.uniform(-0.1, 0.1)
dy0 = lambda: np.random.uniform(-0.1, 0.1)

vmax = 1.5 # lambda: np.random.uniform(1.3, 1.8) # 1.5
alpha = -0.1 #  lambda: -1*(0.1*np.random.uniform(0, 1) + 0.01*np.random.uniform(0, 5)) # -0.1

data = []

loop = None
# sigmas = [0.05, 0.175, 0.25, 0.5]# np.linspace(0, 1, 21)
sigmas = [0.125]# np.linspace(0, 1, 21)
Ns = np.arange(0, 151, 5)
Ns
# %%
for sigma in tqdm.tqdm(sigmas):
    m0 = lambda: np.random.normal(76.9, 10*sigma) # 70
    L0 = lambda: np.random.normal(1.17, 0.092*sigma)
    for N in tqdm.tqdm(Ns):
        for _ in range(3):
            sim = Simulation(N, D, z0, v0, y, dy, m0, L0, vmax, alpha, F, x, dx, 'hsv', 'cool')

            initial = [x, dx] + [y0() for i in range(N)] + [dy0() for i in range(N)] + [z0() for i in range(N)] + [v0() for i in range(N)]
            # sol, d = odeint(ode_bridge_full, y0=initial, t=time, full_output = 1, tfirst=True, args=(loop, ))
            sol = solve_ivp(ode_bridge_full_optimized, t_span=(0, 100), y0=initial, args=(loop, ))
            data.append(sol)

# %%
step = len(Ns)*3

sigma_data = []

for i in range(len(sigmas)):
    sigma_data.append(data[i*step: (i+1)*step])

# %%

for i in range(len(sigmas)):
    for j in range(len(sigma_data[i])):
        sol = sigma_data[i][j]
        np.save(f'data/sigma={sigmas[i]}-N={j // 3}-idx={j}', np.vstack([sol.t, sol.y]))

# %%

maxs = np.zeros((len(Ns), len(sigmas), 3))

for k in range(len(Ns)):
    for j in range(len(sigmas)):
        for i in range(3):
            count = k
            test = np.load(f'data/sigma={sigmas[j]}-N={count}-idx={count*3 + i}.npy')
            t = test[0]
            bridge, dbridge, peds, dpeds, zs, vs = parse(test[1:], Ns[count])

            maxs[k][j][i] = np.abs(bridge[-len(t) // 50:]).max()


# %%
plt.plot(Ns, maxs.mean(1).mean(1))
plt.xlabel('number of pedestrians')
plt.ylabel('bridge amplitude')
plt.title('average bridge amplitude over all runs')
plt.savefig('figs/mean-all.pdf')

# %%
plt.plot(Ns, maxs.mean(1))
plt.xlabel('number of pedestrians')
plt.ylabel('bridge amplitude')
plt.legend([1, 2, 3])
plt.title('average bridge amplitude over all sigmas')
plt.savefig('figs/mean-runs.pdf')
# %%
plt.plot(Ns, maxs.mean(2))
plt.xlabel('number of pedestrians')
plt.ylabel('bridge amplitude')
plt.legend(sigmas)
plt.title('first run bridge amplitude over all runs')
plt.savefig('figs/mean-sigmas.pdf')
# %%
import matplotlib.colors as colors

initial = [0.0, 0.0] + [y0() for i in range(N)] + [dy0() for i in range(N)] + [z0() for i in range(N)] + [v0() for i in range(N)]
x = 0.0
dxs = np.linspace(0, 0.008, 11)

norm = colors.Normalize(0, 0.008)

for dx in dxs:
    initial[1] = dx
    sim = Simulation(N, D, z0, v0, y, dy, m0, L0, vmax, alpha, F, x, dx, 'hsv', 'cool')
    sol = solve_ivp(ode_bridge_full_optimized, t_span=(0, 50), y0=initial, args=(loop, ), method='RK23')
    bridge, dbridge, peds, dpeds, zs, vs = parse(sol.y, N)
    plt.plot(bridge[::-1], dbridge[::-1], label='$\dot{x}_0$: ' + '%.1E' % dx, color=plt.get_cmap('Greens')(norm(dx)))

plt.legend(bbox_to_anchor=(1.05, 1))
plt.savefig('figs/bridge-dbridge.pdf')
