import sympy as smp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, odeint
import numba as nb
from numba import njit, jit, vectorize

class Simulation:
    def __init__(self, N, D, z0, v0, y0, dy0, m0, L0, vmax, alpha, F, x, dx, z_color=None, v_color=None):
        functionify = lambda x: x if callable(x) else (lambda: x)

        self.N = N
        self.D = D

        self.z0 = functionify(z0)
        self.v0 = functionify(z0)
        self.y0 = functionify(y0)
        self.dy0 = functionify(dy0)

        self.m0 = functionify(m0)
        self.L0 = functionify(L0)

        self.vsgn = np.vectorize(lambda y: 1 - 2 * (y < 0))

        self.v_to_o = lambda v, L: 2 * np.cbrt(v / (L**2 * (1.352/1.34)))

        self.l = 23.25 # limit_cycle_damping_parameter
        self.a = 0.47 # limit_cycle_amplitude_parameter
        self.p = 0.63

        @jit(nb.float64[:](nb.float64[:], nb.float64[:], nb.float64[:], nb.float64, nb.float64, nb.float64))
        def H(y, dy, o, a, l, p):
            return -(o)**2 * (y - p*(1 - 2 * (y < 0))) + l*(dy**2 + (o)**2 * (a**2 - (y - p*(1 - 2 * (y < 0)))**2))*dy

        @jit([nb.float64[:](nb.float64[:], nb.float64[:], nb.float64[:]), nb.float64[:](nb.float64[:], nb.float64[:], nb.float64)])
        def H1(y, o, p):
            return -(o)**2 * (p - y)

        self.H1 = H1
        self.H = H # np.vectorize(H)
        # self.Hi = lambda y, dy, fv: (fv*o)**2 * (y - p*self.vsgn(y)) - l*(dy**2 - (fv*o)**2 * (y - p*self.vsgn(y))**2 + (fv*o)**2 * a**2)*dy
        self.bridge_hz = 1.03
        self.O = (self.bridge_hz*2*np.pi)
        self.h = self.bridge_hz*2*np.pi * 0.04

        @jit(nb.float64[:](nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64, nb.float64))
        def ddx(x, dx, H, r, h, O):
            return 1 / (1 - r.sum()) * ((H*r).sum() - h*dx - (O*O)*x)

        self.ddx = ddx

        @jit([nb.float64[:](nb.float64[:]), nb.float64(nb.float64)])
        def f(x):
            return 0.35*x*x*x - 1.59*x*x + 2.93*x

        self.f = f

        @jit(nb.float64[:](nb.float64[:], nb.float64[:]))
        def f_max(v, vmax):
            # return (0.35*v*v*v - 1.59*v*v + 2.93*v) / (0.35*vmax*vmax*vmax - 1.59*vmax*vmax + 2.93*vmax)
            # return v / vmax
            return np.ones(len(v))

        self.f_max = f_max # np.vectorize(f_max)

        @jit(nb.float64[:](nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.int32))
        def social(z, v, pz, pvmax, palpha, N): # assuming F function is equal
            forces = np.zeros(N)
            for i in range(N):
                social_force = (2.0*((pz - z[i]) > 0)*np.exp(-5.0*(pz - z[i]))).sum() # SOCIAL FORCE FUNCTION self.F(z[i], pz).sum()
                forces[i] = palpha[i]*(v[i] - pvmax[i]) - v[i]*social_force

            return forces

        self.social = social # np.vectorize(social, signature='(n),(n)->(n)')

        @jit([nb.float64[:](nb.float64, nb.float64[:]), nb.float64(nb.float64, nb.float64)])
        def F(zi, zj):
            return 2.0*((zj - zi) > 0)*np.exp(-5.0*(zj - zi))

        self.F0 = F # non vectorized
        self.F = F # vectorize(F) # np.vectorize(F)

        self.x = x
        self.dx = dx

        # currently constants
        self.vmax = functionify(vmax)
        self.alpha = functionify(alpha)

        # list of pedestrians
        self.initialize_pedestrians()

        self.z_color = plt.cm.get_cmap(z_color, N)
        self.v_color = plt.cm.get_cmap(v_color)

        self.history = []

    def initialize_pedestrians(self):
        self.people = [Person(self.z0(), self.v0(), self.y0(), self.dy0(), self.vmax(), self.alpha(), self.F0, self.m0(), self.L0(), self)
                       for _ in range(self.N)]

    def simulate(self, time, reset=True, loop=10):
        '''
        Pass a person object
        '''
        offset = 0 # only non-zero when reset=False
        if reset: # continue existing simulation
            self.initialize_pedestrians()
            self.data = np.zeros((self.N, 2, len(time)))
        else:
            offset = self.data.shape[2] # previous number of time steps
            self.time_history.append(time) # keep track of all potential reruns
            self.data = np.pad(x, [(0,0), (0,0), (0, len(time))])

        initial = self.get_forward_state()

        dt = time[1] - time[0]

        for i in range(self.N):
            self.data[i, :, offset] = initial[i]

        for k in range(offset, len(time) - 1):
            for i in range(self.N):
                if loop:
                    if self.data[i, 0, k] > loop:
                        self.data[i, :, k] = 0

                self.data[i][:, k + 1] = Simulation.rk4(self.people[i].ode, time[k], self.data[i, :, k], dt)

            for i in range(self.N): # update person position and velocity
                self.people[i].z = self.data[i, 0, k + 1]
                self.people[i].v = self.data[i, 1, k + 1]

    def get(self, prop):
        if prop == 'x':
            return np.array([self.x])
        elif prop == 'dx':
            return np.array([self.dx])
        return np.array([getattr(person, prop) for person in self.people])

    def plot_forward_dots(self, frame):
        for i in range(self.N):
            pos, vel = self.data[i, 0:2, frame]
            plt.scatter(pos, 0, color=self.z_color(i), s=100)
            plt.arrow(pos, 0, 0, vel, color=self.v_color(vel/self.vmax()))

    def plot_dots(self, frame):
        for i in range(self.N):
            forward_pos, forward_vel = self.data[i, 0:2, frame]
            lateral_pos, lateral_vel = self.data[i, 2:4, frame]
            plt.scatter(forward_pos, lateral_pos, color=self.z_color(i), s=100)
            plt.arrow(forward_pos, lateral_pos, forward_vel, lateral_vel, color=self.v_color(forward_vel/self.vmax()))

    def plot_dots_from_data(self, frame, forward_pos, forward_vel, lateral_pos, lateral_vel, axs):
        for i in range(self.N):
            axs.scatter(forward_pos[i, frame], lateral_pos[i, frame], color=self.z_color(i), s=100)
            axs.arrow(forward_pos[i, frame], lateral_pos[i, frame], forward_vel[i, frame], lateral_vel[i, frame], color=self.v_color(forward_vel[i, frame]/self.vmax()))

    '''
    integrates wrt the whole system
    incorporates bridge dynamics

    Mx'' + 2hx' + Omega^2 x = -sum_i=1^N m_i y_i''
    y_i'' + H(y_i, y_i', t) = -x''

    where H comes from model 3
    '''
    def ode_bridge_full_optimized(self, t, S, loop):
        '''
        numpy optimized
        '''

        self.history.append(S)

        x = S[0, None]
        dx = S[1, None]
        peds = S[2:][:self.N] # lateral pos
        dpeds = S[2:][self.N:2 * self.N] # lateral vel

        zs = S[2:][2 * self.N:3 * self.N] # forward pos
        vs = S[2:][3 * self.N:4 * self.N] # forward vel

        if loop is not None:
            for i in range(self.N):
                if zs[i] > loop:
                    zs[i] = 0.00
                    vs[i] = 0.01 # need to change their entering velocity


        o = np.sqrt(9.81/self.get('L'))  # self.v_to_o(vs, self.get('L')) # np.sqrt(9.81/self.get('L')) / 2 # 1.04 #  #

        self.x = x
        self.dx = dx
        for i in range(self.N): # update person position and velocity
            self.people[i].z = zs[i]
            self.people[i].v = vs[i]
            self.people[i].y = peds[i]
            self.people[i].dy = dpeds[i]

        M = 113e3

        try: # proportionally scale
            dpeds = dpeds * self.f_max(np.abs(vs), np.abs(self.history[-2][2:][3 * self.N:4 * self.N]))
        except:
            pass

        r = self.get('m') / (M + self.get('m'))

        H = self.H(peds, dpeds, o*self.f_max(np.abs(vs), np.abs(self.get('vmax'))), self.a, self.l, self.p)

        # H = self.H1(peds, o*self.f_max(vs, self.get('vmax')), self.p)

        ddx = self.ddx(x, dx, H, r, self.h, self.O)

        return np.hstack(
            [dx, ddx,
            dpeds,
            -ddx - H,
            vs, self.social(zs, vs, self.get('z'), self.get('vmax'), self.get('alpha'), self.N)])

    def plot_parameters(self):
        parameters = ['m', 'alpha', 'vmax', 'L']
        fig, ax = plt.subplots(1, len(parameters), figsize=(16, 4))
        for p in parameters:
            i = parameters.index(p)
            ax[i].hist(self.get(p))
            ax[i].set_title(p)

        plt.tight_layout()

    @staticmethod
    def parse(S, N):
        bridge, dbridge = S[0:2]
        peds = S[2:][:N] # lateral pos
        dpeds = S[2:][N:2 * N] # lateral vel

        zs = S[2:][2 * N:3 * N] # forward pos
        vs = S[2:][3 * N:4 * N] # forward vel

        return bridge, dbridge, peds, dpeds, zs, vs

    def __iter__(self):
        for i in range(len(self.people)):
            yield self.people[i]

    @staticmethod
    def rk4(f, t, x, h):

        k1 = h * (f(t, x))
        k2 = h * (f((t+h/2), (x+k1/2)))
        k3 = h * (f((t+h/2), (x+k2/2)))
        k4 = h * (f((t+h), (x+k3)))
        k = (k1+2*k2+2*k3+k4)/6
        x = x + k

        return x

class Person(Simulation):
    def __init__(self, z, v, y, dy, vmax, alpha, F, m, L, sim):
        self.z = z # sagittal
        self.v = v # sagittal
        self.y = y # lateral
        self.dy = dy # lateral

        self.vmax = vmax
        self.alpha = alpha
        self.F = F
        self.sim = sim

        self.m = m
        self.L = L

    '''
    z' = v
    v' = a*(v - vmax) - v*sum_j(F(z - zj))
    '''
    def ode(self, t, x):
        z, v = x
        forces = 0
        for person in self.sim.people:
            if self != person:
                forces += self.F(z, person.z)
        return np.hstack([np.array([v]),
        (self.alpha*(v - self.vmax) - v*forces)])


# %%


N = 120
D = 1

def F(zi, zj):
    kappa = 2.0
    c = 5.0

    heaviside = lambda x: 1 if x > 0 else 0
    return kappa*heaviside(zj - zi)*np.exp(-c*(zj - zi))

z0 = lambda: np.random.uniform(low=0, high=1)
v0 = lambda: np.random.normal(loc=0.8, scale=0.1)

y = 0.008
dy = 0.0001
x = 0.000
dx = 0.1

sigma = 0.05
m0 = lambda: np.random.normal(76.9, 10*sigma) # 70
L0 = lambda: np.random.normal(1.17, 0.092*sigma)

vmax = 1.5 # lambda: np.random.normal(1.5, 0.125) # 1.5 # np.random.uniform(1.3, 1.8) # 1.5
alpha = lambda: -0.1 # -1*(0.1*np.random.uniform(0, 1) + 0.01*np.random.uniform(0, 5)) # -0.1

sim = Simulation(N, D, z0, v0, y, dy, m0, L0, vmax, alpha, F, x, dx, 'hsv', 'cool')

sim.plot_parameters()
# %%
y0 = lambda: np.random.uniform(-0.1, 0.1)
dy0 = lambda: np.random.uniform(-0.1, 0.1)

initial = np.array([x, dx] + [y0() for i in range(N)] + [dy0() for i in range(N)] + [z0() for i in range(N)] + [v0() for i in range(N)])
loop = None # 10

# sol, d = odeint(sim.ode_bridge_full, y0=initial, t=time, full_output = 1, tfirst=True, args=(loop, ))
sol = solve_ivp(sim.ode_bridge_full_optimized, t_span=(0, 100), y0=initial, args=(loop, ), method='RK23')
 # %%
time = sol.t
S = sol.y

bridge, dbridge = S[0:2]
peds = S[2:][:N] # lateral pos
dpeds = S[2:][N:2 * N] # lateral vel

zs = S[2:][2 * N:3 * N] # forward pos
vs = S[2:][3 * N:4 * N] # forward vel

# %% lateral positions
sgn = lambda y: 1 - 2 * (y < 0)
# plt.plot(time, sim.p - peds.transpose()*sgn(peds.transpose()));
plt.plot(time, peds.T)
plt.title('lateral position');

# %% lateral velocities
# plt.plot(np.cbrt(vs.transpose() / sim.get('L')**2)*2);

plt.plot(time, dpeds.T);
plt.title('lateral velocity')
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
# (np.sqrt(9.81/ sim.get('L')) / vs.T**0.58).std()
#
#
# sim.f(vs[:, 3]) / sim.f(sim.get('vmax'))
#
# np.sqrt(9.81/ sim.get('L')) * vs.T / sim.get('vmax')
#
# plt.plot(time, sim.v_to_o(vs.T, sim.get('L')));

# %%
plt.plot(time, (bridge.T)); # bridge position
plt.title('bridge position')
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
# f = lambda v, vmax: (0.35*v*v*v - 1.59*v*v + 2.93*v) / (0.35*vmax*vmax*vmax - 1.59*vmax*vmax + 2.93*vmax)
#
# test = np.array([sim.f_max(vs[:, i], sim.get('vmax')) for i in range(len(time))])
#
# for i in range(len(time)):
#     plt.plot(vs[:, i], sim.f_max(vs[:, i], sim.get('vmax')))


 # %%

import matplotlib.animation as animation

fig, axs = plt.subplots(1, 2, figsize=(12, 2), dpi=160, gridspec_kw={'width_ratios': [3, 1]})

vector_color = plt.cm.get_cmap('cool')
def animate(frame):
    frame = frame * 4
    axs[0].cla()
    axs[0].set_title(f'time: {round(time[frame], 2)}')
    axs[0].set_xlim(zs[:, frame].min() - 3, zs[:, frame].max() + 3)
    # axs[0].set_xlim(0, loop)
    axs[0].set_ylim(peds.min() - 0.1, peds.max() + 0.1)
    sim.plot_dots_from_data(frame, zs, vs, peds, dpeds, axs[0])

    axs[1].cla()
    axs[1].set_title('bridge velocity')
    axs[1].plot(time[:frame], dbridge[:frame])

anim = animation.FuncAnimation(fig, animate, frames=len(time))

anim.save('figs/optimized.mp4')
# %% WARNING: took over 6 gb of harddrive space to save all data
import tqdm
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
sigmas = [0.05, 0.175, 0.25, 0.5]# np.linspace(0, 1, 21)
Ns = np.arange(0, 151, 5)
# %%
for sigma in tqdm.tqdm(sigmas):
    m0 = lambda: np.random.normal(76.9, 10*sigma) # 70
    L0 = lambda: np.random.normal(1.17, 0.092*sigma)
    for N in tqdm.tqdm(Ns):
        for _ in range(3):
            sim = Simulation(N, D, z0, v0, y, dy, m0, L0, vmax, alpha, F, x, dx, 'hsv', 'cool')

            initial = [x, dx] + [y0() for i in range(N)] + [dy0() for i in range(N)] + [z0() for i in range(N)] + [v0() for i in range(N)]
            # sol, d = odeint(sim.ode_bridge_full, y0=initial, t=time, full_output = 1, tfirst=True, args=(loop, ))
            sol = solve_ivp(sim.ode_bridge_full_optimized, t_span=(0, 100), y0=initial, args=(loop, ))
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
            bridge, dbridge, peds, dpeds, zs, vs = Simulation.parse(test[1:], Ns[count])

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
    sol = solve_ivp(sim.ode_bridge_full_optimized, t_span=(0, 10), y0=initial, args=(loop, ), method='RK23')
    bridge, dbridge, peds, dpeds, zs, vs = Simulation.parse(sol.y, N)
    plt.plot(bridge[::-1], dbridge[::-1], label='$\dot{x}_0$: ' + '%.1E' % dx, color=plt.get_cmap('Greens')(norm(dx)))

plt.legend(bbox_to_anchor=(1.05, 1))
