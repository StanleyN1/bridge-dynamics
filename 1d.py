import sympy as smp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, odeint



class Simulation:
    def __init__(self, N, D, z0, v0, y0, dy0, H, m0, L0, vmax, alpha, F, x, dx, z_color=None, v_color=None):
        functionify = lambda x: x if callable(x) else (lambda: x)

        self.N = N
        self.D = D

        self.z0 = functionify(z0)
        self.v0 = functionify(z0)
        self.y0 = functionify(y0)
        self.dy0 = functionify(dy0)

        self.m0 = functionify(m0)
        self.L0 = functionify(L0)

        l = 0.1
        o = 0.7
        p = 0.3
        a = 0.2
        nu = 0.67
        # m = 0.11
        # h = 0.1
        # O = 1.
        self.vsgn = lambda x: 1 if x >= 0 else -1
        self.Hi = lambda y, dy, fv: -(fv*o)**2 * (y - p*self.vsgn(y)) + l*(dy**2 + nu**2 * (a**2 - (y - p*self.vsgn(y)))**2)*dy

        # self.Hi = lambda y, dy, fv: (fv*o)**2 * (y - p*self.vsgn(y)) - l*(dy**2 - (fv*o)**2 * (y - p*self.vsgn(y))**2 + (fv*o)**2 * a**2)*dy


        self.F = F
        self.f = lambda x: 0.35*x**3 - 1.59*x**2 + 2.93*x


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
        self.people = [Person(self.z0(), self.v0(), self.y0(), self.dy0(), self.vmax(), self.alpha(), self.F, self.m0(), self.L0(), self)
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

    def simulate_bridge(self, time, reset=True, loop=10):
        '''
        Pass a person object
        '''
        offset = 0 # only non-zero when reset=False
        if reset: # continue existing simulation
            self.initialize_pedestrians()
            self.data = np.zeros((self.N, 6, len(time)))
        else:
            offset = self.data.shape[2] # previous number of time steps
            self.time_history.append(time) # keep track of all potential reruns
            self.data = np.pad(self.data, [(0,0), (0,0), (0, len(time))])

        initial = np.hstack([self.get_forward_state(), self.get_lateral_state(), np.full_like(self.get_forward_state(), [x, dx])])

        dt = time[1] - time[0]

        for i in range(self.N):
            self.data[i, :, offset] = initial[i]

        for k in range(offset, len(time) - 1):

            for i in range(self.N):
                if loop:
                    if self.data[i, 0, k] > loop:
                        self.data[i, :, k] = 0.0

                self.data[i][:, k + 1] = Simulation.rk4(self.people[i].ode_bridge, time[k], self.data[i, :, k], dt)

            # self.data[:, 4:6, k + 1] = np.full_like(self.get_forward_state(), self.data[:, 4:6, k].sum(0))

            for i in range(self.N): # update person position and velocity
                self.people[i].z = self.data[i, 0, k + 1]
                self.people[i].v = self.data[i, 1, k + 1]
                self.people[i].y = self.data[i, 2, k + 1]
                self.people[i].dy = self.data[i, 3, k + 1]
                self.people[i].x = self.data[i, 4, k + 1]
                self.people[i].dx = self.data[i, 5, k + 1]

    def get(self, prop):
        if prop == 'x':
            return np.array([self.x])
        elif prop == 'dx':
            return np.array([self.dx])
        return np.array([getattr(person, prop) for person in self.people])

    def get_forward_state(self):
        state = np.array([np.array([person.z, person.v]) for person in self.people]).reshape(-1, 2)
        return state

    def get_lateral_state(self):
        state = np.array([np.array([person.y, person.dy]) for person in self.people]).reshape(-1, 2)
        return state

    def get_bridge_state(self):
        state = np.array([self.x, self.dx]).reshape(-1, 2)
        return state

    def get_position(self, idx):
        return self.data[:, 0, idx]

    def get_velocity(self, idx):
        return self.data[:, 1, idx]

    def get_bridge_position(self, idx):
        return self.data[:, 2, idx]

    def get_bridge_velocity(self, idx):
        return self.data[:, 3, idx]

    def plot_forward_position(self, time):
        if self.data[:, 0].any():
            for i in range(self.N):
                plt.plot(time, self.data[i, 0][0:len(time)], color=self.z_color(i))

    def plot_forward_velocity(self, time):
        if self.data[:, 1].any():
            for i in range(self.N):
                plt.plot(time, self.data[i, 1][0:len(time)], color=self.z_color(i))

    def plot_lateral_position(self, time):
        if self.data[:, 2].any():
            for i in range(self.N):
                plt.plot(time, self.data[i, 2][0:len(time)], color=self.z_color(i))

    def plot_lateral_velocity(self, time):
        if self.data[:, 3].any():
            for i in range(self.N):
                plt.plot(time, self.data[i, 3][0:len(time)], color=self.z_color(i))

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

    def ode_bridge_lateral_no_forward(self, t, S):
        x, dx = S[0:2]
        peds = S[2:][:self.N]
        dpeds = S[2:][self.N:]



        M = 5060
        r = self.get('m') / (M + self.get('m').sum())

        f_max = lambda x, i: 1

        H = sum([
            self.people[i].m * self.Hi(peds[i], dpeds[i], f_max(self.people[i].v, i)) for i in range(self.N)
        ])

        h = 0.1
        O = 1.

        ddx = 1/(M - r) * (H - 2*h*dx - O**2 * x)

        return np.hstack(
            [dx, ddx] +
            [
                dpeds[i] for i in range(self.N)
            ] +
            [
                -ddx - self.Hi(peds[i], dpeds[i], f_max(self.people[i].v, i)) for i in range(self.N)
            ])

    def social(self, z, v): # assuming F function is equal
        forces = np.zeros(self.N)
        for i in range(self.N):
            social_force = sum([self.F(z[i], person.z) for person in self.people if self != person])
            forces[i] = self.get('alpha')[i]*(v[i] - self.get('vmax')[i]) - v[i]*social_force

        return forces

    '''
    integrates wrt the whole system
    incorporates bridge dynamics

    Mx'' + 2hx' + Omega^2 x = -sum_i=1^N m_i y_i''
    y_i'' + H(y_i, y_i', t) = -x''

    where H comes from model 3
    '''
    def ode_bridge_full(self, t, S, loop):
        '''
        current full pedestrian-bridge ode simulation
        '''
        np.errstate(all='raise')
        try:
            # self.history.append(S)
            x, dx = S[0:2]
            peds = S[2:][:self.N] # lateral pos
            dpeds = S[2:][self.N:2 * self.N] # lateral vel

            zs = S[2:][2 * self.N:3 * self.N] # forward pos
            vs = S[2:][3 * self.N:4 * self.N] # forward vel

            if loop:
                for i in range(len(zs)):
                    if zs[i] > loop:
                        zs[i] = 0.00
                        vs[i] = 0.25 # need to change their entering velocity

            l = 23.25 # limit_cycle_damping_parameter
            a = 0.47 # limit_cycle_amplitude_parameter
            p = 0.63
            o = 1.04

            self.vsgn = lambda y: 1 - 2 * (y < 0)

            Hi = lambda y, dy, fv: -(fv*o)**2 * (y - p*self.vsgn(y)) + l*(dy**2 + (fv*o)**2 * (a**2 - (y - p*self.vsgn(y))**2))*dy
            # Hi = lambda y, dy, fv: -(o + fv)**2 * (y - p*self.vsgn(y)) + l*(dy**2 + (o + fv)**2 * (a**2 - (y - p*self.vsgn(y))**2))*dy

            f = lambda x: 0.35*x**3 - 1.59*x**2 + 2.93*x

            # implementing idea from carroll et al 2013 (eq 21)
            # f_T = lambda v, i: o * ((f(v) / f(self.people[i].vmax)) - 1)
            # dpeds = (f(vs) / f(self.get('v'))) * dpeds # need to renormalize velocity (use the fact that pedestrian data hasn't been updated)


            self.x = x
            self.dx = dx
            for i in range(self.N): # update person position and velocity
                self.people[i].z = zs[i]
                self.people[i].v = vs[i]
                self.people[i].y = peds[i]
                self.people[i].dy = dpeds[i]

            M = 100000
            r = self.get('m').sum()

            # l = 0.1
            # o = 0.7
            # p = 0.3
            # a = 0.2
            # nu = 0.67

            # parameters from kevin's code


            # f_max = lambda v, i: f(v) / f(self.people[i].vmax)
            # f_max = lambda v, i: v / self.people[i].vmax
            f_max = lambda v, i: 1

            H = sum(
                self.people[i].m * Hi(peds[i], dpeds[i], f_max(self.people[i].v, i)) for i in range(self.N)
            )

            # self.get('m') * Hi(peds[i], dpeds[i], f_max(self.people[i].v, i))
            h = 0.1
            O = 1.

            bridge_hz = 1.03
            O = (bridge_hz*2*np.pi) ** 2
            h = bridge_hz*2*np.pi * 0.04

            ddx = 1/(M - r) * (H - 2*h*dx - O**2 * x)

            return np.hstack(
                [dx, ddx] + \
                [
                    dpeds[i] for i in range(self.N)
                ] + \
                [
                    -ddx - Hi(peds[i], dpeds[i], f_max(self.people[i].v, i)) for i in range(self.N)
                ] + [vs, self.social(zs, vs)])
        except:
            print(x)
            print(dx)

    def plot_parameters(self):
        parameters = ['m', 'alpha', 'vmax', 'L']
        fig, ax = plt.subplots(1, len(parameters), figsize=(16, 4))
        for p in parameters:
            i = parameters.index(p)
            ax[i].hist(self.get(p))
            ax[i].set_title(p)

        plt.tight_layout()

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
N = 10
D = 1

z0 = lambda: np.random.uniform(low=0, high=5)
v0 = lambda: np.random.normal(loc=0.8, scale=0.01)
y = 0.008
dy = 0.0001
x = 0.000
dx = -0.005

m0 = lambda: np.random.uniform(65, 80) # 70
L0 = 1.12

vmax = lambda: np.random.uniform(1.3, 1.8) # 1.5
alpha = lambda: -1*(0.1*np.random.uniform(0, 1) + 0.01*np.random.uniform(0, 5)) # -0.1

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

sim = Simulation(N, D, z0, v0, y, dy, H, m0, L0, vmax, alpha, F, x, dx, 'hsv', 'cool')

time = np.linspace(0, 1, 11)

sim.plot_parameters()
# %%
# sim.simulate(time, reset=True)

# %%
y0 = lambda: np.random.uniform(-0.2, 0.2)
dy0 = lambda: np.random.uniform(-0.1, 0.1)

initial = [x, dx] + [y0() for i in range(N)] + [dy0() for i in range(N)] + [z0() for i in range(N)] + [v0() for i in range(N)]
loop = 5
# sol, d = odeint(sim.ode_bridge_full, y0=initial, t=time, full_output = 1, tfirst=True, args=(loop, ))
sol = solve_ivp(sim.ode_bridge_full, t_span=(0, 150), y0=initial, args=(loop, ))
 # %%
time = sol.t
S = sol.y

bridge, dbridge = S[0:2]
peds = S[2:][:N] # lateral pos
dpeds = S[2:][N:2 * N] # lateral vel

zs = S[2:][2 * N:3 * N] # forward pos
vs = S[2:][3 * N:4 * N] # forward vel
# %% lateral positions
plt.plot(time, peds.transpose());
plt.title('lateral position');
# %% lateral velocities
plt.plot(time, dpeds.transpose());
plt.title('lateral velocity')
# %% phase diagram of lateral direction
plt.plot(peds.transpose(), dpeds.transpose());
plt.title('lateral position vs velocity')
# %%
plt.plot(time, zs.transpose()); # forward position
plt.title('forward position')
# %%
plt.plot(time, vs.transpose()); # lateral position
plt.title('forward velocity')
# %%
plt.plot(time, bridge.transpose()); # bridge position
plt.title('bridge position')
# %%
plt.plot(time, dbridge.transpose()); # bridge position
plt.title('bridge velocity')
# %%
plt.plot(bridge.transpose(), dbridge.transpose()); # bridge position
plt.title('bridge position vs velocity')

# %%
import matplotlib.animation as animation

fig, axs = plt.subplots(1, 2, figsize=(12, 2), dpi=160, gridspec_kw={'width_ratios': [3, 1]})

vector_color = plt.cm.get_cmap('cool')
def animate(frame):
    axs[0].cla()
    axs[0].set_title(f'time: {round(time[frame], 2)}')
    # axs[0].set_xlim(zs[:, frame].min() - 3, zs[:, frame].max() + 3)
    axs[0].set_xlim(0, loop)
    axs[0].set_ylim(peds.min() - 0.1, peds.max() + 0.1)
    sim.plot_dots_from_data(frame, zs, vs, peds, dpeds, axs[0])

    axs[1].cla()
    axs[1].set_title('bridge velocity')
    axs[1].plot(time[:frame], dbridge[:frame])

anim = animation.FuncAnimation(fig, animate, frames=len(time) - 150)

anim.save('figs/2d-social-loop-N=100.mp4')
# %%
