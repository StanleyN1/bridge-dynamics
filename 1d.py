import sympy as smp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, odeint

class Person(Simulation):
    def __init__(self, z, v, vmax, alpha, F, sim):
        self.z = z
        self.v = v
        self.vmax = vmax
        self.alpha = alpha
        self.F = F
        self.sim = sim

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

class Simulation:
    def __init__(self, N, D, z0, v0, vmax, alpha, F, z_color=None, v_color=None):
        functionify = lambda x: x if callable(x) else (lambda: x)

        self.N = N
        self.D = D

        self.z0 = functionify(z0)
        self.v0 = functionify(z0)

        self.F = F

        # currently constants
        self.vmax = functionify(vmax)
        self.alpha = functionify(alpha)

        # list of pedestrians
        self.initialize_pedestrians()
        functionify(None)()

        self.z_color = plt.cm.get_cmap(z_color, N)
        self.v_color = plt.cm.get_cmap(v_color)

        self.time_history = []

    def initialize_pedestrians(self):
        self.people = [Person(self.z0(), self.v0(), self.vmax(), self.alpha(), self.F, self) for _ in range(self.N)]



    def simulate(self, time, reset=True, loop=10):
        '''
        Pass a person object
        '''
        offset = 0 # only non-zero when reset=False
        if reset: # continue existing simulation
            self.initialize_pedestrians()
            self.x = np.zeros((self.N, 2, len(time)))
        else:
            offset = x.shape[2] # previous number of time steps
            self.time_history.append(time) # keep track of all potential reruns
            self.x = np.pad(x, [(0,0), (0,0), (0, len(time))])

        initial = self.get_state()

        dt = time[1] - time[0]

        for i in range(self.N):
            self.x[i, :, offset] = initial[i]

        for k in range(offset, len(time) - 1):
            for i in range(self.N):
                if loop:
                    if self.x[i, 0, k] > loop:
                        self.x[i, :, k] = 0

                self.x[i][:, k + 1] = Simulation.rk4(self.people[i].ode, time[k], self.x[i, :, k], dt)

            for i in range(self.N): # update person position and velocity
                self.people[i].z = self.x[i, 0, k + 1]
                self.people[i].v = self.x[i, 1, k + 1]

    def get_state(self):
        state = np.array([np.array([person.z, person.v]) for person in self.people]).reshape(-1, 2)
        return state

    def get_position(self):
        return self.get_state()[:, 0]

    def get_velocity(self):
        return self.get_state()[:, 1]

    def plot_position(self, time):
        if self.x[:, 0].any():
            for i in range(self.N):
                plt.plot(time, self.x[i, 0][0:len(time)], color=self.z_color(i))

    def plot_velocity(self, time):
        if self.x[:, 1].any():
            for i in range(self.N):
                plt.plot(time, self.x[i, 1][0:len(time)], color=self.v_color(i))

    def plot_dots(self, frame):
        for i in range(self.N):
            pos, vel = self.x[i, :, frame]
            plt.scatter(pos, 0, color=self.z_color(i), s=100)
            plt.arrow(pos, 0, 0, vel, color=self.v_color(vel/self.vmax()))

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

# %%
N = 15
D = 1
z0 = lambda: np.random.uniform(low=0, high=10, size=D)
v0 = lambda: np.random.normal(loc=1.4, scale=0.2, size=D) / 2

vmax = 3.0
alpha = -0.1

def F(zi, zj):
    kappa = 2.0
    c = 5.0

    heaviside = lambda x: 1 if x > 0 else 0
    return kappa*heaviside(zj - zi)*np.exp(-c*(zj - zi))

sim = Simulation(N, D, z0, v0, vmax, alpha, F, 'hsv', 'cool')

time = np.linspace(0, 10, 101)
sim.simulate(time, reset=True)

# %% positions
sim.plot_position(time)
# %%
import matplotlib.animation as animation

fig, axs = plt.subplots(figsize=(12, 2), dpi=120)

vector_color = plt.cm.get_cmap('cool')
def animate(frame):
    plt.cla()
    plt.xlim(0, 10)
    plt.ylim(0, 5)
    sim.plot_dots(frame)

anim = animation.FuncAnimation(fig, animate, frames=len(time) - 50)

anim.save('1d-class-test-50.mp4')
