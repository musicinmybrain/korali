#!/usr/bin/env python

import numpy as np

import abf

if __name__ == '__main__':
    sim = abf.Swimmers()
    ABFs = sim.ABFs

    wcs = [a.stepOutFrequency() for a in ABFs]

    # velocity matrix
    V = np.zeros((len(wcs), len(wcs)))

    for i, a in enumerate(ABFs):
        for j, wc in enumerate(wcs):
            V[i,j] = a.velocity(wc)

    Vinv = np.linalg.inv(V)
    X = np.array(sim.positions, copy=True)

    def travel_time(n1):
        """
        n1: direction of the first gather.
        """
        n1 /= np.sqrt(np.sum(n1**2))
        n2 = np.array([n1[1], -n1[0]])

        def tt(n):
            d = np.dot(X, n)
            betas = np.dot(Vinv, d)
            return np.sum(np.abs(betas))
        return tt(n1) + tt(n2)

    # optimize from simple area bombing (it s 1D)
    thetas = np.linspace(0, np.pi, 512)
    travel_times = np.zeros_like(thetas)
    for i, theta in enumerate(thetas):
        n1 = np.array([np.cos(theta), np.sin(theta)])
        travel_times[i] = travel_time(n1)

    theta = thetas[np.argmin(travel_times)]
    print(f"Optimal theta = {theta}")

    # construct the trajectory
    n1 = np.array([np.cos(theta), np.sin(theta)])
    n2 = np.array([n1[1], -n1[0]])


    sim.reset()

    def execute(n):
        d = np.dot(X, n)
        betas = np.dot(Vinv, d)
        steps = 0
        reward = 0

        for b, wc in zip(betas, wcs):
            nsteps = int((abs(round(b / sim.dt))))
            s = np.sign(b)
            action = [-s*n[0], -s*n[1], wc]
            steps += nsteps
            for step in range(nsteps):
                sim.advance(action)
                reward = reward + sim.getReward()
        print(f"reward: {reward}")
        return steps


    tot_steps = 0
    tot_steps += execute(n1)
    tot_steps += execute(n2)
    sim.dumpTrajectoryToCsv("optimal.csv")

    print(f"took {tot_steps} steps")
