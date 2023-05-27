import matplotlib.pyplot as plt
from phi import flow
import numpy as np

# X: inflow force at (x_direction_magnitude,y_direction_magnitude)
def single_query(X, m, N_TIME_STEPS = 30):

    velocity = flow.StaggeredGrid(
        values=(0.0, 0.0),
        extrapolation=0.0,
        x=m,
        y=m,
        bounds=flow.Box(x=50, y=50),
    )
    smoke = flow.CenteredGrid(
        values=0.0,
        extrapolation=flow.extrapolation.BOUNDARY,
        x=m,
        y=m,
        bounds=flow.Box(x=50, y=50),
    )
    inflow = 0.2 * flow.CenteredGrid(
        values=flow.SoftGeometryMask(
            flow.Sphere(
                x=25,
                y=9.5,
                radius=5,
            )
        ),
        extrapolation=0.0,
        bounds=smoke.bounds,
        resolution=smoke.resolution,
    )

 #   @flow.math.jit_compile
    def step(velocity_prev, smoke_prev, X, dt=1.0):
        smoke_next = flow.advect.mac_cormack(smoke_prev, velocity_prev, dt) + inflow
        buoyancy_force = smoke_next * (X[0], X[1]) @ velocity
        velocity_tent = flow.advect.semi_lagrangian(velocity_prev, velocity_prev, dt) + buoyancy_force * dt
        velocity_next, pressure = flow.fluid.make_incompressible(velocity_tent)
        return velocity_next, smoke_next

    vel_res = None
    # %matplotlib tk
    for i in range(N_TIME_STEPS):
        velocity, smoke = step(velocity, smoke, X)
        # vector[1] for y velocity map
        velocity_values_extracted = velocity.at_centers().values.vector[1].numpy('y,x')
        vel_res = velocity_values_extracted
    return vel_res
