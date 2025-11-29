# %%
import numpy as np
from dipy.viz import window, actor
from dipy.core.sphere import HemiSphere, disperse_charges

# %%
rng=np.random.default_rng()
n_pts=64
theta=rng.random(n_pts)*np.pi
phi=rng.random(n_pts)*2*np.pi
hsph_initial=HemiSphere(theta=theta,phi=phi)

# %%
scene=window.Scene()
scene.add(actor.point(hsph_initial.vertices,window.colors.green,point_radius=0.02))
window.show(scene)
# %%
