# %%
%run -i ../bart/startup

# %% import lib
import numpy as np
import matplotlib.pyplot as plt

# %%
def bitmask(lst: list) -> int:
    """bitmask from list of indices

    Args:
        lst (list): indices

    Returns:
        int: mask
    >>> assert bitmask([0, 2, 3]) == 13
    >>> assert bitmask([]) == 0
    """
    mask = 0
    for i in lst:
        mask |= 1 << i
    return mask
def show_timesteps(data, timesteps):
    time_dimension = 5
    n = len(timesteps)
    # If input is a string → load with bart
    if isinstance(data, str):
        data_arr = bart(1, 'show -m', data)  # or just load with cfl.readcfl if you prefer
    else:
        data_arr = data                      # assume already a numpy array
    
    # Extract individual timesteps
    slices = []
    for i, t in enumerate(timesteps):
        slice_t = bart(1, f'slice {time_dimension} {t}', data_arr)
        slices.append(slice_t)
        # Optional: save to disk like the original _slice$i
        # bart(1, f'toimg _slice{i}', slice_t)
    
    # Stack along new dimension (dimension 6 in BART convention)
    stacked = bart(1, f'join 6', *slices)
    # Clean up temporary variables if you used named ones (not needed here because we pass arrays directly)
    
    # Reshape to make one long row of images: (DIM*n, DIM, 1, ...) → (DIM*n, DIM)
    # Original: bart reshape $(bart bitmask 1 6) $((DIM*ind)) 1 input output
    bitmask = bart(1, 'bitmask 1 6')           # returns [1, 64] → integer 65
    new_dims = f"{dim * n} 1"
    reshaped = bart(1, f'reshape {bitmask} {new_dims}', stacked)
    
    # Convert to magnitude (like imshow usually does)
    magnitude = bart(1, 'rss 8', reshaped)     # or 'mag' if complex → real
    
    # Plot all timesteps side by side
    plt.figure(figsize=(4 * n, 4))
    plt.imshow(magnitude.squeeze(), cmap='gray')
    plt.axis('off')
    plt.title(f"Timesteps: {timesteps}")
    plt.show()

# %%
n_x = 192
n_comp = 11
comp_geom=bart(1, f'phantom -T -x{n_x} -b')
# %%
print(comp_geom.shape)
print(comp_geom.max())
print(comp_geom.min())
# %%
TR=0.0034
REP=400
T1=3
T2=1

NUM_SIMULATIONS=1

comp_water=bart(1, f'signal -F -I -r{TR} -n{REP} -1 {T1}:{T1}:{NUM_SIMULATIONS} -2 {T2}:{T2}:{NUM_SIMULATIONS}')
# %%
print(comp_water.shape)
print(f'max: {np.max(np.real_if_close(comp_water))}, min: {np.min(np.real_if_close(comp_water))}')
plt.plot(np.squeeze(np.real_if_close(comp_water)))
# %%
_comp_tubes=bart(1, f'signal -F -I -r{TR} -n{REP} -1 {0.5}:{2}:{3} -2 {0.005}:{0.2}:{3}')
# %%
print(_comp_tubes.shape)
print(_comp_tubes.min())
print(_comp_tubes.max())
# %%
comp_tubes=bart(1, f'reshape {bitmask([6, 7])} 9 1', _comp_tubes)
# %%
def plt_comp_tubes(line,ax):
    ax.plot(np.real_if_close(line))
comp_tubes_squeeze = comp_tubes.squeeze()
n=comp_tubes_squeeze.shape[-1]
fig, axes = plt.subplots(nrows=n, figsize=(3, 3 * n))
for i in range(n):
    plt_comp_tubes(comp_tubes_squeeze[:,i], axes[i])
# %%
comp_simu=bart(1, f'join 6', comp_water, comp_tubes, comp_water)

# %%
phantom=bart(1, f'fmac -s {bitmask([6])}', comp_geom, comp_simu)

# %%
print(phantom.shape)
print(phantom.max())
print(phantom.min())

# %%
t = (0, 100, 150, 200, 300, 399)
show_timesteps(phantom, t)
# %%
_slices = phantom.squeeze()[:,:, t]
slices = np.transpose(_slices, (2, 0, 1)).reshape(n_x * len(t), n_x)
# %%
plt.imshow(np.abs(slices), cmap='gray')
plt.axis('off')
# %%
