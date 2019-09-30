import numpy as np
import matplotlib.pyplot as plt

header = ["epoch_id","eigenvector_ABS"]
eigerr = np.array( [[32,0.052066717168937],
[48,0.030700074707731],
[64,0.022813341170428],
[80,0.016773069896906],
[96,0.015850476370356],
[112,0.023858656532983],
[128,0.032345770944394],
[144,0.040679204151823],
[160,0.048477275222999],
[176,0.056219202897372],
[192,0.062635218621664],
[208,0.068135724700112],
[224,0.073183325685921],
[240,0.077679463649992]] )


# Get ylim
fig, ax = plt.subplots(1, 1, figsize=(5,4))
ax.set_xlabel('Problem size')
ax.set_ylabel('Absolute Error')
plt.axvline(x=32, c='black', linestyle='--')
plt.axvline(x=128, c='black', linestyle='--')
plt.xticks(eigerr[::2,0])
ax.plot(eigerr[:,0], eigerr[:,1], linestyle='-', marker='o', color = "#8A4F7D")
ymin, ymax = ax.get_ylim()
plt.savefig('test-varying-sizes-app-e.eps', format='eps', dpi=200)
plt.close()
fig, ax = plt.subplots(1, 1, figsize=(5,4))
ax.set_xlabel('Problem size')
ax.set_ylabel('Absolute Error')
plt.axvline(x=32, c='black', linestyle='--')
plt.axvline(x=128, c='black', linestyle='--')
plt.xticks(eigerr[::2,0])
ax.fill_betweenx([0,ymax], 32, 128, alpha=0.1, facecolor='#F7EF81')

ax.plot(eigerr[:,0], eigerr[:,1], linestyle='-', marker='o', color = "#8A4F7D")
ax.set_ylim(0, ymax)
plt.savefig('test-varying-sizes-app-e.eps', format='eps', dpi=200)
