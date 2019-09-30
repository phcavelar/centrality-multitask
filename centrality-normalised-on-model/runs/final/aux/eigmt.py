import numpy as np
import matplotlib.pyplot as plt

header = ["epoch_id","eigenvector_ABS"]
eigerr = np.array( [[32,0.044909920762906],
[48,0.02518685488879],
[64,0.020066283596034],
[80,0.019358806121745],
[96,0.022378010784226],
[112,0.032727201791388],
[128,0.041687428860429],
[144,0.049630543123888],
[160,0.056006705477532],
[176,0.061157348567977],
[192,0.065106902463067],
[208,0.068673042067405],
[224,0.071090247809925],
[240,0.073188497606269]] )

# Get ylim
fig, ax = plt.subplots(1, 1, figsize=(5,4))
ax.set_xlabel('Problem size')
ax.set_ylabel('Absolute Error')
plt.axvline(x=32, c='black', linestyle='--')
plt.axvline(x=128, c='black', linestyle='--')
plt.xticks(eigerr[::2,0])
ax.plot(eigerr[:,0], eigerr[:,1], linestyle='-', marker='o', color = "#8A4F7D")
ymin, ymax = ax.get_ylim()
plt.savefig('test-varying-sizes-bcde-abserr.eps', format='eps', dpi=200)
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
plt.savefig('test-varying-sizes-bcde-abserr.eps', format='eps', dpi=200)
