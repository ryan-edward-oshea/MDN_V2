# Product Estimation

### About
This repository contains source code for the following papers:

- <i>["Seamless retrievals of chlorophyll-a from Sentinel-2 (MSI) and Sentinel-3 (OLCI) in inland and coastal waters: A machine-learning approach"](https://www.sciencedirect.com/science/article/pii/S0034425719306248). N. Pahlevan, et al. (2020). Remote Sensing of Environment. 111604. 10.1016/j.rse.2019.111604.</i>
- <i>["Robust algorithm for estimating total suspended solids (TSS) in inland and nearshore coastal waters"](https://www.sciencedirect.com/science/article/abs/pii/S0034425720301383). S.V. Balasubramanian, et al. (2020). Remote Sensing of Environment. 111768. 10.1016/j.rse.2020.111768.</i> [Code](https://github.com/BrandonSmithJ/MDN/tree/master/benchmarks/tss/SOLID).
- <i>["Hyperspectral retrievals of phytoplankton absorption and chlorophyll-a in inland and nearshore coastal waters"](https://www.sciencedirect.com/science/article/pii/S0034425720305733). N. Pahlevan, et al. (2021). Remote Sensing of Environment. 112200. 10.1016/j.rse.2020.112200.</i>
- <i>["A Chlorophyll-a Algorithm for Landsat-8 Based on Mixture Density Networks"](https://www.frontiersin.org/articles/10.3389/frsen.2020.623678/full). B. Smith, et al. (2021). Frontiers in Remote Sensing. 623678. 10.3389/frsen.2020.623678.</i>
- <i>["Leveraging multimission satellite data for spatiotemporally coherent cyanoHAB monitoring"](https://www.frontiersin.org/articles/10.3389/frsen.2023.1157609/full). K. Fickas, et al. (2021). Frontiers in Remote Sensing. 1157609. 10.3389/frsen.2023.1157609.</i>
<br>

### Usage
The package can be cloned into a directory with:

`git clone https://github.com/ryan-edward-oshea/MDN_V2.git`

Alternatively, you may use pip to install:

`pip install git+https://github.com/ryan-edward-oshea/MDN_V2`

<br>

The code may then either be used as a library, such as with the following:
```
from MDN import image_estimates, get_tile_data, get_sensor_bands
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import numpy as np
sensor = "S3A"

# Tile should be the output of an atmospheric correction program e.g. SeaDAS
kwargs={'product'  :'chl,tss,cdom,pc',
        'benchmark': False,
        'sat_bands': True}

bands, Rrs = get_tile_data('/path/to/tile.nc', sensor, allow_neg=False,**kwargs)

#Rrs = np.random.rand(4, 5, len(get_sensor_bands(sensor)))

chla_tss_cdom, idxs = image_estimates(Rrs, sensor=sensor,**kwargs)
print(chla_tss_cdom, type(chla_tss_cdom), chla_tss_cdom.shape, idxs)

chla = chla_tss_cdom[:,:,idxs['chl']]
TSS  = chla_tss_cdom[:,:,idxs['tss']]
cdom = chla_tss_cdom[:,:,idxs['cdom']]
pc   = chla_tss_cdom[:,:,idxs['pc']]

print(chla,TSS,cdom)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)

chl_im = ax1.imshow(chla,vmin=0.1, vmax=100, cmap='jet', aspect='auto',norm=LogNorm())
fig.colorbar(chl_im, ax=ax1)
ax1.set_title('Chl')

TSS_im = ax2.imshow(TSS,vmin=0.1, vmax=100, cmap='jet', aspect='auto',norm=LogNorm())
fig.colorbar(TSS_im, ax=ax2)
ax2.set_title('TSS')

cdom_im = ax3.imshow(cdom,vmin=0.1, vmax=1, cmap='jet', aspect='auto',norm=LogNorm())
fig.colorbar(cdom_im, ax=ax3)
ax3.set_title('CDOM')

pc_im = ax4.imshow(pc,vmin=0.1, vmax=100, cmap='jet', aspect='auto',norm=LogNorm())
fig.colorbar(pc_im, ax=ax4)
ax4.set_title('PC')
plt.show()

```

*Note:* The user-supplied input values should correspond to R<sub>rs</sub> (units of 1/sr). 

Current performance is shown in the following scatter plots, with 50% of the data used for training and 50% for testing. Note that the models supplied in this repository are trained using 100% of the <i>in situ</i> data, and so observed performance may differ slightly. 

<p align="center">
	<img src=".res/S2B_benchmark.png?raw=true" height="311" width="721.5"></img>
	<br>
	<br>
	<img src=".res/OLCI_benchmark.png?raw=true" height="311" width="721.5"></img>
		<br>
	<br>
	<img src=".res/OLCI_benchmark_PC.png?raw=true" height="311" width="721.5"></img>
</p>



