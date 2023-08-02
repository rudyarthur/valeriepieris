import numpy as np
from valeriepieris import valeriepieris

input_data = np.loadtxt("gpw_v4_population_count_rev11_2020_1_deg.asc", skiprows=6 )
input_data[ input_data < 0] = 0


###Basic
data_bounds = [ -90,90, -180,180 ] ##[lowest lat, highest lat, lowest lon, highest lon]
target_fracs = [0.25, 0.5, 1]
rmin, smin, best_latlon, data, new_bounds  = valeriepieris(input_data,  data_bounds, target_fracs)		

for i,f in enumerate(target_fracs):
	print("At f={}, radius={}, population={}, centre={}".format( f, rmin[i], smin[i], best_latlon[i] ) )

###Focus on europe
data_bounds = [ -90,90, -180,180 ]
europe_bounds = [ 34.1,80, -25,34.9 ] 
target_fracs = [0.5]
rmin, smin, best_latlon, europe_data, europe_data_bounds  = valeriepieris(input_data,  data_bounds, 0.5, target_bounds=europe_bounds)		

for i,f in enumerate(target_fracs):
	print("At f={}, radius={}, population={}, centre={}".format( f, rmin[i], smin[i], best_latlon[i] ) )
print("data in ", europe_data_bounds, "has shape", europe_data.shape)


###Narrow the search
data_bounds = [ -90,90, -180,180 ] ##[lowest lat, highest lat, lowest lon, highest lon]
target_fracs = [0.5]
search_bounds = [ 24,50, -125, -66 ] #~continental US
rmin, smin, best_latlon, data, new_bounds  = valeriepieris(input_data,  data_bounds, target_fracs, search_bounds=search_bounds)		

for i,f in enumerate(target_fracs):
	print("At f={}, radius={}, population={}, centre={}".format( f, rmin[i], smin[i], best_latlon[i] ) )


##Plots
data_bounds = [ -90,90, -180,180 ] ##[lowest lat, highest lat, lowest lon, highest lon]
target_fracs = [0.5]
rmin, smin, best_latlon, data, new_bounds  = valeriepieris(input_data,  data_bounds, target_fracs)		

from valeriepieris import precompute_trig
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import cartopy.geodesic as geodesic
from shapely.geometry import Polygon

projection = ccrs.Robinson()
	
fig = plt.figure( figsize=(20,10) )
ax = plt.axes(projection=projection)
ax.set_global()
ax.coastlines()
		
row, col = data.shape
pt = precompute_trig(*new_bounds, row, col, centered=False)



for i,f in enumerate(target_fracs):	
	for c in best_latlon[i]:
		circle_points = geodesic.Geodesic().circle(lat= c[0], lon=c[1], radius=rmin[i]*1000, n_samples=100, endpoint=False )
		geom = Polygon(circle_points)
		ax.add_geometries((geom,), crs=ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=2, zorder=2 )
		ax.add_geometries((geom,), crs=ccrs.PlateCarree(), facecolor='m', alpha=0.1, edgecolor='none', zorder=2) 
		ax.scatter( [c[1]], [c[0]], transform=ccrs.PlateCarree(), c='k', zorder=3, s=100, marker='o', label="VP centre = ({:.2f},{:.2f})".format(f, c[0], c[1]) + "\nVP radius = {}km".format(round(rmin[i])) )

popvals = ax.pcolormesh(pt['lon'], pt['lat'], data, transform=ccrs.PlateCarree(), cmap="jet", zorder=1, norm=LogNorm(vmin=1,vmax=np.max(data))  )
cb = fig.colorbar(popvals, ax=ax)
cb.set_label('Population', size=16)
cb.ax.tick_params(labelsize=16) 

ax.legend(loc="upper left", prop={'size': 16})
plt.tight_layout()		
plt.savefig("test_global.png", dpi=fig.dpi)
plt.show()	
plt.close()
