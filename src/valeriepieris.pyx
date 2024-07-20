from libc.math cimport sin, asin, cos, atan2, sqrt, pow, abs
import numpy as np
cimport numpy as cnp

cdef double WGS84_RADIUS = 6378.137
cdef double EARTH_RADIUS = WGS84_RADIUS #6371.0087714150598 #mean earth radius (2\\ *a* + *b*)/3
cdef double PI = 3.141592653589793
cdef double hC = 20015.114352233686	#half of the earth's circumference
cdef double PId2 = 1.5707963267948966	#pi/2
cdef double deg2rad = 0.017453292519943295	
cdef double rad2deg = 57.29577951308232


	
def distance_flat(xlat1, xlon1, xlat2, xlon2):
	#https://www.themathdoctors.org/distances-on-earth-3-planar-approximation/
	cdef double a = PI/2 - (deg2rad * xlat1)
	cdef double b = PI/2 - (deg2rad * xlat2)
	cdef double c = sqrt(a*a + b*b - 2*a*b*cos( deg2rad*(xlon2-xlon1) ))

	return EARTH_RADIUS*c
				
	
def distance_haversine(xlat1, xlon1, xlat2, xlon2):

	cdef double lat1 = deg2rad * xlat1
	cdef double lon1 = deg2rad * xlon1
	cdef double lat2 = deg2rad * xlat2
	cdef double lon2 = deg2rad * xlon2

	cdef double sin_dlat = sin((lat2-lat1)*0.5)
	cdef double sin_dlon = sin((lon2-lon1)*0.5)
	cdef double cos_lat1 = cos(lat1)
	cdef double cos_lat2 = cos(lat2)

	cdef h =  sqrt(sin_dlat*sin_dlat + cos_lat1*cos_lat2*sin_dlon*sin_dlon)
	if h > 1: return EARTH_RADIUS*PI	
	return 2*EARTH_RADIUS*asin(h)
				
				
def distance_greatcircle(xlat1, xlon1, xlat2, xlon2):

	cdef double lat1 = deg2rad * xlat1
	cdef double lon1 = deg2rad * xlon1
	cdef double lat2 = deg2rad * xlat2
	cdef double lon2 = deg2rad * xlon2

	cdef double sin_lat1 = sin(lat1)
	cdef double cos_lat1 = cos(lat1)
	cdef double sin_lat2 = sin(lat2)
	cdef double cos_lat2 = cos(lat2)

	cdef double delta_lon = lon2 - lon1
	cdef double cos_delta_lon = cos(delta_lon)
	cdef double sin_delta_lon = sin(delta_lon)
	
	return EARTH_RADIUS * atan2(
		sqrt(   (cos_lat2 * sin_delta_lon) * (cos_lat2 * sin_delta_lon)  
			  + (cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * cos_delta_lon) * (cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * cos_delta_lon)
		),
		sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_delta_lon)

def grid_areas(row, col):
	#difference of spherical caps
	#A = R² (sin φ1 − sin φ2) (θ1 − θ2).
	latitudes = np.deg2rad( np.linspace(90, -90, row+1) )
	return 2*np.pi*EARTH_RADIUS*EARTH_RADIUS*(np.sin(latitudes[:col-1]) - np.sin(latitudes[1:]))* (1/(col))


def precompute_trig(llat, ulat, llon, ulon, row_, col_, centered=False, asnp=True):
	'''
	Does some simple trigonometry once for reuse

			Parameters:
					llat (double): latitude lower bound
					ulat (double): latitude upper bound
					llon (double): longitude lower bound
					ulon (double): longitude upper bound
					row (int): number of lat bins
					col (int): number of lon bins
					centered (bool): place the grid point in the centre of the grid (true) or corner (false)
					asnp (bool): return a dict of numpy arrays
			Returns:
					dict or tuple of arrays
	'''
	cdef int i

	row = row_
	col = col_
	symm_lon = 0
	if ulon == 180 and ulon == -llon: symm_lon = 1
	
	c = 0
	cdef double lat_step = (ulat - llat)/row
	if centered: c = 0.5*lat_step
	latitudes = llat + np.arange(row,0,-1)*lat_step - c
		
	cdef double lon_step = (ulon - llon)/col
	if centered: c = 0.5*lon_step
	longitudes = llon + np.arange(col)*lon_step + c 
	
	sin_latitudes = np.zeros(row)
	cos_latitudes = np.zeros(row)
	for i in range(row): sin_latitudes[i] = sin(latitudes[i]*deg2rad)
	for i in range(row): cos_latitudes[i] = cos(latitudes[i]*deg2rad)	
	vdist = np.arange(row)*EARTH_RADIUS*(ulat-llat)*deg2rad/row #better to use this rather than recompute for floating point comparisons, not ideal

	dlat = np.zeros(row)
	dlon = np.zeros(col)
	
	if asnp:
		return {'lat': np.array(latitudes), 'lon': np.array(longitudes), 'sin_lat': np.array(sin_latitudes), 'cos_lat': np.array(cos_latitudes), 'vdist': np.array(vdist), 
		'dlon':dlon, 'dlat':dlat , 'symm_lon':symm_lon, 'centered':centered, 'row':row, 'col':col, 'lat_step':lat_step, 'lon_step':lon_step}
	cdef double[:] latitudes_ = latitudes
	cdef double[:] longitudes_ = longitudes
	cdef double[:] sin_latitudes_ = sin_latitudes
	cdef double[:] cos_latitudes_ = cos_latitudes
	cdef double[:] vdist_ = vdist
	cdef double[:] dlat_ = dlat
	cdef double[:] dlon_ = dlon
	return latitudes_, longitudes_, sin_latitudes_, cos_latitudes_, vdist_, dlat_, dlon_, row, col, symm_lon, lat_step, lon_step


def latlon_to_id(lat, lon, pt):
	'''
	Find the closest grid points to lat, lon

			Parameters:
					lat (double): latitude 
					lon (double): longitude 
					pt (dict): output of precompute_trig

			Returns:
					int, int
	'''
	clat = np.argmin( np.abs(pt['lat'] - lat ) )
	clon = np.argmin( np.abs(pt['lon'] - lon ) )
	return clat, clon


def id_to_latlon(clat_, clon_, pt):	
	'''
	Find the lat, lon at grid points

			Parameters:
					clat_ (double): latitude id
					clon_ (double): longitude id
					pt (dict): output of precompute_trig

			Returns:
					double, double
	'''	
	cdef int clat = clat_
	cdef int clon = clon_
	return pt['lat'][clat], pt['lon'][clon]

def snap_to_grid(lat, lon, pt):
	'''
	Find the closest grid points to lat, lon

			Parameters:
					lat (double): latitude 
					lon (double): longitude 
					pt (dict): output of precompute_trig

			Returns:
					double, double
	'''
	clat, clon = latlon_to_id(lat, lon, pt)
	return id_to_latlon( clat, clon, pt )


	
		


		
			
cdef void grid_distances_from_id(int lat_id,  int lon_id, double [:, :] distance, 
 int row,
 int col,
 int symm_lon,
 double[:] latitudes,
 double[:] longitudes,
 double[:] sin_latitudes,
 double[:] cos_latitudes,
 double[:] vdist,
 double[:] dlat,
 double[:] dlon ):

	
	cdef double xlat = latitudes[lat_id]
	cdef double xlon = longitudes[lon_id]
		
	cdef int i, j, lat_i, lon_j, flip_lon
	cdef int hcol = col//2 + 1
	cdef int remaining_col_l = 0
	cdef int remaining_col_r = 0
	
	if not symm_lon:
		#rdist = longitudes[col-1] - xlon
		#ldist = xlon - longitudes[0]
		#if rdist < ldist:
		if longitudes[col-1] + longitudes[0] < 2*xlon: 
			hcol = col - lon_id
			remaining_col_l = 0
			remaining_col_r = lon_id - hcol + 1
		else:
			hcol = lon_id + 1
			remaining_col_l = lon_id + hcol
			remaining_col_r = col


	cdef double cos_lat2 = cos(deg2rad*xlat)
	cdef double h;

	for i in range(row): 
		h = sin(deg2rad*(xlat - latitudes[i])*0.5)
		dlat[i] = h*h

	for j in range(col):
		h = sin(deg2rad*(xlon - longitudes[j])*0.5)
		dlon[j] = h*h

	cdef int max_id = row-lat_id
	lat_i = lat_id
	for i in range(row):
		lon_j = lon_id		
		#I'm avoiding if/else inside loops because of https://en.algorithmica.org/hpc/pipelining/branchless/
		distance[lat_i,lon_j] = vdist[(i<max_id)*(2*i-row) + (row-i)] 
		
		lon_j = (lon_j + 1)%col
		for j in range(1,hcol):		

			h = min(1,sqrt(dlat[lat_i] + cos_latitudes[lat_i]*cos_lat2*dlon[lon_j]))				
			distance[lat_i,lon_j] = 2*EARTH_RADIUS*asin(h)		#approx sin is probably fine!
			distance[lat_i, (col + lon_id - j )%col] = distance[lat_i,lon_j] 

			lon_j = (lon_j + 1)%col
			
		for j in range(remaining_col_l,remaining_col_r):		

			h = min(1,sqrt(dlat[lat_i] + cos_latitudes[lat_i]*cos_lat2*dlon[lon_j]))				
			distance[lat_i,lon_j] = 2*EARTH_RADIUS*asin(h)		#approx sin is probably fine!

			lon_j = (lon_j + 1)%col
				
		lat_i = (lat_i + 1)%row


	
def call_grid_distances_from_id(distance, lat_id, lon_id, pt):
	'''
	Find the (haversine) distance from every point to lat_id, lon_id

			Parameters:
					distance (np.array): array to fill
					lat_id (int): latitude 
					lon_id (int): longitude 
					pt (dict): output of precompute_trig

			Returns:
					void
	'''
	global row, col, symm_lon, latitudes, longitudes, sin_latitudes, cos_latitudes, vdist, dlat, dlon

	cdef int xlat = lat_id
	cdef int xlon = lon_id
	cdef double [:,:] distance_view = distance
	grid_distances_from_id(xlat, xlon, distance_view, pt['row'], pt['col'], pt['symm_lon'], pt['lat'], pt['lon'], pt['sin_lat'], pt['cos_lat'], pt['vdist'], pt['dlat'], pt['dlon'])


	
	
	
	
	
	
	
	
	

	
		
cdef void expand(double [:, :] distance, double [:,:] data, cnp.int32_t [:,:] ew, double [:] target_vars, int row, int col, int clat, int clon, int symm, int torad, double new_rad):
	
	cdef double target = target_vars[0]
	cdef double rad = target_vars[1]
	cdef double dsum = target_vars[2]
	cdef double rad_inc = target_vars[3]
	cdef double max_dist = target_vars[4]

	cdef int mod = 2*col - symm*col
	cdef int hcol = col//(1 + symm)
	if torad: 
		rad_inc = 0
		rad = new_rad
	else:
		if dsum <= 0:
			rad -= rad_inc
			
	cdef double old_dsum
	
	cdef int it=0
	cdef int u, ud, llon, rlon, llond, rlond, ulat
	

	while (not torad and (dsum < target and rad <= max_dist)) or (torad and (it<1)):

		rad += rad_inc
		it += 1
		
		for ud in range(-1,2,2):	
			u = (ud+1)//2	 
			ulat = clat+u
				
			while ulat < row and ulat >= 0 and distance[ulat, 0] <= rad: #if a row of zeros was added remember and subtract off the trailing zeros TODO, relevant for p = 100%
				#go right
				rlon = (clon + ew[ulat,1])%(2*col - symm*col)
				rlond = ew[ulat,1]
				while rlon < col and ew[ulat,1] < hcol and distance[ulat, rlond] <= rad:
					
					dsum += data[ulat, rlon]
					ew[ulat,1] += 1
					
					rlon = (clon + ew[ulat,1])%mod
					rlond = ew[ulat,1]
					
				#go left
				llon = (clon-1-ew[ulat,0]+symm*col)%mod
				llond = 1+ew[ulat,0]
				while llon >= 0 and ew[ulat,0] < hcol and distance[ulat, llond] <= rad: ##too far away or all around the small circle

					dsum += data[ulat, llon]
					ew[ulat,0] += 1
					
					llon = (clon-1-ew[ulat,0]+symm*col)%mod
					llond = 1+ew[ulat,0]				
							
				u += ud
				ulat = clat+u
			
	target_vars[1] = rad
	target_vars[2] = dsum
	
	

def call_expand(distance, data, ew, target, rad, rad_inc, dsum, clat, clon, symm=False, torad=False, new_rad=0):
	
	cdef double max_dist = np.max(distance)
	cdef double [:] target_vars = np.array([target, rad, dsum, rad_inc, max_dist])
	r, c = distance.shape
	cdef int row = r
	cdef int col = c
	cdef int lat = clat
	cdef int lon = clon
	cdef double xnew_rad = new_rad
	
	cdef double [:,:] distance_view = distance
	cdef double [:,:] data_view = data
	cdef cnp.int32_t [:,:] ew_view = ew

	expand(distance_view, data_view, ew_view, target_vars, row, col, lat, lon, symm , torad, xnew_rad)
	return target_vars[1], target_vars[2]


cdef void rshift(double [:, :] distance, double [:,:] data, cnp.int32_t [:,:] ew, double [:] target_vars, int row, int col, int clat, int clon, int symm):

	cdef double rad = target_vars[1]
	cdef double dsum = target_vars[2]

	cdef int mod = 2*col - symm*col
	cdef int hcol = col//(1 + symm)

	cdef int u, ud, sublon, addlon, sublond, addlond, ulat
	
	for ud in range(-1,2,2):	
		
		u = (1+ud)//2	 
		ulat = clat + u

		while ulat < row and ulat >= 0 and distance[ulat, 0] <= rad:		

			if symm and ew[ulat,0] == hcol and ew[ulat,0] == ew[ulat,1]: 
				u += ud
				ulat = clat+u
				continue;

			#sublon is left
			sublon = (clon-1-ew[ulat,0]+symm*col)%mod
			sublond = (1+ew[ulat,0])

			while sublon >= 0 and sublon < col and distance[ulat, sublond] > rad: 
				dsum -= data[ulat, sublon]
				ew[ulat,0] -= 1
				
				sublon = (clon-1-ew[ulat,0]+symm*col)%mod
				sublond = 1+ew[ulat,0]	

			ew[ulat,0] += 1

			##addlon is right			
			ew[ulat,1] -= 1
			addlon = (clon + ew[ulat,1]+symm*col)%mod		
			addlond = ew[ulat,1]

			while addlon < col and addlon >= 0 and distance[ulat, addlond] <= rad: 					
				dsum += data[ulat, addlon]				
				ew[ulat,1] += 1
				
				addlon = (clon + ew[ulat,1]+symm*col)%mod 			
				addlond = ew[ulat,1]	

			
			u += ud
			ulat = clat+u
				
	target_vars[2] = dsum	

def call_rshift(distance, data, ew, xdsum, xclat, xclon, xrad, symm=True):
	cdef double [:] target_vars = np.array([0,xrad, xdsum])
	r, c = distance.shape
	cdef int row = r
	cdef int col = c
	cdef int clat = xclat
	cdef int clon = xclon

	cdef double [:,:] distance_view = distance
	cdef double [:,:] data_view = data
	cdef cnp.int32_t [:,:] ew_view = ew

	rshift( distance_view, data_view, ew_view, target_vars, row, col, clat,  clon, symm )
	return target_vars[2]
	
#######################	
	
cdef void lshift(double [:, :] distance, double [:,:] data, cnp.int32_t [:,:] ew, double [:] target_vars, int row, int col, int clat, int clon, int symm):

	cdef double rad = target_vars[1]
	cdef double dsum = target_vars[2]

	cdef int mod = 2*col - symm*col
	cdef int hcol = col//(1 + symm)

	cdef int u, ud, sublon, addlon, sublond, addlond, ulat

	for ud in range(-1,2,2):	
		
		u = (1+ud)//2	 
		ulat = clat + u

		while ulat < row and ulat >= 0 and distance[ulat, 0] <= rad:		

			if symm and ew[ulat,0] == hcol and ew[ulat,0] == ew[ulat,1]: 
				u += ud
				ulat = clat+u
				continue;

			#sublon is right
			sublon = (clon+ew[ulat,1]+symm*col)%mod
			sublond = ew[ulat,1]

			while sublon >= 0 and sublon < col and distance[ulat, sublond] > rad: ##too far away or all around the small circle	
				
				dsum -= data[ulat, sublon]
				ew[ulat,1] -= 1
				sublon = (clon+ew[ulat,1]+symm*col)%mod
				sublond = ew[ulat,1]	

			ew[ulat,1] += 1

			##addlon is left		
			ew[ulat,0] -= 1
			addlon = (clon -1 - ew[ulat,0]+symm*col)%mod		
			addlond = 1+ew[ulat,0]
			
			while addlon < col and addlon >= 0 and distance[ulat, addlond] <= rad: 					
				
				
				dsum += data[ulat, addlon]				
				ew[ulat,0] += 1
				addlon = (clon -1 - ew[ulat,0]+symm*col)%mod		
				addlond = 1+ew[ulat,0]	
			
			u += ud
			ulat = clat+u

	target_vars[2] = dsum	

def call_lshift(distance, data, ew, xdsum, xclat, xclon, xrad, symm=True):
	cdef double [:] target_vars = np.array([0,xrad, xdsum])
	r, c = distance.shape
	cdef int row = r
	cdef int col = c
	cdef int clat = xclat
	cdef int clon = xclon

	cdef double [:,:] distance_view = distance
	cdef double [:,:] data_view = data
	cdef cnp.int32_t [:,:] ew_view = ew

	lshift( distance_view, data_view, ew_view, target_vars, row, col, clat,  clon, symm )
	return target_vars[2]
	
#######################	


cdef void vshift(double [:, :] distance, double [:,:] data, cnp.int32_t [:,:] ew, double [:] target_vars, int row, int col, int clat, int clon, int symm):
			
				
	cdef double rad = target_vars[1]
	cdef double dsum = target_vars[2]	
		
	cdef int mod = 2*col - symm*col
	cdef int hcol = col//(1 + symm)
		
	cdef int ud, u, ulat, llon, rlon, rlond, llond, l, add, sgn
					
	for ud in range(-1,2,2):
		u = (1+ud)//2	 
		ulat = clat + u
		
		while ulat < row and ulat >= 0 and (distance[ulat, 0] <= rad or ew[ulat,1] > 0 or ew[ulat,0] > 0): 


			
			#go right
			l = max(0,ew[ulat,1]-1) 
			rlon = ( clon + l +symm*col)%mod
			rlond = l


			add = (  (not ew[ulat,1]) or (distance[ulat, rlond] <= rad) ) 
			if add and ew[ulat,1]: l += 1
			
			rlon = ( clon + l +symm*col)%mod				
			rlond = l
			
			sgn = (2*add-1)		

			while rlon >= 0 and rlon < col and l < hcol and l >= 0:
			
				if add and distance[ulat, rlond] > rad:	
					break
				elif not add and distance[ulat, rlond] <= rad:
					l -= sgn
					break
					
				dsum += data[ulat, rlon]*sgn				
				l += sgn
				rlon = (clon + l + symm*col)%mod
				rlond = l
			
			ew[ulat, 1] = max(0,l) 


						
			if not add and (ew[ulat,0] == 0 and ew[ulat,1] <= 1): 
				u += ud
				ulat = clat+u
				continue
			
			#go left
			l = max(0,ew[ulat,0]-1) 
			llon = (clon-1-l+symm*col)%mod
			llond = l+1

			add = ( (not ew[ulat,0])  or (distance[ulat, llond] <= rad) ) 			
			if add and ew[ulat,0]: l += 1
			llon = (clon-1-l+symm*col)%mod
			llond = l+1
			
			sgn = (2*add-1)
			while llon >= 0 and llon < col and l < hcol and l >= 0:
				
				if add and distance[ulat, llond] > rad:	
					break
				elif not add and distance[ulat, llond] <= rad:
					l -= sgn
					break				
					
				dsum += data[ulat, llon]*sgn				
				l += sgn
				
				llon = (clon-1-l+symm*col)%mod
				llond = l+1
			
			ew[ulat,0]  = max(0,l)

			u += ud
			ulat = clat+u

	target_vars[2] = dsum	

def call_vshift(distance, data, ew, xdsum, xclat, xclon, xrad, symm=True):
	cdef double [:] target_vars = np.array([0,xrad, xdsum])
	r, c = distance.shape
	cdef int row = r
	cdef int col = c
	cdef int clat = xclat
	cdef int clon = xclon

	cdef double [:,:] distance_view = distance
	cdef double [:,:] data_view = data
	cdef cnp.int32_t [:,:] ew_view = ew

	vshift( distance_view, data_view, ew_view, target_vars, row, col, clat,  clon, symm)

	return target_vars[2]

	
	



			
cdef void shrink(double [:, :] distance, double [:,:] data, cnp.int32_t [:,:] ew, double [:] target_vars, int row, int col, int clat, int clon, int symm, int torad, double new_rad):

	cdef double target = target_vars[0]
	cdef double rad = target_vars[1]
	cdef double dsum = target_vars[2]	
	cdef double rad_inc = target_vars[3]
	cdef double max_dist = target_vars[4]


	cdef int mod = 2*col - symm*col
	cdef int hcol = col//(1 + symm)


	if torad: 
		rad_inc = 0
	else:
		new_rad = rad
		if rad == 0:
			rad += rad_inc
			
			
	cdef int it=0
	cdef int u, ud, llon, rlon, llond, rlond, ulat
	cdef double old_dsum

	while (not torad and (dsum > target and new_rad >= 0)) or (torad and (it<1)):
		new_rad -= rad_inc
		old_dsum = dsum
		it += 1	

		for ud in range(-1,2,2):		
			u = (1+ud)//2	 	 
			ulat = clat+u
		
			while ulat < row and ulat >= 0 and distance[ulat, 0] <= rad: #all the lats where we possibly have included data
				
				#go right
				ew[ulat,1] -= 1
				rlon = (clon + ew[ulat,1])%mod
				rlond = ew[ulat,1]
				
				while ew[ulat,1] < hcol and ew[ulat,1] >= 0 and distance[ulat, rlond] > new_rad: 
					dsum -= data[ulat, rlon]
					ew[ulat,1] -= 1
					rlon = (clon + ew[ulat,1])%mod
					rlond = ew[ulat,1]

				ew[ulat,1] += 1
				
				#go left				
				ew[ulat,0] -= 1
				llon = (clon-1-ew[ulat,0]+symm*col)%mod
				llond = ew[ulat,0]+1
				
				while ew[ulat,0] < hcol and ew[ulat,0] >= 0 and distance[ulat, llond] > new_rad: ##too far away or all around the small circle
					dsum -= data[ulat, llon]
					ew[ulat,0] -= 1
					llon = (clon-1-ew[ulat,0]+symm*col)%mod
					llond = ew[ulat,0]+1
				
				ew[ulat,0] += 1
				
			
				u += ud
				ulat = clat+u
		
	target_vars[1] = new_rad
	target_vars[2] = dsum
	
def call_shrink(distance, data, ew, target, rad, rad_inc, dsum, clat, clon, symm=False, torad=False, new_rad=0):

	cdef double max_dist = np.max(distance)
	cdef double [:] target_vars = np.array([target, rad, dsum, rad_inc, max_dist])	
	r, c = distance.shape
	cdef int row = r
	cdef int col = c
	cdef int lat = clat
	cdef int lon = clon
	cdef double xnew_rad = new_rad
	
	cdef double [:,:] distance_view = distance
	cdef double [:,:] data_view = data
	cdef cnp.int32_t [:,:] ew_view = ew

	shrink(distance_view, data_view, ew_view, target_vars, row, col, lat, lon, symm, torad, xnew_rad)
	
	return target_vars[1], target_vars[2]


	
					
	
	
cdef void refine(double [:, :] distance, double [:,:] data, cnp.int32_t [:,:] ew, double [:] target_vars, int row, int col, int clat, int clon, int symm, double thresh):

	#force the best to be within one grid cell
	if target_vars[3] > target_vars[1]:
		shrink(distance, data, ew, target_vars, row, col, clat, clon, symm, 1, 0)
	else:
		shrink(distance, data, ew, target_vars, row, col, clat, clon, symm, 0, 0)	
	cdef double lrad = target_vars[1]

	expand(distance, data, ew, target_vars, row, col, clat, clon, symm , 0, 0)
	cdef double urad = target_vars[1]
		
	cdef double target = target_vars[0]
	cdef double rad = target_vars[1]
	cdef double dsum = target_vars[2]
	
	cdef double old_dsum = dsum
	cdef double old_rad = rad	
	cdef double new_rad = (urad + lrad)/2
	cdef int grow = 0

	cdef int it = 0
	cdef int max_it = 500

	while it < max_it and lrad != urad:

		if grow:
			expand(distance, data, ew, target_vars, row, col, clat, clon, symm , 1, new_rad)
			dsum = target_vars[2]
		else:	
			shrink(distance, data, ew, target_vars, row, col, clat, clon, symm , 1, new_rad)
			dsum = target_vars[2]

		if abs(urad - lrad) < thresh and dsum >= target: 		
			break;
			
		if dsum < target: 
			lrad = new_rad
			grow = 1	
		else:
			urad = new_rad
			grow = 0
		
		old_rad = new_rad
		new_rad = (lrad + urad)/2			

		old_dsum = dsum
		it += 1


	if it == max_it: print("refine maxits reached idx (", clat, clon, ") sum =", dsum, "[", lrad, urad, "]")

	

def call_refine(distance, data, ew, target, rad, rad_inc, dsum, clat, clon, symm=False, thresh=1):

	cdef double max_dist = np.max(distance)
	cdef double [:] target_vars = np.array([target, rad, dsum, rad_inc, max_dist])	
	
	r, c = distance.shape
	cdef int row = r
	cdef int col = c
	cdef int lat = clat
	cdef int lon = clon
	
	cdef double xthresh = thresh
	cdef double [:,:] distance_view = distance
	cdef double [:,:] data_view = data
	cdef cnp.int32_t [:,:] ew_view = ew

	refine( distance_view, data_view, ew_view, target_vars, row, col, lat, lon, symm, xthresh)

	return target_vars[1], target_vars[2]	





########################################


def select_data(input_data, data_bounds, target_bounds):
	'''
	Subset the input array so it is tight to the target boundary
	returns (a view of) the subset, the grid snapped boundary and a flag for whole earth longitude

			Parameters:
					input_data (np.array): data
					data_bounds (list): [lower_lat, upper_lat, lower_lon, upper_lon] 
					target_bounds (list): [lower_lat, upper_lat, lower_lon, upper_lon] 
			Returns:
					np.array, list, bool
	'''
		
	r,c = input_data.shape
	cdef int row = r
	cdef int col = c

	if not target_bounds: target_bounds = [*data_bounds]
	cdef int symm = (target_bounds[2] == -target_bounds[3]) and (target_bounds[3] == 180)
	
	pt = precompute_trig(*data_bounds, row, col, centered=False, asnp=True)
	
	llat, llon = latlon_to_id(target_bounds[0],target_bounds[2],pt)
	ulat, ulon = latlon_to_id(target_bounds[1],target_bounds[3],pt)


	bounds = [ pt['lat'][llat], pt['lat'][ulat], pt['lon'][llon],  pt['lon'][ulon] ]

	if target_bounds[0] <= bounds[0]:
		bounds[0] -= pt['lat_step']	
		llat += 1
		
	if target_bounds[1] >= bounds[1]:
		bounds[1] = min(data_bounds[1], bounds[1] + pt['lat_step'])		
		ulat = max(ulat-1, 0)
		
	if target_bounds[2] <= bounds[2]:
		bounds[2]  = max( data_bounds[2], bounds[2] - pt['lon_step'] )
		llon = max(llon-1,0)

	if target_bounds[3] >= bounds[3]:
		ulon += 1
		bounds[3] += pt['lon_step']
		
	np_data = input_data[ ulat:llat, llon:ulon ]
	return np_data, bounds, symm

def valeriepieris(input_data, data_bounds, target_frac, target_bounds=None, search_bounds=None, thresh=1):

	'''
	Find the valeriepieris circle(s)
	returns radii, population, centres, data subset used, boundary used

			Parameters:
					input_data (np.array): data
					data_bounds (list): [lower_lat, upper_lat, lower_lon, upper_lon] 
					target_frac (double or list): the proportion(s) of the population included in the circle
					target_bounds (list): [lower_lat, upper_lat, lower_lon, upper_lon] only use data in this bbox
					search_bounds (list): [lower_lat, upper_lat, lower_lon, upper_lon] only search for centres in this bbox
					thresh (double): only refine circles to this precision
			Returns:
				return 	np.array(rmin), np.array(smin), best_coords, np_data, data_bounds
	'''
	
	try:
		_ = iter(target_frac)
	except TypeError as te:
		target_frac = [target_frac]
	cdef int ntarget = len(target_frac)
	
	r,c = input_data.shape
	cdef int row = r
	cdef int col = c
	cdef int symm

	np_data, data_bounds, symm = select_data(input_data, data_bounds, target_bounds)
		
	pt = precompute_trig(*data_bounds, row, col, centered=False, asnp=True)
	ddistance = np.zeros( (row, col), dtype=np.double )
	call_grid_distances_from_id(ddistance, 0, 0, pt) 
	cdef double rad_inc = ddistance[row-1,0]/(row-1)
	cdef double max_dist = np.max(ddistance)
		
	row, col = np_data.shape
	cdef double[:] latitudes 
	cdef double[:] longitudes 
	cdef double[:] sin_latitudes 
	cdef double[:] cos_latitudes 
	cdef double[:] vdist 
	cdef double[:] dlat 
	cdef double[:] dlon 

	latitudes, longitudes, sin_latitudes, cos_latitudes, vdist, dlat, dlon, row, col, symm, lat_step, lon_step = precompute_trig(*data_bounds, row, col, asnp=False, centered=True)
	
	cdef int lat_from = 0
	cdef int lat_to = row
	cdef int lon_from = 0
	cdef int lon_to = col
	if search_bounds:
		pt['lat'] = np.array(latitudes)
		pt['lon'] = np.array(longitudes)
		
		lat_to, lon_from = latlon_to_id(search_bounds[0],search_bounds[2],pt)
		lat_from, lon_to = latlon_to_id(search_bounds[1],search_bounds[3],pt)
		
		lat_to += 1
		lon_to += 1	
	

	cdef double total = np.sum(np_data)
	
	np_distance = np.zeros( (row, col), dtype=np.double )
	np_ew = np.zeros( (ntarget,row,2), dtype=np.int_)		
	cdef double [:, :] distance = np_distance
	cdef double [:,:] data = np_data
	cdef cnp.int32_t [:,:,:] ew = np_ew

	cdef double [:,:] target_vars = np.empty( (ntarget,5) )
	cdef int i
	for i in range(ntarget):
		if target_frac[i] >= 1:
			target_vars[i,0] = total*(1-1e-8)
		else:
			target_vars[i,0] = target_frac[i] * total
			
		target_vars[i,1] = 0 #rad
		target_vars[i,2] = 0 #dsum
		target_vars[i,3] = rad_inc
		target_vars[i,4] = max_dist

	cdef double[:] rmin = np.ones( ntarget )*np.inf
	cdef double[:] smin = np.zeros( ntarget ) 
	cdef list lat_coords = [ [] for i in range(ntarget) ]
	cdef list lon_coords = [ [] for i in range(ntarget) ]
	cdef int lat_min = 0
	cdef int lon_min = 0
	cdef int shift_dir = -1
	cdef int clon = lon_from-1
	cdef int clat = lat_from	


	for clat in range(lat_from, lat_to):
		shift_dir *= -1
		clon += shift_dir

		grid_distances_from_id(clat, 0, distance, row, col, symm, latitudes, longitudes, sin_latitudes, cos_latitudes, vdist, dlat, dlon)
		start = lon_from + ((1-shift_dir)//2) * (lon_to-1-lon_from) 

		for clon_i in range(lon_from, lon_to):

			if clon == start:
				if clat == lat_from:
					for i in range(ntarget): expand(distance, data, ew[i], target_vars[i], row, col, clat, clon, symm, 0, 0)
				else:
					for i in range(ntarget): vshift(distance, data, ew[i], target_vars[i], row, col, clat, clon, symm)					
			else:
				if shift_dir == 1:
					for i in range(ntarget): rshift(distance, data, ew[i], target_vars[i], row, col, clat, clon, symm)
				else:
					for i in range(ntarget): lshift(distance, data, ew[i], target_vars[i], row, col, clat, clon, symm)
				
			
			for i in range(ntarget): 
				if target_vars[i,2] > target_vars[i,0]:

					shrink(distance, data, ew[i], target_vars[i], row, col, clat, clon, symm, 0, 0)
					expand(distance, data, ew[i], target_vars[i], row, col, clat, clon, symm, 0, 0)
					
					if target_vars[i,1] == rmin[i]: 
						lat_coords[i].append( clat )
						lon_coords[i].append( clon )
					elif target_vars[i,1] < rmin[i]: 
						rmin[i] = target_vars[i,1]
						smin[i] = target_vars[i,2]
						lat_coords[i] = [ clat ]
						lon_coords[i] = [ clon ]

			clon += shift_dir
						
	##could refine as we go but seems faster to do it here...
	ew = np.zeros( (ntarget,row,2), dtype=np.int_)
	target_vars = np.empty( (ntarget,5) )
	for i in range(ntarget):
		if target_frac[i] >= 1:
			target_vars[i,0] = total*(1-1e-8)
		else:
			target_vars[i,0] = target_frac[i] * total
			
		target_vars[i,1] = 0 #rad
		target_vars[i,2] = 0 #dsum
		target_vars[i,3] = rad_inc
		target_vars[i,4] = max_dist
	cdef list best_lat_coords = [ [] for i in range(ntarget) ]
	cdef list best_lon_coords = [ [] for i in range(ntarget) ]
	cdef double xthresh = thresh

	rmin = np.ones( ntarget )*np.inf
	smin = np.zeros( ntarget )
	
	cdef int nc,old_lat,old_lon,ti
	for ti in range(ntarget):
		
		nc = len(lat_coords[ti])
		old_lat = -1
		old_lon = -1
	
		
		for i in range(nc):
			clat, clon = lat_coords[ti][i], lon_coords[ti][i]

			if i == 0:
				grid_distances_from_id(clat, 0, distance, row, col, symm, latitudes, longitudes, sin_latitudes, cos_latitudes, vdist, dlat, dlon)			
				expand(distance, data, ew[ti], target_vars[ti], row, col, clat, clon, symm, 0, 0)
			else:
				shift_dir = 1	
				if clon < old_lon: shift_dir = -1	
				for j in range( abs(clon - old_lon) ):
					old_lon += shift_dir
					if shift_dir == 1:
						rshift(distance, data, ew[ti], target_vars[ti], row, col, old_lat, old_lon, symm)
					else:
						lshift(distance, data, ew[ti], target_vars[ti], row, col, old_lat, old_lon, symm)

				if clat != old_lat: 
					for j in range( clat - old_lat ):
						old_lat += 1
						grid_distances_from_id(old_lat, 0, distance, row, col, symm, latitudes, longitudes, sin_latitudes, cos_latitudes, vdist, dlat, dlon)			
						vshift(distance, data, ew[ti], target_vars[ti], row, col, old_lat, old_lon, symm)

			if target_vars[ti,2] > target_vars[ti,0]:
					refine(distance, data, ew[ti], target_vars[ti], row, col, clat, clon, symm, xthresh)
					
					if target_vars[ti,1] == rmin[ti]: 
						best_lat_coords[ti].append( clat )
						best_lon_coords[ti].append( clon )
					elif target_vars[ti,1] < rmin[ti]: 
						rmin[ti] = target_vars[ti,1]
						smin[ti] = target_vars[ti,2]
						best_lat_coords[ti] = [clat]
						best_lon_coords[ti] = [clon]

			old_lat = lat_coords[ti][i]
			old_lon = lon_coords[ti][i]

	for ti in range(ntarget): rmin[ti] = max(rad_inc, rmin[ti])

	##no double comprehension to avoid compiler warning! 
	cdef list best_coords = []
	cdef int blat, blon
	for ti in range(ntarget): 
		best_coords.append( [] )
		for i in range( len(best_lat_coords[ti]) ):
			blat = best_lat_coords[ti][i]
			blon = best_lon_coords[ti][i]
			best_coords[ti].append( (latitudes[blat], longitudes[blon])  ) 
			
	
	return 	np.array(rmin), np.array(smin), best_coords, np_data, data_bounds


def centre_of_population(input_data, data_bounds):
	'''
	Find the centre of population (US census definition)
			Parameters:
					input_data (np.array): data
					data_bounds (list): [lower_lat, upper_lat, lower_lon, upper_lon] 
			Returns:
				return double, double
	'''
	
	cdef int row, col
	row, col = input_data.shape

	cdef double[:] latitudes 
	cdef double[:] longitudes 
	cdef double[:] sin_latitudes 
	cdef double[:] cos_latitudes 
	cdef double[:] vdist 
	cdef double[:] dlat 
	cdef double[:] dlon 

	latitudes, longitudes, sin_latitudes, cos_latitudes, vdist, dlat, dlon, row, col, symm, lat_step, lon_step = precompute_trig(*data_bounds, row, col, asnp=False, centered=True)
		

	cdef double [:,:] data = input_data
	cdef double clat = 0
	cdef double clon = 0
	cdef double pop = 0
	cdef double cpop = 0
	cdef int i, j
	for i in range(row):
		for j in range(col):
			pop += data[i,j]
			clat += data[i,j] * latitudes[i]
			cpop += data[i,j] * cos_latitudes[i]
			clon += data[i,j] * cos_latitudes[i] * longitudes[j]

	xlat, xlon = clat/pop, clon/cpop
	pt = {}
	pt['lat'] = np.array(latitudes)
	pt['lon'] = np.array(longitudes)
		
	return snap_to_grid(xlat, xlon, pt)




def centre_of_population_3d(input_data, data_bounds):
	'''
	Find the centre of population (3d definition)
			Parameters:
					input_data (np.array): data
					data_bounds (list): [lower_lat, upper_lat, lower_lon, upper_lon] 
			Returns:
				return double, double
	'''	
	cdef int row, col, i, j
	row, col = input_data.shape

	cdef double[:] latitudes 
	cdef double[:] longitudes 
	cdef double[:] sin_latitudes 
	cdef double[:] cos_latitudes 
	cdef double[:] vdist 
	cdef double[:] dlat 
	cdef double[:] dlon 

	latitudes, longitudes, sin_latitudes, cos_latitudes, vdist, dlat, dlon, row, col, symm, lat_step, lon_step = precompute_trig(*data_bounds, row, col, asnp=False, centered=True)
	sin_longitudes_ = np.sin( np.array(longitudes)*deg2rad )
	cos_longitudes_ = np.cos( np.array(longitudes)*deg2rad )
	cdef double[:] sin_longitudes = sin_longitudes_
	cdef double[:] cos_longitudes = cos_longitudes_
	
	cdef double [:,:] data = input_data
	cdef double xb = 0
	cdef double yb = 0
	cdef double zb = 0
	cdef double pop = 0

	for i in range(row):
		for j in range(col):
			pop += data[i,j]
			xb += data[i,j] * cos_longitudes[j] * cos_latitudes[i]
			yb += data[i,j] * sin_longitudes[j] * cos_latitudes[i]
			zb += data[i,j] * sin_latitudes[i]
	
	xb /= pop
	yb /= pop
	zb /= pop
	cdef double norm = sqrt(xb*xb + yb*yb + zb*zb)	

			
	xlat, xlon = asin(zb/norm)*rad2deg, atan2(yb,xb)*rad2deg
	
	pt = {}
	pt['lat'] = np.array(latitudes)
	pt['lon'] = np.array(longitudes)

	return snap_to_grid(xlat, xlon, pt)
		
	

def median_centre_of_population(input_data, data_bounds, maxits = 100):
	'''
	Find the median centre of population (Fermat Weber point)
			Parameters:
					input_data (np.array): data
					data_bounds (list): [lower_lat, upper_lat, lower_lon, upper_lon] 
					max_its (int): should converge way before this
			Returns:
				return double, double
	'''	
	#inefficient but it's fast anyway
	cdef double xlat, xlon
	xlat, xlon = centre_of_population(input_data, data_bounds)

	cdef int row, col, i, j
	row, col = input_data.shape

	cdef double[:] latitudes 
	cdef double[:] longitudes 
	cdef double[:] sin_latitudes 
	cdef double[:] cos_latitudes 
	cdef double[:] vdist 
	cdef double[:] dlat 
	cdef double[:] dlon 

	latitudes, longitudes, sin_latitudes, cos_latitudes, vdist, dlat, dlon, row, col, symm, lat_step, lon_step = precompute_trig(*data_bounds, row, col, asnp=False, centered=True)
	pt = {}

	pt['lat'] = np.array(latitudes)
	pt['lon'] = np.array(longitudes)

	np_distance = np.zeros( (row, col), dtype=np.double )

	cdef double [:,:] data = input_data	
	cdef double [:, :] distance = np_distance
	cdef int clat, clon

	cdef double nlat, nlon, norm, f
	cdef double eps = 1e-8
	cdef int it = 0
	while it < maxits:
		it += 1
		#calculate the distance from every point to the guess
		clat, clon = latlon_to_id(xlat, xlon, pt)		
		grid_distances_from_id(clat, clon, distance, row, col, symm, latitudes, longitudes, sin_latitudes, cos_latitudes, vdist, dlat, dlon)
		
		#standard algo blows up at a grid point, see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC26449/pdf/pq001423.pdf to fix it
		norm = 0
		nlat = 0
		nlon = 0
		pop = 0
		for i in range(row):
			for j in range(col):
				if distance[i,j] == 0: continue
				f = data[i,j]/distance[i,j]
				pop += data[i,j]
				norm += f  
				nlat += f*latitudes[i]
				nlon += f*longitudes[j]
		
		nlat = (1 - (data[clat, clon]/pop)) * (nlat / norm) + (data[clat, clon]/pop) * xlat
		nlon = (1 - (data[clat, clon]/pop)) * (nlon / norm) + (data[clat, clon]/pop) * xlon
		
		if abs(nlat - xlat) < eps and abs(nlon - xlon) < eps: 
			return snap_to_grid(nlat, nlon, pt)
		
		xlat = nlat
		xlon = nlon
	
	print("maxits reached")
	return snap_to_grid(xlat, xlon, pt)
	
	

def standard_distance(input_data, data_bounds, xlat, xlon):
	'''
	Find the Bachi standard distance
			Parameters:
					input_data (np.array): data
					data_bounds (list): [lower_lat, upper_lat, lower_lon, upper_lon] 
					xlat (double): centre lat
					xlon (double): centre lon
			Returns:
				return double
	'''	
	cdef int row, col
	row, col = input_data.shape

	cdef double[:] latitudes 
	cdef double[:] longitudes 
	cdef double[:] sin_latitudes 
	cdef double[:] cos_latitudes 
	cdef double[:] vdist 
	cdef double[:] dlat 
	cdef double[:] dlon 

	latitudes, longitudes, sin_latitudes, cos_latitudes, vdist, dlat, dlon, row, col, symm, lat_step, lon_step = precompute_trig(*data_bounds, row, col, asnp=False, centered=True)

	np_distance = np.zeros( (row, col), dtype=np.double )
	cdef double [:,:] data = input_data	
	cdef double [:, :] distance = np_distance
	cdef int clat, clon
	
	pt = {}
	pt['lat'] = np.array(latitudes)
	pt['lon'] = np.array(longitudes)	
	clat, clon = latlon_to_id(xlat, xlon, pt)
	grid_distances_from_id(clat, clon, distance, row, col, symm, latitudes, longitudes, sin_latitudes, cos_latitudes, vdist, dlat, dlon)
	
	cdef double pop = 0
	cdef double sdist = 0
	cdef double avd = 0
	cdef int count = 0
	cdef int i, j
	for i in range(row):
		for j in range(col):
			pop += data[i,j]
			sdist += data[i,j] * distance[i,j] * distance[i,j]
			if data[i,j] != 0:
				avd += distance[i,j]
				count += 1
					
					
	return sqrt( sdist / pop )
