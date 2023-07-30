# valeriepieris
Find valeriepieris circles.

Expects 2d-numpy arrays from e.g. [SEDAC](https://sedac.ciesin.columbia.edu/data/set/gpw-v4-population-count-rev11/data-download)

```
import numpy as np
input_data = np.loadtxt("gpw_v4_population_count_rev11_2020_1_deg.asc", skiprows=6 )
input_data[ input_data < 0] = 0
```
Then call
```
from valeriepieris import valeriepieris
data_bounds = [ -90,90, -180,180 ]
target_fracs = [0.25, 0.5, 1]
rmin, smin, best_latlon, data, new_bounds  = valeriepieris(input_data,  data_bounds, target_fracs)		
```
