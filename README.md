# OutlierFinder
Find 1D outliers assuming a normal distribution

# Install
1. Clone the repository `git clone https://github.com/agentlans/OutlierFinder`
3. Install `pip install .`

# Use

```python
import numpy as np
from OutlierFinder import find_outliers

# Generate example data
rng = np.random.default_rng(12345)
# In this dataset,
# The first 100 values are normally distributed around mean -1.
# The last 5 values are normally distributed around mean 3 which should be flagged as outliers.
d = np.concatenate([rng.standard_normal(100) - 1, rng.standard_normal(5) + 3])

# Try to find the outliers in this dataset
find_outliers(d)
```

# Author, Licence

Copyright ©️ 2023 Alan Tseng

GNU General Public License v3
