import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm
from sklearn.linear_model import RANSACRegressor
from sklearn.neighbors import LocalOutlierFactor

# 1D outlier finding

def drop_inf(x, y):
  "Removes (x, y) pairs where either the x or the y value is NaN or infinite."
  df = pd.DataFrame({'X': x, 'Y': y})
  # Get rid of infinite and NaN
  df.replace([np.inf, -np.inf], np.nan, inplace=True)
  df = df.dropna()
  def to_np(colname):
    return df[colname].to_numpy().reshape(-1, 1)
  return to_np('X'), to_np('Y')

def find_outliers(x, return_bool=True):
  """Determines whether each element of x is an outlier.
  Assumes that most x are normally distributed with a few outlier samples mixed in.
  x: a 1D Numpy array
  return_bool: whether the results should be booleans instead of +1/-1.
  Returns a Numpy array of boolean values indicating whether the corresponding elements of x are outliers.
  If return_bool is False, then returns +1 (inlier) or -1 (outlier) for each sample like the Scipy convention.
  """
  x = x.flatten()
  # Find the number of elements of x that's <= each element
  sample_cdf = rankdata(x, method="max") / x.shape # CDF of observations
  dist_quantiles = norm.ppf(sample_cdf) # theoretical quantiles
  # Plot actual quantiles (y-axis) vs. theoretical quantiles (x-axis)
  # In other words, a Q-Q plot
  qqx, qqy = drop_inf(dist_quantiles, x)
  rr = RANSACRegressor().fit(qqx, qqy)
  # Deviations of the regression
  resid = qqy - rr.predict(qqx)
  # Find outliers
  lof = LocalOutlierFactor().fit_predict(resid)
  if return_bool:
    return (lof == -1)
  else:
    return lof

"""
# Example
rng = np.random.default_rng(12345)
# In this dataset,
# The first 100 values are normally distributed around mean -1.
# The last 5 are normally distributed around mean 3 which should be flagged as outliers.
d = np.concatenate([rng.standard_normal(100) - 1, rng.standard_normal(5) + 3])
find_outliers(d)
"""
