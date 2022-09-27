### import packages
from lzma import MODE_FAST
import math
import statistics
import numpy as np
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt

### numeric data
x = [8.0, 1, 2.5, 4, 28.0]
x_with_nan = [8.0, 1, 2.5, math.nan, 4, 28.0]

x
x_with_nan

### nan value (interchangable functions)
math.isnan(np.nan), np.isnan(math.nan)
math.isnan(x_with_nan[3]), np.isnan(x_with_nan[3])

### create an array
y, y_with_nan = np.array(x), np.array(x_with_nan)
z, z_with_nan = pd.Series(x), pd.Series(x_with_nan)

y
y_with_nan

z
z_with_nan

### measures of central tendency
## mean
mean_ = sum(x) / len(x)
mean_

mean_ = statistics.mean(x)
mean_

mean_ = statistics.fmean(x)
mean_

mean_ = statistics.mean(x_with_nan)
mean_

mean_ =statistics.fmean(x_with_nan)
mean_

mean_ = np.mean(y)
mean_

mean_ = y.mean()
mean_

np.mean(y_with_nan)
y_with_nan.mean()

np.nanmean(y_with_nan)

mean_ = z.mean()
mean_

z_with_nan.mean()

## weighted mean
0.2 * 2 + 0.5 * 4 + 0.3 * 8

x = [8.0, 1, 2.5, 4, 28.0]
w = [0.1, 0.2, 0.3, 0.25, 0.15]
wmean = sum(w[i] * x[i] for i in range(len(x))) / sum(w)
wmean

wmean = sum(x_ * w_ for (x_, w_) in zip(x, w)) / sum(w)
wmean

w = np.array([0.1, 0.2, 0.3, 0.0, 0.2, 0.1])
(w * y_with_nan).sum() / w.sum()

np.average(y_with_nan, weights=w)

np.average(z_with_nan, weights=w)

## harmonic mean
hmean = len(x) / sum(1 / item for item in x)
hmean

hmean = statistics.harmonic_mean(x)
hmean

statistics.harmonic_mean(x_with_nan)
statistics.harmonic_mean([1, 0,2])

scipy.stats.hmean(y)
scipy.stats.hmean(z)

gmean = 1
for item in x:
    gmean *= item

gmean **= 1 / len(x)
gmean

gmean = statistics.geometric_mean(x)
gmean

gmean = statistics.geometric_mean(x_with_nan)
gmean

scipy.stats.gmean(y)
scipy.stats.gmean(z)

## median
n = len(x)
if n % 2: 
    median_ = sorted(x)[round(0.5*(n-1))]
else:
    x_ord, index = sorted(x), round (0.5 * n)
    median_ = 0.5 * (x_ord[index-1] + x_ord[index])
    
median_

median_ = statistics.median(x)
median_

median_ = statistics.median(x[:-1])
median_

statistics.median_low(x[:-1])
statistics.median_high(x[:-1])

statistics.median(x_with_nan)
statistics.median_low(x_with_nan)
statistics.median_high(x_with_nan)

median_ = np.median(y)
median_

median_ = np.median(y[:-1])
median_

z.median()
z_with_nan.median()

## mode
u = [2, 3, 2, 8, 12]
mode_ = max((u.count(item), item) for item in set(u))[1]
mode_

mode_ = statistics.mode(u)
mode_

mode_ = statistics.multimode(u)
mode_

v = [12, 15, 12, 15, 21, 15, 12]
statistics.mode(v)
statistics.multimode(v)

statistics.mode([2, math.nan, 2])
statistics.multimode([2, math.nan, 2])
statistics.mode([2, math.nan, 0, math.nan, 5])

u, v = np.array(u), np.array(v)
mode_ = scipy.stats.mode(u)
mode_

mode_ = scipy.stats.mode(v)
mode_

mode_.mode
mode_.count

u, v, w = pd.Series(u), pd.Series(v), pd.Series([2, 2, math.nan])
u.mode()
v.mode()
w.mode()

### measures of variability
## variance
n = len(x)
mean_ = sum(x) / n 
var_ = sum((item - mean_)**2 for item in x) / (n-1)
var_

var_ = statistics.variance(x)
var_

statistics.variance(x_with_nan)

var_ = np.var(y, ddof=1)
var_

var_ = y.var(ddof=1)
var_

np.var(y_with_nan, ddof=1)
y_with_nan.var(ddof=1)

np.nanvar(y_with_nan, ddof=1)

z.var(ddof=1)
z_with_nan.var(ddof=1)

## standard deviation
std_ = var_ ** 0.5
std_

std_ = statistics.stdev(x)
std_

np.std(y, ddof=1)
y.std(ddof=1)
np.std(y_with_nan, ddof=1)
y_with_nan.std(ddof=1)
np.nanstd(y_with_nan, ddof=1)

z.std(ddof=1)
z_with_nan.std(ddof=1)

x = [8.0, 1, 2.5, 4, 28.0]
n = len(x)
mean_ = sum(x) / n
var_ = sum((item - mean_)**2 for item in x) / (n - 1)
std_ = var_ ** 0.5
skew_ = (sum((item - mean_)**3 for item in x)
        * n / ((n - 1) * (n - 2) * std_**3))
skew_ # positive skew means x has a right-side tail

y, y_with_nan = np.array(x), np.array(x_with_nan)
scipy.stats.skew(y, bias=False)
scipy.stats.skew(y_with_nan, bias=False)

z, z_with_nan = pd.Series(x), pd.Series(x_with_nan)
z.skew()
z_with_nan.skew()

## percentiles
x = [-5.0, -1.1, 0.1, 2.0, 8.0, 12.8, 21.0, 25.8, 41.0]
statistics.quantiles(x, n=2)
statistics.quantiles(x, n=4, method='inclusive')

y = np.array(x)
np.percentile(y,5)
np.percentile(y, 95)

np.percentile(y, [25,50,75])
np.median(y)

y_with_nan = np.insert(y, 2, np.nan)
y_with_nan
np.nanpercentile(y_with_nan, [25, 50, 75])

np.quantile(y, 0,05)
np.quantile(y, 0.95)
np.quantile(y, [0.25, 0.5, 0.75])
np.nanquantile(y_with_nan, [0.25, 0.5, 0.75])

z, z_with_nan = pd.Series(y), pd.Series(y_with_nan)
z.quantile(0.05)
z.quantile(0.95)
z.quantile([0.25, 0.5, 0.75])
z_with_nan.quantile([0.25, 0.5, 0.75])

np.ptp(y)
np.ptp(z)
np.ptp(y_with_nan)
np.ptp(z_with_nan)

np.amax(y) - np.amin(y)
np.nanmax(y_with_nan) - np.nanmin(y_with_nan)
y.max() - y.min()
z.max() - z.min()
z_with_nan.max() - z_with_nan.min()

quartiles = np.quantile(y, [0.25, 0.75])
quartiles[1] - quartiles[0]

quartiles = z.quantile([0.25, 0.75])
quartiles[0.75] - quartiles[0.25]

### summary of descriptive stats
result = scipy.stats.describe(y, ddof=1, bias=False)
result

result.nobs
result.minmax[0] # min
result.minmax[1] # max
result.mean
result.variance
result.skewness
result.kurtosis

result = z.describe()
result

result['mean']
result['std']
result['min']
result['max']
result['25%']
result['50%']
result['75%']

### measures of correlation between pairs of data
x = list(range(-10, 11))
y = [0, 2, 2, 2, 2, 3, 3, 6, 7, 4, 7, 6, 6, 9, 4, 5, 5, 10, 11, 12, 14]
x_, y_ = np.array(x), np.array(y)
x__, y__ = pd.Series(x_), pd.Series(y_)

### covariance
n = len(x)
mean_x, mean_y - sum(x) / n, sum(y)/ n 
cov_xy = (sum((x[k] - mean_x) * (y[k] - mean_y) for k in range(n))
        / (n - 1))
cov_xy

cov_matrix = np.cov(x_,y_)
cov_matrix

x_.var(ddof=1)
y_.var(ddof=1)

cov_xy = cov_matrix[0, 1]
cov_xy

cov_xy = cov_matrix([1, 0])
cov_xy

cov_xy = x__.cov(y__)
cov_xy

cov_xy = y__.cov(x__)
cov_xy

### correlation coefficient
var_x = sum((item - mean_x)**2 for item in x) / (n - 1)
var_y = sum((item - mean_y)**2 for item in y) / (n - 1)
std_x, std_y = var_x ** 0.5, var_y ** 0.5
r = cov_xy / (std_x * std_y)
r

r, p = scipy.stats.pearsonr(x_, y_)
r
p

corr_matrix = np.corrcoef(x_, y_)
corr_matrix

r = corr_matrix[0, 1]
r

r = corr_matrix[1, 0]
r

scipy.stats.linregress(x_, y_)

result = scipy.stats.linregress(x_, y_)
r = result.rvalue
r

r = x__.corr(y__)
r

r = y__.corr(x__)
r

### working with 2D data
a = np.array([[1, 1, 1],
              [2, 3, 1],
              [4, 9, 2],
              [8, 27, 4],
              [16, 1, 1]])
a

np.mean(a)
a.mean()
np.median(a)
a.var(ddof=1)

np.mean(a, axis=0)
a.mean(axis=0)

np.mean(a, axis=1)
a.mean(axis=1)

np.median(a, axis=0)
np.median(a, axis=1)
a.var(axis=0, ddof=1)
a.var(axis=1, ddof=1)

scipy.stats.gmean(a)
scipy.stats.gmean(a, axis=0)

scipy.stats.gmean(a, axis=1)

scipy.stats.gmean(a, axis=None)

scipy.stats.describe(a, axis=None, ddof=1, bias=False)
scipy.stats.describe(a, ddof=1, bias=False)
scipy.stats.describe(a, axis=1, ddof=1, bias=False)

result = scipy.stats.describe(a, axis=1, ddof=1, bias=False)
result.mean

### dataframes 
row_names = ['first', 'second', 'third', 'fourth', 'fifth']
col_names = ['A', 'B', 'C']
df = pd.DataFrame(a, index=row_names, columns=col_names)
df

df.mean()
df.var()

df.mean(axis=1)
df.var(axis=1)

df['A']

df['A'].mean()
df['A'].var()

df.values
df.to_numpy()

df.describe()

df.describe().at['mean', 'A']
df.describe().at['50%', 'B']

### visualizing data
plt.style.use('ggplot')

### box plots
np.random.seed(seed=0)
x = np.random.randn(1000)
y = np.random.randn(100)
z = np.random.randn(10)

fig, ax = plt.subplots()
ax.boxplot((x, y, z), vert=False, showmeans=True, meanline=True,
           labels=('x', 'y', 'z'), patch_artist=True,
           medianprops={'linewidth': 2, 'color': 'purple'},
           meanprops={'linewidth': 2, 'color': 'red'})


### histograms 
hist, bin_edges = np.histogram(x, bins=10)
hist
bin_edges

fig, ax = plt.subplots()
ax.hist(x, bin_edges, cumulative=False)
ax.set_xlabel('x')
ax.set_ylabel('Frequency')
plt.show()

fig, ax = plt.subplots()
ax.hist(x, bin_edges, cumulative=True)
ax.set_xlabel('x')
ax.set_ylabel('Frequency')
plt.show()

### pie charts
x, y, z = 128, 256, 1024

fig, ax = plt.subplots()
ax.pie((x, y, z), labels=('x', 'y', 'z'), autopct='%1.1f%%')
plt.show()

### bar charts
x = np.arange(21)
y = np.random.randint(21, size=21)
err = np.random.randn(21)

fig, ax = plt.subplots())
ax.bar(x, y, yerr=err)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

### x-y plots 
x = np.arange(21)
y = 5 + 2 * x + 2 * np.random.randn(21)
slope, intercept, r, *__ = scipy.stats.linregress(x, y)
line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'

fig, ax = plt.subplots()
ax.plot(x, y, linewidth=0, marker='s', label='Data points')
ax.plot(x, intercept + slope * x, label=line)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(facecolor='white')
plt.show()

### heatmaps
matrix = np.cov(x, y).round(decimals=2)
fig, ax = plt.subplots()
ax.imshow(matrix)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('x', 'y'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('x', 'y'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, matrix[i, j], ha='center', va='center', color='w')
plt.show()

matrix = np.corrcoef(x, y).round(decimals=2)
fig, ax = plt.subplots()
ax.imshow(matrix)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('x', 'y'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('x', 'y'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, matrix[i, j], ha='center', va='center', color='w')
plt.show()

