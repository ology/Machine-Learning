from typing import List
from collections import Counter
from ch02_linear_algebra import sum_of_squares, dot
import math

x_list = [5, 1, 10, 2, 9, 5, 2]
y_list = [1, 1, 1, 2, 9, 5, 1]

def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)

#print(mean(x_list))

def _median_odd(xs: List[float]) -> float:
    """If len(xs) is odd, the median is the middle element"""
    return sorted(xs)[len(xs) // 2]

def _median_even(xs: List[float]) -> float:
    """If len(xs) is even, it's the average of the middle two elements"""
    sorted_xs = sorted(xs)
    hi_midpoint = len(xs) // 2  # e.g. length 4 => hi_midpoint 2
    return (sorted_xs[hi_midpoint - 1] + sorted_xs[hi_midpoint]) / 2

def median(v: List[float]) -> float:
    """Finds the 'middle-most' value of v"""
    return _median_even(v) if len(v) % 2 == 0 else _median_odd(v)

#print(median([1, 10, 2, 9, 5]))
#print(median([1, 9, 2, 10]))

def quantile(xs: List[float], p: float) -> float:
    """Returns the pth-percentile value in x"""
    p_index = int(p * len(xs))
    return sorted(xs)[p_index]

#print(quantile(x_list, 0.10))
#print(quantile(x_list, 0.25))
#print(quantile(x_list, 0.50))
#print(quantile(x_list, 0.75))
#print(quantile(x_list, 0.90))

def mode(x: List[float]) -> List[float]:
    """Returns a list, since there might be more than one mode"""
    counts = Counter(x)
    #print(counts);
    max_count = max(counts.values())
    return [
        x_i for x_i, count in counts.items()
        if count == max_count
    ]

#print(set(mode(x_list)))

def data_range(xs: List[float]) -> float:
    return max(xs) - min(xs)

#print(data_range(x_list))

def _de_mean(xs: List[float]) -> List[float]:
    """Translate xs by subtracting its mean (so the result has mean 0)"""
    x_bar = mean(xs)
    return [x - x_bar for x in xs]

def variance(xs: List[float]) -> float:
    """Almost the average squared deviation from the mean"""
    assert len(xs) >= 2, "variance requires at least two elements"
    n = len(xs)
    deviations = _de_mean(xs)
    return sum_of_squares(deviations) / (n - 1)

#print(variance(x_list))

def standard_deviation(xs: List[float]) -> float:
    """The standard deviation is the square root of the variance"""
    return math.sqrt(variance(xs))

#print(standard_deviation(x_list))

def interquartile_range(xs: List[float]) -> float:
    """Returns the difference between the 75%-ile and the 25%-ile"""
    return quantile(xs, 0.75) - quantile(xs, 0.25)

#print(interquartile_range(x_list))

def covariance(xs: List[float], ys: List[float]) -> float:
    assert len(xs) == len(ys), "xs and ys must have same number of elements"
    return dot(_de_mean(xs), _de_mean(ys)) / (len(xs) - 1)

#print(covariance(x_list, y_list))

def correlation(xs: List[float], ys: List[float]) -> float:
    """Measures how much xs and ys vary in tandem about their means"""
    stdev_x = standard_deviation(xs)
    stdev_y = standard_deviation(ys)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(xs, ys) / stdev_x / stdev_y
    else:
        return 0    # if no variation, correlation is zero

#print(correlation(x_list, y_list))
