# basic.py

def square(number):
    """
    This function returns the square of a given number
    """
    return number ** 2

def double(number):
    """
    This function returns twice the value of a given number
    """
    return number * 2

def add(a, b):
    """
    This function returns the sum of given numbers
    """
    return a + b

# stats.py

def mean(numbers):
    """
    This function calculates the mean of a list of numbers
    """
    return sum(numbers) / len(numbers)

def median(numbers):
    """
    This function calculates the median of a list of numbers
    """
    numbers.sort()
    n = len(numbers)
    if n % 2 == 0:
        return (numbers[n // 2 - 1] + numbers[n // 2]) / 2
    return numbers[n // 2]

# __init__.py

from . import basic
from . import stats

# Now your directory structure should look like
"""
mymath
mymath/__init__.py
mymath/basic.py
mymath/statistics.py
"""
