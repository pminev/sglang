"""
Usage:
python3 local_example_dia.py
"""

import sglang as sgl

def some_random_decorator(func):
    def decorator():
        print("Decorator applied")
        func()
        print("Decorator finished")
    return decorator

@some_random_decorator
def some_function():
    print("Function executed")

some_function()

