
from typing import TypeVar

T = TypeVar('T')
D = TypeVar('D')

def default(var : T, val : D) -> T | D:
    return val if var is None else var