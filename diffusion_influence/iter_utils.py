import itertools
from itertools import islice
from typing import Callable, Generic, Iterable, Iterator, TypeVar

T = TypeVar("T")
S = TypeVar("S")


def batched(iterable: Iterable[T], n: int) -> Iterable[tuple[T, ...]]:
    """
    Equivalent to https://docs.python.org/3/library/itertools.html#itertools.batched
    but `itertools.batched()` only becomes available in python 3.12.
    """
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch


class SizedIterator(Iterator[T]):
    def __init__(self, iterator: Iterator[T], size: int):
        self.iterator = iterator
        self.size = size

    def __next__(self) -> T:
        return next(self.iterator)

    def __len__(self) -> int:
        return self.size


class SizedIterable(Iterable, Generic[T]):
    """
    A handy wrapper for an iterable that has (or should have) a known size.
    Useful when passing an iterable to a function that expects a `Sized` element,
    or works better with a `Sized` element (e.g. tqdm progress bar).
    """

    def __init__(self, iterable: Iterable[T], size: int):
        self.iterable = iterable
        self.size = size

    def __iter__(self):
        # Return a sized iterator as well.
        return SizedIterator(iter(self.iterable), self.size)

    def __len__(self) -> int:
        return self.size


class ReiterableMap(Iterable, Generic[T, S]):
    def __init__(self, func: Callable[[T], S], iterable: Iterable[T]):
        self.func = func
        self.iterable = iterable

    def __iter__(self):
        yield from map(self.func, self.iterable)


class ChainedIterable(Iterable, Generic[T]):
    def __init__(self, *iterables: Iterable[T]):
        self.iterables = iterables

    def __iter__(self) -> Iterator[T]:
        yield from itertools.chain(*self.iterables)


def reiterable_map(func: Callable[[T], S], iterable: Iterable[T]) -> Iterable[S]:
    return ReiterableMap(func, iterable)


def func_on_enum(f: Callable[[T], S]) -> Callable[[tuple[int, T]], tuple[int, S]]:
    def new_f(ix):
        i = ix[0]
        x = ix[1]
        return i, f(x)

    return new_f
