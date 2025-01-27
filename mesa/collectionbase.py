"""Base class for collections in Mesa.

Core Objects: CollectionBase
"""

from __future__ import annotations

import copy
import functools
import operator
import warnings
from collections.abc import Callable, Hashable, Iterable, Iterator, MutableSet, Sequence
from random import Random
from typing import Any, TypeVar, Generic

T = TypeVar('T')

class CollectionBase(MutableSet, Sequence, Generic[T]):
    """A base collection class that provides set-like and sequence functionality.
    
    This class serves as a base for specialized collections like AgentSet and LayerCollection,
    providing common functionality for managing, filtering, and manipulating collections of objects.

    Attributes:
        random (Random): The random number generator for the collection.

    Notes:
        A `UserWarning` is issued if `random=None`. You can resolve this warning by explicitly
        passing a random number generator.
    """

    def __init__(self, items: Iterable[T], random: Random | None = None):
        """Initialize the collection.

        Args:
            items (Iterable[T]): An iterable of items to be included in the collection.
            random (Random | None): The random number generator. If None, a new unseeded one is created.
        """
        if random is None:
            warnings.warn(
                "Random number generator not specified, this can make models non-reproducible. "
                "Please pass a random number generator explicitly",
                UserWarning,
                stacklevel=2,
            )
            random = Random()
        self.random = random
        self._initialize_storage(items)

    def _initialize_storage(self, items: Iterable[T]) -> None:
        """Initialize the storage mechanism for the collection.
        
        This method should be implemented by subclasses to define how items are stored.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """Return the number of items in the collection."""
        raise NotImplementedError

    def __iter__(self) -> Iterator[T]:
        """Provide an iterator over the items in the collection."""
        raise NotImplementedError

    def __contains__(self, item: T) -> bool:
        """Check if an item is in the collection."""
        raise NotImplementedError

    def select(
        self,
        filter_func: Callable[[T], bool] | None = None,
        at_most: int | float = float("inf"),
        inplace: bool = False,
        item_type: type[T] | None = None,
    ) -> CollectionBase[T]:
        """Select a subset of items based on a filter function and/or quantity limit.

        Args:
            filter_func: A function that takes an item and returns True if it should be included.
            at_most: Maximum number of items to select. If float < 1, represents a fraction.
            inplace: If True, modifies the current collection; otherwise, returns a new one.
            item_type: The type of items to select.

        Returns:
            A new collection containing the selected items, unless inplace is True.
        """
        inf = float("inf")
        if filter_func is None and item_type is None and at_most == inf:
            return self if inplace else copy.copy(self)

        if at_most <= 1.0 and isinstance(at_most, float):
            at_most = int(len(self) * at_most)

        def item_generator(filter_func, item_type, at_most):
            count = 0
            for item in self:
                if count >= at_most:
                    break
                if (not filter_func or filter_func(item)) and (
                    not item_type or isinstance(item, item_type)
                ):
                    yield item
                    count += 1

        items = item_generator(filter_func, item_type, at_most)
        return self._update(items) if inplace else self.__class__(items, self.random)

    def shuffle(self, inplace: bool = False) -> CollectionBase[T]:
        """Randomly shuffle the order of items.

        Args:
            inplace: If True, shuffles in place; otherwise, returns a new collection.

        Returns:
            The shuffled collection.
        """
        raise NotImplementedError

    def sort(
        self,
        key: Callable[[T], Any] | str,
        ascending: bool = False,
        inplace: bool = False,
    ) -> CollectionBase[T]:
        """Sort the items based on a key function or attribute name.

        Args:
            key: Function or attribute name to sort by.
            ascending: If True, sort in ascending order.
            inplace: If True, sorts in place; otherwise, returns a new collection.

        Returns:
            The sorted collection.
        """
        if isinstance(key, str):
            key = operator.attrgetter(key)

        sorted_items = sorted(self, key=key, reverse=not ascending)
        return self._update(sorted_items) if inplace else self.__class__(sorted_items, self.random)

    def _update(self, items: Iterable[T]) -> CollectionBase[T]:
        """Update the collection with new items.

        Args:
            items: The new items to update the collection with.

        Returns:
            The updated collection.
        """
        raise NotImplementedError

    def do(self, method: str | Callable, *args, **kwargs) -> CollectionBase[T]:
        """Apply a method or function to each item.

        Args:
            method: Name of method to call or function to apply.
            *args: Arguments to pass to the method.
            **kwargs: Keyword arguments to pass to the method.

        Returns:
            The collection itself for chaining.
        """
        if isinstance(method, str):
            for item in self:
                getattr(item, method)(*args, **kwargs)
        else:
            for item in self:
                method(item, *args, **kwargs)
        return self

    def map(self, method: str | Callable, *args, **kwargs) -> list[Any]:
        """Apply a method or function to each item and collect results.

        Args:
            method: Name of method to call or function to apply.
            *args: Arguments to pass to the method.
            **kwargs: Keyword arguments to pass to the method.

        Returns:
            List of results from applying the method to each item.
        """
        if isinstance(method, str):
            return [getattr(item, method)(*args, **kwargs) for item in self]
        else:
            return [method(item, *args, **kwargs) for item in self]

    def agg(self, attribute: str, func: Callable) -> Any:
        """Aggregate an attribute across all items.

        Args:
            attribute: Name of the attribute to aggregate.
            func: Function to apply to the attribute values.

        Returns:
            Result of applying the function to the attribute values.
        """
        values = self.get(attribute)
        return func(values)

    def get(
        self,
        attr_names: str | list[str],
        handle_missing: str = "error",
        default_value: Any = None,
    ) -> list[Any] | list[list[Any]]:
        """Get attributes from all items.

        Args:
            attr_names: Name(s) of attributes to get.
            handle_missing: How to handle missing attributes ('error' or 'default').
            default_value: Value to use for missing attributes if handle_missing='default'.

        Returns:
            List of attribute values or list of lists if multiple attributes requested.
        """
        is_single_attr = isinstance(attr_names, str)

        if handle_missing == "error":
            if is_single_attr:
                return [getattr(item, attr_names) for item in self]
            else:
                return [[getattr(item, attr) for attr in attr_names] for item in self]
        elif handle_missing == "default":
            if is_single_attr:
                return [getattr(item, attr_names, default_value) for item in self]
            else:
                return [[getattr(item, attr, default_value) for attr in attr_names] 
                        for item in self]
        else:
            raise ValueError(
                f"Unknown handle_missing option: {handle_missing}, "
                "should be one of 'error' or 'default'"
            )

    def __getitem__(self, index: int | slice) -> T | list[T]:
        """Get item(s) at the specified index or slice."""
        raise NotImplementedError 