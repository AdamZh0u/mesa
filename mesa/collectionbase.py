"""Base class for collections in Mesa.

Core Objects: CollectionBase
"""

from __future__ import annotations

import copy
import functools
import operator
import warnings
import weakref
from collections.abc import Callable, Hashable, Iterable, Iterator, MutableSet, Sequence
from random import Random
from typing import Any, Generic, TypeVar, Literal, overload

T = TypeVar("T")


class CollectionBase(MutableSet, Sequence, Generic[T]):
    """A base collection class that provides set-like and sequence functionality.

    This class serves as a base for specialized collections like AgentSet and CellCollection,
    providing common functionality for managing, filtering, and manipulating collections of objects.
    It uses weak references to allow proper garbage collection of items.

    Features:
        - Set operations (add, remove, discard)
        - Sequence operations (indexing, iteration)
        - Filtering and selection
        - Attribute access and modification
        - Method invocation on items
        - Aggregation operations
        - Random shuffling and sorting

    The class uses weak references to store items, which means items can be garbage collected
    when they are no longer referenced elsewhere in the program. This is particularly useful
    for collections where items may be removed during simulation.

    Example:
        >>> class MyCollection(CollectionBase[MyItem]):
        ...     def __init__(self, items, random=None):
        ...         super().__init__(items, random)
        ...
        >>> collection = MyCollection([item1, item2], random=model.random)
        >>> filtered = collection.select(lambda x: x.value > 10)
        >>> collection.do("update")  # Calls update() on each item

    Attributes:
        random (Random): The random number generator for the collection.
        _items (WeakKeyDictionary): Internal storage using weak references to items.

    Notes:
        - A `UserWarning` is issued if `random=None`. You can resolve this warning by explicitly
          passing a random number generator.
        - Subclasses should implement _initialize_storage() if they need custom storage behavior.
        - Methods that modify the collection (shuffle, sort, etc.) have an inplace parameter
          to control whether they modify the existing collection or return a new one.
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
        self._items = weakref.WeakKeyDictionary({item: None for item in items})

    def __len__(self) -> int:
        """Return the number of items in the collection."""
        return len(self._items)

    def __iter__(self) -> Iterator[T]:
        """Provide an iterator over the items in the collection."""
        return self._items.keys()

    def __contains__(self, item: T) -> bool:
        """Check if an item is in the collection."""
        return item in self._items

    def __getitem__(self, index: int | slice) -> T | list[T]:
        """Get item(s) at the specified index or slice."""
        return list(self._items.keys())[index]

    def select(
        self,
        filter_func: Callable[[T], bool] | None = None,
        at_most: int | float = float("inf"),
        inplace: bool = False,
        item_type: type[T] | None = None,
    ) -> CollectionBase[T]:
        """Select a subset of items from the collection based on a filter function and/or quantity limit.

        Args:
            filter_func (Callable[[T], bool], optional): A function that takes an item and returns True if 
                the item should be included in the result. Defaults to None, meaning no filtering is applied.
            at_most (int | float, optional): The maximum amount of items to select. Defaults to infinity.
              - If an integer, at most the first number of matching items are selected.
              - If a float between 0 and 1, at most that fraction of original items are selected.
            inplace (bool, optional): If True, modifies the current collection; otherwise, returns a
                new collection. Defaults to False.
            item_type (type[T], optional): The class type of the items to select. Defaults to
                None, meaning no type filtering is applied.

        Returns:
            CollectionBase: A new collection containing the selected items, unless inplace is True,
                in which case the current collection is updated.

        Notes:
            - at_most just returns the first n or fraction of items. To take a random sample,
              shuffle() beforehand.
            - at_most is an upper limit. When specifying other criteria, the number of items
              returned can be smaller.
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
        # Get the actual items from weakrefs
        items = []
        for ref in self._items.keyrefs():
            if (item := ref()) is not None:
                items.append(item)
            
        self.random.shuffle(items)

        if inplace:
            self._items = weakref.WeakKeyDictionary({item: None for item in items})
            return self
        else:
            return self.__class__(items, self.random)

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
        return (
            self._update(sorted_items)
            if inplace
            else self.__class__(sorted_items, self.random)
        )

    def _update(self, items: Iterable[T]) -> CollectionBase[T]:
        """Update the collection with new items.

        Args:
            items: The new items to update the collection with.

        Returns:
            The updated collection.
        """
        # Convert any weakrefs to actual items
        actual_items = []
        for item in items:
            if isinstance(item, weakref.ReferenceType):
                if (actual_item := item()) is not None:
                    actual_items.append(actual_item)
            else:
                actual_items.append(item)
            
        self._items = weakref.WeakKeyDictionary({item: None for item in actual_items})
        return self

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
            for itemref in self._items.keyrefs():
                if (item := itemref()) is not None:
                    getattr(item, method)(*args, **kwargs)
        else:
            for itemref in self._items.keyrefs():
                if (item := itemref()) is not None:
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
            return [
                getattr(item, method)(*args, **kwargs)
                for itemref in self._items.keyrefs()
                if (item := itemref()) is not None
            ]
        else:
            return [
                method(item, *args, **kwargs)
                for itemref in self._items.keyrefs()
                if (item := itemref()) is not None
            ]

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

    @overload
    def get(
        self,
        attr_names: str,  # when passing a single attribute name
        handle_missing: Literal["error", "default"] = "error",
        default_value: Any = None,
    ) -> list[Any]: ...  # returns a list

    @overload
    def get(
        self,
        attr_names: list[str],  # when passing a list of attribute names
        handle_missing: Literal["error", "default"] = "error",
        default_value: Any = None,
    ) -> list[list[Any]]: ...  # returns a list of lists


    def get(
        self,
        attr_names,
        handle_missing: str = "error",
        default_value: Any = None,
    ):
        """Retrieve the specified attribute(s) from each item in the collection.

        Args:
            attr_names (str | list[str]): The name(s) of the attribute(s) to retrieve from each item.
            handle_missing (str, optional): How to handle missing attributes. Can be:
                                            - 'error' (default): raises an AttributeError if attribute is missing.
                                            - 'default': returns the specified default_value.
            default_value (Any, optional): The default value to return if 'handle_missing' is set to 'default'
                                           and the item does not have the attribute.

        Returns:
            list[Any]: A list with the attribute value for each item if attr_names is a str.
            list[list[Any]]: A list with a lists of attribute values for each item if attr_names is a list of str.

        Raises:
            AttributeError: If 'handle_missing' is 'error' and the item does not have the specified attribute(s).
            ValueError: If an unknown 'handle_missing' option is provided.
        """
        is_single_attr = isinstance(attr_names, str)

        if handle_missing == "error":
            if is_single_attr:
                return [getattr(item, attr_names) for item in self]
            else:
                return [
                    [getattr(item, attr) for attr in attr_names] for item in self._items
                ]
        elif handle_missing == "default":
            if is_single_attr:
                return [
                    getattr(item, attr_names, default_value) for item in self._items
                ]
            else:
                return [
                    [getattr(item, attr, default_value) for attr in attr_names]
                    for item in self._items
                ]
        else:
            raise ValueError(
                f"Unknown handle_missing option: {handle_missing}, "
                "should be one of 'error' or 'default'"
            )

    def set(self, attr_name: str, value: Any) -> CollectionBase[T]:
        """Set a specified attribute to a given value for all items in the collection.

        Args:
            attr_name (str): The name of the attribute to set.
            value (Any): The value to set the attribute to.

        Returns:
            CollectionBase: The collection instance itself, after setting the attribute.
        """
        for item in self._items:
            setattr(item, attr_name, value)
        return self
