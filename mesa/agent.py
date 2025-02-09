"""Agent related classes.

Core Objects: Agent and AgentSet.
"""

# Mypy; for the `|` operator purpose
# Remove this __future__ import once the oldest supported Python is 3.10
from __future__ import annotations

import contextlib
import copy
import functools
import itertools
import operator
import warnings
import weakref
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Iterator, MutableSet, Sequence
from random import Random

# mypy
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np

if TYPE_CHECKING:
    # We ensure that these are not imported during runtime to prevent cyclic
    # dependency.
    from mesa.model import Model
    from mesa.space import Position

from mesa.collectionbase import CollectionBase


class Agent:
    """Base class for a model agent in Mesa.

    Attributes:
        model (Model): A reference to the model instance.
        unique_id (int): A unique identifier for this agent.
        pos (Position): A reference to the position where this agent is located.

    Notes:
          unique_id is unique relative to a model instance and starts from 1

    """

    # this is a class level attribute
    # it is a dictionary, indexed by model instance
    # so, unique_id is unique relative to a model, and counting starts from 1
    _ids = defaultdict(functools.partial(itertools.count, 1))

    def __init__(self, model: Model, *args, **kwargs) -> None:
        """Create a new agent.

        Args:
            model (Model): The model instance in which the agent exists.
            args: passed on to super
            kwargs: passed on to super

        Notes:
            to make proper use of python's super, in each class remove the arguments and
            keyword arguments you need and pass on the rest to super

        """
        super().__init__(*args, **kwargs)

        self.model: Model = model
        self.unique_id: int = next(self._ids[model])
        self.pos: Position | None = None
        self.model.register_agent(self)

    def remove(self) -> None:
        """Remove and delete the agent from the model.

        Notes:
            If you need to do additional cleanup when removing an agent by for example removing
            it from a space, consider extending this method in your own agent class.

        """
        with contextlib.suppress(KeyError):
            self.model.deregister_agent(self)

    def step(self) -> None:
        """A single step of the agent."""

    def advance(self) -> None:  # noqa: D102
        pass

    @classmethod
    def create_agents(cls, model: Model, n: int, *args, **kwargs) -> AgentSet[Agent]:
        """Create N agents.

        Args:
            model: the model to which the agents belong
            args: arguments to pass onto agent instances
                  each arg is either a single object or a sequence of length n
            n: the number of agents to create
            kwargs: keyword arguments to pass onto agent instances
                   each keyword arg is either a single object or a sequence of length n

        Returns:
            AgentSet containing the agents created.

        """

        class ListLike:
            """Helper class to make default arguments act as if they are in a list of length N."""

            def __init__(self, value):
                self.value = value

            def __getitem__(self, i):
                return self.value

        listlike_args = []
        for arg in args:
            if isinstance(arg, (list | np.ndarray | tuple)) and len(arg) == n:
                listlike_args.append(arg)
            else:
                listlike_args.append(ListLike(arg))

        listlike_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, (list | np.ndarray | tuple)) and len(v) == n:
                listlike_kwargs[k] = v
            else:
                listlike_kwargs[k] = ListLike(v)

        agents = []
        for i in range(n):
            instance_args = [arg[i] for arg in listlike_args]
            instance_kwargs = {k: v[i] for k, v in listlike_kwargs.items()}
            agent = cls(model, *instance_args, **instance_kwargs)
            agents.append(agent)
        return AgentSet(agents, random=model.random)

    @property
    def random(self) -> Random:
        """Return a seeded stdlib rng."""
        return self.model.random

    @property
    def rng(self) -> np.random.Generator:
        """Return a seeded np.random rng."""
        return self.model.rng


class AgentSet(CollectionBase[Agent]):
    """A collection of agents within an agent-based model (ABM).

    This class extends CollectionBase to provide specialized functionality for managing
    agents, including weak references to allow proper garbage collection.

    See CollectionBase for the full set of collection operations available.
    """
    
    def __init__(self, items: Iterable[Agent], random: Random | None = None) -> None:
        """Create a new AgentSet.

        Args:
            items (Iterable[Agent]): The agents to add to the set.
            random (Random, optional): The random number generator to use for shuffling. Defaults to None.

        """
        super().__init__(items,random=random)

    def select(
        self,
        filter_func: Callable[[Agent], bool] | None = None,
        at_most: int | float = float("inf"),
        inplace: bool = False,
        agent_type: type[Agent] | None = None,
    ) -> AgentSet:
        """Select a subset of agents from the AgentSet.
        
        See CollectionBase.select() for full documentation.
        """
        return super().select(
            filter_func=filter_func,
            at_most=at_most,
            inplace=inplace,
            item_type=agent_type
        )

    def shuffle_do(self, method: str | Callable, *args, **kwargs) -> AgentSet:
        """Shuffle the agents in the AgentSet and then invoke a method or function on each agent.

        This is an optimized version of calling shuffle() followed by do().

        Args:
            method (str | Callable): The method name or function to call on each agent
            *args: Arguments to pass to the method
            **kwargs: Keyword arguments to pass to the method

        Returns:
            AgentSet: The AgentSet instance itself for method chaining
        """
        weakrefs = list(self._items.keyrefs())
        self.random.shuffle(weakrefs)

        if isinstance(method, str):
            for ref in weakrefs:
                if (agent := ref()) is not None:
                    getattr(agent, method)(*args, **kwargs)
        else:
            for ref in weakrefs:
                if (agent := ref()) is not None:
                    method(agent, *args, **kwargs)

        return self

    def add(self, agent: Agent) -> None:
        """Add an agent to the set."""
        self._items[agent] = None

    def discard(self, agent: Agent) -> None:
        """Remove an agent from the set if present."""
        with contextlib.suppress(KeyError):
            del self._items[agent]

    def remove(self, agent: Agent) -> None:
        """Remove an agent from the set."""
        del self._items[agent]

    def __getstate__(self):
        """Return state for pickling.

        Convert WeakKeyDictionary to a regular list of agents for serialization.
        """
        return {"agents": list(self._items.keys()), "random": self.random}

    def __setstate__(self, state):
        """Set state when unpickling.

        Restore WeakKeyDictionary from the list of agents.
        """
        self.random = state["random"]
        self._update(state["agents"])

    @overload
    def get(
        self,
        attr_names: str,
        handle_missing: Literal["error", "default"] = "error",
        default_value: Any = None,
    ) -> list[Any]: ...

    @overload
    def get(
        self,
        attr_names: list[str],
        handle_missing: Literal["error", "default"] = "error",
        default_value: Any = None,
    ) -> list[list[Any]]: ...

    def get(
        self,
        attr_names: str | list[str],
        handle_missing: str = "error",
        default_value: Any = None,
    ) -> list[Any] | list[list[Any]]:
        """See CollectionBase.get() for full documentation."""
        return super().get(attr_names, handle_missing, default_value)

    def groupby(self, by: Callable | str, result_type: str = "agentset") -> GroupBy:
        """Group agents by the specified attribute or return from the callable.

        Args:
            by (Callable | str): Used to determine what to group agents by
                * if callable, it will be called for each agent and the return is used for grouping
                * if str, it should refer to an attribute on the agent and the value of this 
                  attribute will be used for grouping
            result_type (str, optional): The datatype for the resulting groups {"agentset", "list"}

        Returns:
            GroupBy: A GroupBy object containing the grouped agents

        Notes:
            There might be performance benefits to using `result_type='list'` if you don't need 
            the advanced functionality of an AgentSet.
        """
        groups = defaultdict(list)

        if isinstance(by, Callable):
            for agent in self:
                groups[by(agent)].append(agent)
        else:
            for agent in self:
                groups[getattr(agent, by)].append(agent)

        if result_type == "agentset":
            return GroupBy({k: AgentSet(v, random=self.random) for k, v in groups.items()})
        else:
            return GroupBy(groups)


class GroupBy:
    """Helper class for AgentSet.groupby.

    Attributes:
        groups (dict): A dictionary with the group_name as key and group as values

    """

    def __init__(self, groups: dict[Any, list | AgentSet]):
        """Initialize a GroupBy instance.

        Args:
            groups (dict): A dictionary with the group_name as key and group as values

        """
        self.groups: dict[Any, list | AgentSet] = groups

    def map(self, method: Callable | str, *args, **kwargs) -> dict[Any, Any]:
        """Apply the specified callable to each group and return the results.

        Args:
            method (Callable, str): The callable to apply to each group,

                                    * if ``method`` is a callable, it will be called it will be called with the group as first argument
                                    * if ``method`` is a str, it should refer to a method on the group

                                    Additional arguments and keyword arguments will be passed on to the callable.
            args: arguments to pass to the callable
            kwargs: keyword arguments to pass to the callable

        Returns:
            dict with group_name as key and the return of the method as value

        Notes:
            this method is useful for methods or functions that do return something. It
            will break method chaining. For that, use ``do`` instead.

        """
        if isinstance(method, str):
            return {
                k: getattr(v, method)(*args, **kwargs) for k, v in self.groups.items()
            }
        else:
            return {k: method(v, *args, **kwargs) for k, v in self.groups.items()}

    def do(self, method: str | Callable, *args, **kwargs) -> GroupBy:
        """Apply the specified callable to each group.

        Args:
            method (Callable, str): The callable to apply to each group,

                                    * if ``method`` is a callable, it will be called it will be called with the group as first argument
                                    * if ``method`` is a str, it should refer to a method on the group

                                    Additional arguments and keyword arguments will be passed on to the callable.
            args: arguments to pass to the callable
            kwargs: keyword arguments to pass to the callable

        Returns:
            the original GroupBy instance

        Notes:
            this method is useful for methods or functions that don't return anything and/or
            if you want to chain multiple do calls

        """
        if isinstance(method, str):
            for v in self.groups.values():
                getattr(v, method)(*args, **kwargs)
        else:
            for v in self.groups.values():
                method(v, *args, **kwargs)

        return self

    def count(self) -> dict[Any, int]:
        """Return the count of agents in each group.

        Returns:
            dict: A dictionary mapping group names to the number of agents in each group.
        """
        return {k: len(v) for k, v in self.groups.items()}

    def agg(self, attr_name: str, func: Callable) -> dict[Hashable, Any]:
        """Aggregate the values of a specific attribute across each group using the provided function.

        Args:
            attr_name (str): The name of the attribute to aggregate.
            func (Callable): The function to apply (e.g., sum, min, max, mean).

        Returns:
            dict[Hashable, Any]: A dictionary mapping group names to the result of applying the aggregation function.
        """
        return {
            group_name: func([getattr(agent, attr_name) for agent in group])
            for group_name, group in self.groups.items()
        }

    def __iter__(self):  # noqa: D105
        return iter(self.groups.items())

    def __len__(self):  # noqa: D105
        return len(self.groups)
