"""Decorator-based registry for dataset and method adapters."""

from typing import Callable, Dict

DATASETS: Dict[str, Callable] = {}
METHODS: Dict[str, Callable] = {}


def register_dataset(name: str):
    def deco(cls):
        if name in DATASETS:
            raise ValueError(f"dataset {name} already registered")
        DATASETS[name] = cls
        cls._registry_name = name
        return cls
    return deco


def register_method(name: str):
    def deco(cls):
        if name in METHODS:
            raise ValueError(f"method {name} already registered")
        METHODS[name] = cls
        cls._registry_name = name
        return cls
    return deco


def get_dataset(name: str):
    if name not in DATASETS:
        raise KeyError(
            f"unknown dataset {name!r}; registered: {sorted(DATASETS)}")
    return DATASETS[name]


def get_method(name: str):
    if name not in METHODS:
        raise KeyError(
            f"unknown method {name!r}; registered: {sorted(METHODS)}")
    return METHODS[name]
