"""
capacity.py - the one place where routing capacity is set.

Capacity is given per module presentation (cohort) as WHOLE NUMBERS of places. Each module is
assumed to be supported by one lecturer; for now every lecturer has the same capacity.

    K  contact_per_checkpoint  maximum new contacts at each checkpoint (one number, or a tuple
                               of three numbers for Weeks 2, 4 and 6)
    R  review_per_checkpoint   maximum reviews at each checkpoint
    B  total_contact_budget    maximum unique contacts across Weeks 2, 4 and 6

To change capacity for every module at once, edit DEFAULT_CAPACITY below.

If one module needs a different setting (for example, a larger cohort with an extra tutor),
give it its own setting through `overrides` when calling capacity_for_all_modules(). Any
module named there uses its own numbers, and every other module keeps the default. Module IDs
are module presentations such as "BBB_2013B"; an override for an ID not in `modules` raises
CapacityError. For example:

    capacity_for_all_modules(modules, overrides={"BBB_2013B": ModuleCapacity(15, 8, 35)})

Here BBB_2013B can take 15 contacts and 8 reviews at each checkpoint, with 35 contacts in total.
Every other module stays on the default.
"""

import numbers
import warnings
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

CHECKPOINT_WEEKS = (2, 4, 6)          # Days 14, 28, 42
IntOrTriple = Union[int, Tuple[int, int, int]]


class CapacityError(ValueError):
    """Raised when a capacity setting is not a valid whole-number setting."""


def _check_whole(name, v):
    # bool is a subclass of int, so exclude it explicitly; floats (e.g. 10.0 or 0.05) are refused
    if isinstance(v, bool) or not isinstance(v, numbers.Integral):
        raise CapacityError(f"{name} must be a whole number of places (got {v!r} of type "
                            f"{type(v).__name__}); decimals and proportions are not accepted")
    if v < 0:
        raise CapacityError(f"{name} must be >= 0 (got {v})")
    return int(v)


@dataclass(frozen=True)
class ModuleCapacity:
    contact_per_checkpoint: IntOrTriple = 10    # K
    review_per_checkpoint: int = 5              # R
    total_contact_budget: int = 25              # B

    def __post_init__(self):
        k = self.contact_per_checkpoint
        if isinstance(k, (tuple, list)):
            if len(k) != len(CHECKPOINT_WEEKS):
                raise CapacityError(f"K as a tuple needs {len(CHECKPOINT_WEEKS)} values "
                                    f"(Weeks 2, 4, 6); got {len(k)}")
            k = tuple(_check_whole(f"K at Week {w}", x) for w, x in zip(CHECKPOINT_WEEKS, k))
        else:
            k = _check_whole("K", k)
        object.__setattr__(self, "contact_per_checkpoint", k)
        object.__setattr__(self, "review_per_checkpoint",
                           _check_whole("R", self.review_per_checkpoint))
        b = _check_whole("B", self.total_contact_budget)
        object.__setattr__(self, "total_contact_budget", b)
        ks = self.contact_by_week()
        if b < ks[2]:
            raise CapacityError(f"B ({b}) is smaller than K at Week 2 ({ks[2]}): the first "
                                "checkpoint's contact places could never be filled")
        if b >= sum(ks.values()):
            warnings.warn(f"B ({b}) >= sum of K over Weeks 2, 4, 6 ({sum(ks.values())}): the "
                          "total budget will never limit selection", UserWarning, stacklevel=3)

    def contact_by_week(self) -> Dict[int, int]:
        k = self.contact_per_checkpoint
        if isinstance(k, tuple):
            return dict(zip(CHECKPOINT_WEEKS, k))
        return {w: k for w in CHECKPOINT_WEEKS}

    def contact_at(self, week: int) -> int:
        return self.contact_by_week()[week]

    @property
    def K(self):
        return self.contact_per_checkpoint

    @property
    def R(self):
        return self.review_per_checkpoint

    @property
    def B(self):
        return self.total_contact_budget


# Same capacity for every lecturer / module (change here for all modules)
DEFAULT_CAPACITY = ModuleCapacity(contact_per_checkpoint=10, review_per_checkpoint=5,
                                  total_contact_budget=25)


def capacity_for_all_modules(modules: Sequence[str], default: ModuleCapacity = DEFAULT_CAPACITY,
                             overrides: Optional[Mapping[str, ModuleCapacity]] = None
                             ) -> Dict[str, ModuleCapacity]:
    """{module_id: ModuleCapacity}: every module gets `default` unless it appears in `overrides`."""
    modules = sorted(set(modules))
    overrides = dict(overrides or {})
    unknown = set(overrides) - set(modules)
    if unknown:
        raise CapacityError(f"overrides given for unknown modules: {sorted(unknown)}")
    for m, c in overrides.items():
        if not isinstance(c, ModuleCapacity):
            raise CapacityError(f"override for {m} must be a ModuleCapacity")
    if not isinstance(default, ModuleCapacity):
        raise CapacityError("default must be a ModuleCapacity")
    return {m: overrides.get(m, default) for m in modules}


def capacity_table(capacity: Mapping[str, ModuleCapacity]) -> str:
    lines = [f"{'module':<12} {'K Wk2':>6} {'K Wk4':>6} {'K Wk6':>6} {'R':>4} {'B':>4}  source"]
    for m, c in sorted(capacity.items()):
        k = c.contact_by_week()
        src = "default" if c == DEFAULT_CAPACITY else "override"
        lines.append(f"{m:<12} {k[2]:>6} {k[4]:>6} {k[6]:>6} {c.R:>4} {c.B:>4}  {src}")
    return "\n".join(lines)


def print_capacity_table(capacity: Mapping[str, ModuleCapacity]) -> None:
    print(capacity_table(capacity))
