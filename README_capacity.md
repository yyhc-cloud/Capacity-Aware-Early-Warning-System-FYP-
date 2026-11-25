# Routing capacity

Capacity is set **per module presentation (cohort)** as **whole numbers of places**. Each module
is assumed to be supported by one lecturer. For now every lecturer has the same capacity; later the
values will come from each lecturer's input.

All capacity settings live in one file: `capacity.py`. The routing code (`routing_engine.py`)
takes a capacity dictionary from there and contains no capacity numbers of its own.

## What K, R and B mean

| Symbol | Field | Meaning | Default |
|---|---|---|---|
| K | `contact_per_checkpoint` | Maximum **new contacts** per checkpoint in the module | 10 |
| R | `review_per_checkpoint` | Maximum **reviews** per checkpoint in the module | 5 |
| B | `total_contact_budget` | Maximum **unique contacts** across Weeks 2, 4 and 6 in the module | 25 |

At each checkpoint and module:

- contact places available = min(K at that week, B − contacts already made in that module)
- review places = R

Unused places are not carried over, and are not moved between Contact and Review.

## Rules checked when a capacity is created

- Every value must be a whole number ≥ 0. Decimals and proportions (for example `0.05` or
  `10.5`) are refused with a `CapacityError`.
- B must be at least K at Week 2. Otherwise the first checkpoint's contact places could never
  be filled.
- If B ≥ the sum of K over the three checkpoints, you get a **warning**, not an error,
  because the total budget would then never limit selection.

## K can differ by week

K can be one number (the same at every checkpoint), or three numbers for Weeks 2, 4 and 6:

```python
ModuleCapacity(contact_per_checkpoint=(15, 10, 5), review_per_checkpoint=5, total_contact_budget=25)
```

## How to change capacity

**For all modules:** edit `DEFAULT_CAPACITY` in `capacity.py`, or pass a different default:

```python
from capacity import ModuleCapacity, capacity_for_all_modules
cap = capacity_for_all_modules(modules, default=ModuleCapacity(12, 6, 30))
```

**For one module** (for example, a lecturer with less time):

```python
cap = capacity_for_all_modules(modules, overrides={"BBB_2013B": ModuleCapacity(5, 5, 25)})
```

An override for a module that isn't in `modules` raises an error.

**To use it in routing:**

```python
import routing_engine as RE
log, caps_log = RE.run_routing(pred, "proposed", cap)
RE.check_routing(log, caps_log, pred, "proposed", cap)
```

To show the capacity in use, call `print_capacity_table(cap)`.

## Verification

Every routing run in `capacity_aware_ews_model.ipynb` is checked with
`routing_engine.check_routing`, which confirms that no module goes over its K, R or B limits.
