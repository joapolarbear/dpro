# Memory Estimation

Usage:
```python
from memory import MemoryEstimator

memory_estimator = MemoryEstimator("TENSORFLOW")
estimated_memory_usage = memory_estimator.estimate(dag, param_dict)
```

default unit: MB


## TODO

- [ ] workload specific. need to know the model. 

e.g. select forward nodes
```py
def _is_forward(name):
    if name.startswith(DEL.join([RANK0_PREFIX, FORWARD_CAT]) + "bert"):
        return True
    return False
```

