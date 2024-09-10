# ML-Dynamics
Toolbox to analyze and characterize protein dynamics

> [!NOTE]  
> Note for Chris: Functions will go under the **ML-Dynamics** directory. These functions should be called in **scripts** to create the "ML Pipeline". 

> [!NOTE]  
> Note for Matthias: This is the codebase for the Kemp Eliminase ML project.

> [!IMPORTANT]  
> Step 1 is now written. Driver script is `run_all_replicas.sh` which calls `run_single_replica.sh` for each replica (actual code is located at `run_single.py`.

## Sample Docs - Reading in the Trajectories

`ML-Dynamics/io.py` provides functionality for input/output trajectory reading with MD-Analysis

```

```

## Sample Docs - Calculating Molecular Properties

`ML-Dynamics/properties.py` provides functionality for calculating molecular properties of a system (should be use in conjunction with `io` tools.

```

```

## Sample Docs - Feature Pruning 

`ML-Dynamics/features.py` provides functionality for dimensionality reduction of the included feature set

```

```

## Sample Docs - Test/Train Splits

`ML-Dynamics/splits.py` provides functionality for splitting the current dataset

```

```
