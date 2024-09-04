# ML-Dynamics
Toolbox to analyze and characterize protein dynamics

> [!NOTE]  
> Note for Chris: Functions will go under the **ML-Dynamics** directory. These functions should be called in **scripts** to create the "ML Pipeline". 

> [!NOTE]  
> Note for Matthias: This is the codebase for the Kemp Eliminase ML project.

## DATA TYPES WE WILL NEED
- Data type to store molecular property matrix (of size MxN where M is the # of samples and N is # of features)
- Data type of store feature names
- Data type to store row labels/names/metadata (i.e each row needs to have a label/column corresponding to whether the sample comes from an unevolved/evolved mutant)

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
