def projection(x,y,nDim=3):
    projectionTot = 0
    for dim in range(nDim):
        projectionTot = projectionTot + x[dim]*y[dim]
    return projectionTot
