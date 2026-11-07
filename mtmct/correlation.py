"""Constrained correlation clustering MILP for small calibrated graphs."""
from itertools import combinations
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


def solve(count, probabilities, conflicts=(), max_nodes=40, seconds=30):
    if count > max_nodes or count < 0:
        raise ValueError('Exact optimizer restricted to small graphs')
    if count < 2:
        return [[i] for i in range(count)], {'optimal':True}
    pairs=list(combinations(range(count),2))
    index={pair:i for i,pair in enumerate(pairs)}
    objective=np.zeros(len(pairs))
    for pair,p in probabilities.items():
        if pair not in index or not np.isfinite(p) or not 0 < p < 1:
            raise ValueError('Expected canonical pairs and probabilities in (0,1)')
        objective[index[pair]]=np.log1p(-p)-np.log(p)
    upper=np.ones(len(pairs))
    for pair in conflicts:
        upper[index[tuple(sorted(pair))]]=0
    triples=list(combinations(range(count),3))
    matrix=lil_matrix((3*len(triples),len(pairs)))
    for t,(a,b,c) in enumerate(triples):
        ids=[index[a,b],index[a,c],index[b,c]]
        for r in range(3):
            for k,j in enumerate(ids):
                matrix[3*t+r,j]=-1 if k==r else 1
    result=milp(objective,integrality=np.ones(len(pairs)),bounds=Bounds(0,upper),
                constraints=LinearConstraint(matrix.tocsr(),-np.inf,1),options={'time_limit':seconds})
    if not result.success:
        raise RuntimeError(f'Optimizer did not prove optimality: {result.message}')
    parent=list(range(count))
    def root(i):
        while parent[i]!=i:
            i=parent[i]
        return i
    for (a,b),value in zip(pairs,result.x):
        if value>.5:
            parent[root(b)]=root(a)
    groups={}
    for i in range(count):
        groups.setdefault(root(i),[]).append(i)
    return list(groups.values()),dict(optimal=True,objective=float(result.fun))
