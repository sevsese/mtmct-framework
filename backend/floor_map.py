"""Manual image-to-plan projection, independent of identity association.

Coordinates are normalized plan pixels, never metres or measured 3D positions.
"""
import cv2
import numpy as np


def fit(points):
    source=np.asarray([p['video'] for p in points],dtype=float)
    target=np.asarray([p['plan'] for p in points],dtype=float)
    for values in (source,target):
        if values.ndim!=2 or values.shape[1]!=2 or not 4<=len(values)<=20 or not np.isfinite(values).all():
            raise ValueError('Нужны 4–20 пар конечных координат')
        if np.any(values<0) or np.any(values>1):
            raise ValueError('Координаты должны быть в диапазоне 0–1')
        if cv2.contourArea(cv2.convexHull(values.astype('float32')))<.0001:
            raise ValueError('Точки не должны лежать на одной прямой')
    matrix,_=cv2.findHomography(source,target,method=0)
    if matrix is None or not np.isfinite(matrix).all() or np.linalg.cond(matrix)>1e8:
        raise ValueError('Не удалось построить устойчивую привязку')
    mapped=cv2.perspectiveTransform(source[None].astype('float64'),matrix)[0]
    error=float(np.sqrt(np.mean(np.sum((mapped-target)**2,axis=1))))
    if error>.03:
        raise ValueError('Точки противоречат друг другу: ошибка привязки больше 3% размера плана')
    return {'matrix':matrix.tolist(),'points':points,'fit_error':error,'kind':'manual_estimate'}


def project(observations,cameras,calibrations):
    projected=[]
    for row in observations:
        camera=cameras.get(row['camera_id']);cal=calibrations.get(str(row['camera_id']))
        if not camera or not cal or not camera.get('width') or not camera.get('height'):
            continue
        u=(row['x1']+row['x2'])/2/camera['width'];v=row['y2']/camera['height']
        hull=cv2.convexHull(np.asarray([p['video'] for p in cal['points']],dtype='float32'))
        # No extrapolation outside the manually calibrated floor polygon.
        if cv2.pointPolygonTest(hull,(float(u),float(v)),False)<0: continue
        value=np.asarray(cal['matrix'])@np.array([u,v,1.])
        if abs(value[2])<1e-8: continue
        x,y=value[:2]/value[2]
        if not np.isfinite([x,y]).all() or not (0<=x<=1 and 0<=y<=1): continue
        projected.append({k:row[k] for k in ('camera_id','frame_id','local_id','global_id','timestamp')}|{'x':float(x),'y':float(y)})
    return projected
