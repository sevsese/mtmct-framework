import pytest
from backend.floor_map import fit,project


def pairs():
    return [{'video':[x,y],'plan':[.1+.6*x,.2+.5*y]} for x,y in [(0,0),(1,0),(1,1),(0,1)]]


def test_floor_projection_uses_feet_and_preserves_ids():
    calibration=fit(pairs())
    row={'camera_id':1,'frame_id':25,'local_id':3,'global_id':7,'timestamp':1.,'x1':20,'x2':40,'y1':0,'y2':60}
    result=project([row],{1:{'width':100,'height':100}},{'1':calibration})
    assert len(result)==1
    assert result[0]['x']==pytest.approx(.28,abs=1e-6)
    assert result[0]['y']==pytest.approx(.5,abs=1e-6)
    assert result[0]['global_id']==7
    assert project([row],{1:{'width':100,'height':100}},{})==[]
    assert project([row|{'y2':110}],{1:{'width':100,'height':100}},{'1':calibration})==[]


def test_floor_rejects_ambiguous_and_nonfinite_points():
    with pytest.raises(ValueError):fit(pairs()[:3])
    with pytest.raises(ValueError):fit([{'video':[.1*i,.1*i],'plan':[.2*i,.2*i]} for i in range(4)])
    p=pairs();p[0]['plan']=[float('nan'),0]
    with pytest.raises(ValueError):fit(p)
