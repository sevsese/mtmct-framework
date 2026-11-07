import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from backend import main


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main,'STORE',tmp_path)
    with TestClient(main.app) as c:
        yield c


def upload(client, content, **extra):
    return client.post('/api/datasets/upload',data={'name':'TEST — native result','mode':'results'},
        files=[('files',('global_tracks_2d.csv',content,'text/csv')),*extra.get('files',[])])


CSV='camera_id,frame_id,local_id,x1,y1,x2,y2,confidence,global_id\n3,0,1,1,2,11,22,0.9,7\n15,0,2,4,5,14,25,0.8,7\n'


def test_native_ids_pagination_missing_metadata(client):
    response=upload(client,CSV)
    assert response.status_code==200,response.text
    identifier=response.json()['id']; base=f'/api/datasets/{identifier}'
    assert client.get(base).json()['summary']['global_tracks']==1
    tracks=client.get(base+'/tracks').json()
    assert tracks['total']==1 and tracks['items'][0]['cameras']==[3,15]
    assert tracks['items'][0]['start_time'] is None
    assert client.get(base+'/tracks/7').json()['observations_count']==2
    assert len(client.get(base+'/observations?limit=1').json()['items'])==1
    assert client.get(base+'/observations?camera_id=15').json()['total']==1
    assert client.get(base+'/performance').json()['metrics'] is None
    assert client.get(base+'/video/3').status_code==404
    assert client.post(base+'/process',json={}).status_code==409
    assert client.get(base+'/floorplan').json()['available'] is False


def test_bad_csv_and_inconsistent_mapping_rejected(client):
    assert upload(client,'bad,header\n1,2').status_code==422
    assert upload(client,CSV.replace('11,22','0,1')).status_code==422
    mapping=json.dumps([{'camera_id':3,'local_id':1,'global_id':99}])
    assert upload(client,CSV,files=[('files',('global_ids.json',mapping,'application/json'))]).status_code==422
    assert upload(client,CSV,files=[('files',('manifest.json','[bad','application/json'))]).status_code==422


def test_upload_and_path_boundaries(client):
    r=client.post('/api/datasets/upload',data={'mode':'raw','cameras':json.dumps([
        {'camera_id':1,'filename':'a.mp4'},{'camera_id':1,'filename':'b.mp4'}])},files=[('files',('a.mp4',b'bad'))])
    assert r.status_code==422
    assert client.post('/api/datasets/import',json={'relative_path':'../'}).status_code==422
    assert client.post('/api/datasets/upload',files=[('files',('../manifest.json',b'{}'))]).status_code==422
    assert client.post('/api/datasets/import',json={'relative_path':'runs'},headers={'origin':'https://example.org'}).status_code==403


def test_real_result_adapter(client):
    folder=main.ROOT/'runs/nvidia_branch_1920'
    if not folder.is_dir(): pytest.skip('Local NVIDIA artifacts unavailable')
    r=client.post('/api/datasets/import',json={'relative_path':'runs/nvidia_branch_1920'})
    assert r.status_code==200,r.text
    base=f"/api/datasets/{r.json()['id']}"
    assert client.get(base).json()['summary']['observations']==10249
    cameras=client.get(base+'/cameras').json()
    assert len(cameras)==2 and all(c['fps']==30 for c in cameras)
    response=client.get(base+'/video/3',headers={'Range':'bytes=0-31'})
    assert response.status_code==206 and len(response.content)==32
    obs=client.get(base+'/observations?camera_id=3&frame_id=0').json()['items']
    assert obs[0]['timestamp']==0 and obs[0]['bbox']==[1534.74,654.19,1638.93,1023.2]
    import cv2
    import numpy as np
    png=cv2.imencode('.png',np.full((100,100,3),240,dtype=np.uint8))[1].tobytes()
    assert client.post(base+'/floorplan',files={'file':('plan.png',png,'image/png')}).status_code==200
    pairs=[{'video':[x,y],'plan':[x,y]} for x,y in [(0,0),(1,0),(1,1),(0,1)]]
    assert client.put(base+'/floorplan/calibration/3',json=pairs).status_code==200
    route=client.get(base+'/floorplan/trajectories?start=0&end=1').json()
    assert route['kind']=='manual_estimate' and route['points']
    assert all(p['camera_id']==3 for p in route['points'])
    assert client.get(base+'/frame/3?time=0').headers['content-type']=='image/jpeg'
    assert client.get(base+'/floorplan/trajectories?end=121').status_code==422
    assert client.post(base+'/floorplan',files={'file':('new.png',png,'image/png')}).status_code==200
    assert client.get(base+'/floorplan').json()['calibrated'] is False


def test_optional_floorplan_and_json_shape_validation(client):
    import cv2
    import numpy as np
    r=upload(client,CSV);base=f"/api/datasets/{r.json()['id']}"
    png=cv2.imencode('.png',np.full((16,20,3),240,dtype=np.uint8))[1].tobytes()
    assert client.post(base+'/floorplan',files={'file':('plan.png',png,'image/png')}).status_code==200
    assert client.get(base+'/floorplan').json()['available'] is True
    assert client.put(base+'/floorplan',json=[{'camera_id':3,'x':.5,'y':.4}]).status_code==200
    assert client.put(base+'/floorplan',json=[{'camera_id':999,'x':.5,'y':.4}]).status_code==422
    assert client.get(base+'/floorplan/image').status_code==200
    for payload in ['{}','[1]','[{"camera_id":3']:
        assert upload(client,CSV,files=[('files',('local_tracklets.json',payload,'application/json'))]).status_code==422
    for filename,payload in [('manifest.json','{"inputs":[1]}'),('manifest.json','{"tracking_config":{"camera_sources":[]}}'),('performance.json','[1]')]:
        assert upload(client,CSV,files=[('files',(filename,payload,'application/json'))]).status_code==422


def test_mapping_without_observations_keeps_missing_counts_unknown(client):
    mapping=[{'camera_id':3,'local_id':1,'global_id':7},{'camera_id':15,'local_id':2,'global_id':7}]
    response=client.post('/api/datasets/upload',files=[('files',('global_ids.json',json.dumps(mapping),'application/json'))])
    assert response.status_code==200,response.text
    base=f"/api/datasets/{response.json()['id']}"
    summary=client.get(base).json()['summary']
    assert summary['global_tracks']==1 and summary['local_tracklets']==2
    assert summary['observations'] is None and summary['cameras']==2
    assert client.get(base+'/tracks/7').json()['observations_count'] is None
