from backend.worker import frame_limit


def test_seconds_limit_respects_each_cameras_fps_and_frame_cap():
    options={'max_frames':0,'max_seconds':30}
    assert frame_limit(options,{'fps':25})==750
    assert frame_limit(options,{'fps':20})==600
    assert frame_limit(options|{'max_frames':12},{'fps':20})==12
    assert frame_limit({'max_frames':900},{})==900
    assert frame_limit(options|{'camera_max_seconds':{'4':10}},{'camera_id':4,'fps':20})==200
