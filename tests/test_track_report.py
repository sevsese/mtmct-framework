from mtmct.track_report import build_global_track_report


def test_report_skips_suppressed_tracklets():
    rows = [dict(camera_id=0, local_id=i, start_frame=0, end_frame=25) for i in (1, 2)]
    report = build_global_track_report(rows, {(0, 1): 7}, 25.)
    assert len(report) == 1
    assert report[0]['global_id'] == 7
    assert [s['local_id'] for s in report[0]['segments']] == [1]
