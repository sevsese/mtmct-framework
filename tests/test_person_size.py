import numpy as np
import pytest

from mtmct.corridor_zoom import filter_person_size


def test_original_size_gate_rejects_tiny_people_and_keeps_boundary():
    boxes = np.array([[0, 0, 30, 100, .9, 0], [0, 0, 40, 80, .9, 0],
                      [0, 0, 40, 100, .9, 0]])
    assert filter_person_size(boxes, (1000, 2000, 3), (.02, .1)).tolist() == boxes[2:].tolist()
    assert filter_person_size(boxes[:0], (1000, 2000, 3), (.02, .1)).shape == (0, 6)
    with pytest.raises(ValueError):
        filter_person_size(boxes, (1000, 2000, 3), (-.1, .1))


def test_head_and_torso_do_not_spawn_tracks_beside_full_person():
    from mtmct.corridor_zoom import distinct_crop_boxes, suppress_body_parts
    full = np.array([[1385,442,1636,1146,.85,0]])
    crop = np.array([[1405,442,1632,659,.95,0], [1700,400,1800,800,.9,0]])
    np.testing.assert_array_equal(distinct_crop_boxes(full,crop),crop[1:])
    boxes = np.concatenate((full,crop))
    np.testing.assert_array_equal(suppress_body_parts(boxes),boxes[[0,2]])


def test_group_box_is_rejected_without_removing_individuals():
    from mtmct.corridor_zoom import reject_group_boxes
    previous=[[0,0,40,100],[50,0,90,100]]
    boxes=np.array([[0,0,90,100,.9,0],[0,0,40,100,.8,0],[50,0,90,100,.8,0]])
    np.testing.assert_array_equal(reject_group_boxes(boxes,previous),boxes[1:])
