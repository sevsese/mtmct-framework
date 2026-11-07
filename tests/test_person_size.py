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
