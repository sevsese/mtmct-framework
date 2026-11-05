"""Distinguish normal video EOF from an incomplete decode."""
import cv2


def read_frame(cap, frame_id: int):
    ok, frame = cap.read()
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not ok and count > 0 and frame_id < count:
        raise RuntimeError(f"Video decoding stopped at frame {frame_id} of {count}")
    return ok, frame
