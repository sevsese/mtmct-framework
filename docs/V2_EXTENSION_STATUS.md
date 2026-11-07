# Extension status

The office two-minute run `office_v2_120s_nozoom_gap60_v1` uses no zoom,
the existing office guards, indexed appearance candidates, a 60-second
episode gap, and a confirmed-anchor cross-recording prototype bank.
It still uses agglomeration and heuristic costs. Its inference was started
before completing all development and it is not an all-features benchmark.

Implemented and unit tested:

- cKDTree appearance shortlist plus same-camera interval neighbors;
- separate `global_track_episodes.csv/json` with a strict greater-than-60s gap;
- model-scoped SQLite prototype bank, quality/support-weighted updates,
  repeated-sample protection and conservative cross-recording matching;
- supervised cost calibration with recording-disjoint validation;
- small-graph correlation MILP requiring supplied calibration;
- TrackEval HOTA/DetA/AssA adapter for an explicitly custom multicamera 2D protocol.

Experimental limitations:

- Cross-recording memory thresholds are not validated on office identities.
- Calibration fitted on Warehouse_000 and checked on Warehouse_002 is not
  a validated office probability. These scenes were used in earlier research.
- MILP is optional, limited to 40 nodes, and not enabled in the office run.
- HOTA adapter is not the official NVIDIA 3D benchmark protocol.
- Candidate retrieval can miss valid cross-camera pairs; compare recall against
  exhaustive enumeration before claiming a quality improvement.
- Same-camera dense overlaps can still produce quadratic candidates.
- Floor calibration uncertainty, automatic clock calibration, unified automatic
  correction and complete frontend use of episode/persistent IDs remain unfinished.
- Background evidence remains an optional separate experiment.

Validation before handoff: 17 targeted tests passed. NVIDIA calibration artifacts
and evaluation reports are stored locally under runs, not committed as results.
