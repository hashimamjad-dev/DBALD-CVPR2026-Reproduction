`src/lane_repro/evaluation.py` loads TuSimple's original evaluator functions, omitting only its Python 2 command-line block. The 200-ms runtime gate remains enabled. The separate ungated score is clearly labeled and must not replace the official score. The additional one-to-one geometry diagnostic uses a different protocol and is not official ACC or ASR.

CULane's official C++ evaluation lives in `third_party/SCNN/tools/lane_evaluation` when checked out on a supported filesystem. It computes lane IoU/F1, not the paper's CULane ACC. A full official CULane evaluator has not yet been run here.
