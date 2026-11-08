import os

# OpenMP (torch, OpenCV) starts one thread per core by default. On the login node (96 cores) thread creation fails
# and libgomp exits the process with code 1 and no traceback. Set before torch/cv2 are imported by the tests.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "4")
