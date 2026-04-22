#----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2025).
#----------------------------------------------------------------------------------------------------

import numpy as np
import cv2

def build_matrix_of_indices(height, width):
    return np.indices((height, width), dtype=np.float32).transpose(1, 2, 0)

def mask_to_tight_box(mask):
    a = np.transpose(np.nonzero(mask))
    if a.size == 0:
        return [0, 0, 0, 0]
    bbox = np.min(a[:, 1]), np.min(a[:, 0]), np.max(a[:, 1]), np.max(a[:, 0])
    return bbox
    
def imread_indexed(filename):
    # Use IMREAD_UNCHANGED to preserve uint16 label PNGs (e.g. OCID).
    # IMREAD_GRAYSCALE silently divides uint16 by 256 → labels become 0.
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        # Multi-channel label PNG (e.g. tabletop uses RGB-colored labels).
        # Convert to grayscale to get unique per-object values, matching
        # the original IMREAD_GRAYSCALE behavior.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.int32)