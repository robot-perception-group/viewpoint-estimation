# Script to generate synthetic training data for the viewpoint estimation model using the HSMAL model
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from scipy.sparse import csc_matrix

from chumpy_compat import ChumpyBypasser, force_to_tensor

class HSMALMaster:
    def __init__(self, pkl_path):
        print(f"Loading raw data from {pkl_path}...")
        with open(pkl_path, 'rb') as f:
            raw_data = ChumpyBypasser(f, encoding='latin1').load()

        self.v_template = force_to_tensor(raw_data['v_template'])
        self.shapedirs = force_to_tensor(raw_data['shapedirs'])

        jr = raw_data['J_regressor']
        if isinstance(jr, csc_matrix): jr = jr.toarray()
        self.J_regressor = force_to_tensor(jr)

        if len(self.shapedirs.shape) == 1:
             v_count = self.v_template.shape[0]
             self.shapedirs = self.shapedirs.view(v_count, 3, -1)

        self.idx_tail, self.idx_neck = 0, 15
        print(f"Ready. Template: {self.v_template.shape}, Shapes: {self.shapedirs.shape}")


    def get_coco_confidence(self, j3d_cam, coco_indices):
        """Return constant confidence=2 for all COCO joints."""
        _ = j3d_cam
        return np.full(len(coco_indices), 2.0, dtype=np.float32)

    def generate_dataset(self, n=10000):


        print(f"Generating {n} samples (Full 35 + COCO 17)...")


        #  Sample random shape parameters (betas) and camera angles
        betas = torch.randn(n, self.shapedirs.shape[2]) * 0.05
        yaws = np.random.uniform(0, 360, n)
        pitches = np.random.uniform(0, 45, n)

        #  Build shaped mesh and regress skeleton joints in canonical model space
        shape_offset = torch.einsum('vcl,nl->nvc', self.shapedirs, betas)
        v_shaped = self.v_template.unsqueeze(0) + shape_offset
        joints_3d = torch.matmul(self.J_regressor, v_shaped)

        coco_indices = [34, 35, 9, 4, 11, 6, 13, 8, 15, 1, 23, 18, 25, 20, 27, 22, 33]

        final_2d, final_3d = [], []
        full_2d, full_3d = [], []

        for i in range(n):
            j3d_raw = joints_3d[i].numpy()

            #  Canonical alignment:
            #    rotate so tail->neck vector points to the reference direction.
            spine_vec = j3d_raw[self.idx_neck] - j3d_raw[self.idx_tail]
            raw_angle = np.degrees(np.arctan2(spine_vec[0], spine_vec[1]))
            correction = R.from_euler('z', -raw_angle + 180, degrees=True)
            j3d_calibrated = j3d_raw @ correction.as_matrix().T

            #  Apply sampled camera/view rotations.
            #    First pitch around x, then yaw around z.
            combined_rot = (
                R.from_euler('x', pitches[i], degrees=True)
                * R.from_euler('z', -yaws[i], degrees=True)
            )
            j3d_cam = j3d_calibrated @ combined_rot.as_matrix().T

            #  Keep full joints for debug.
            f3d = j3d_cam
            f2d = j3d_cam[:, [0, 2]]
            full_3d.append(f3d)
            full_2d.append(f2d)

            # Select COCO subset used by downstream viewpoint model
            k3d = j3d_cam[coco_indices]
            final_3d.append(k3d)

            #  Project to 2D and normalize per-sample into a unit box
            #    This removes absolute scale/translation and keeps only geometry
            conf = self.get_coco_confidence(j3d_cam, coco_indices)
            j2d_raw = j3d_cam[coco_indices][:, [0, 2]]
            min_xy = j2d_raw.min(axis=0)
            max_xy = j2d_raw.max(axis=0)
            j2d_norm = (j2d_raw - min_xy) / (np.max(max_xy - min_xy) + 1e-6)

            # Store normalized (x, y, confidence) tuples
            sample_2d = np.hstack([j2d_norm, conf.reshape(-1, 1)])
            final_2d.append(sample_2d)

        return {
            "kpts": np.array(final_2d, dtype=np.float32),
            "kpts_3d": np.array(final_3d, dtype=np.float32),
            "kpts_full": np.array(full_2d, dtype=np.float32),
            "kpts_3d_full": np.array(full_3d, dtype=np.float32),
            "yaw": np.array(yaws, dtype=np.float32),
        }
if __name__ == "__main__":
    PKL_FILE = ".../my_smpl_0000_horse_new_skeleton_horse.pkl" # available at: https://sites.google.com/view/cv4horses
    try:
        master = HSMALMaster(PKL_FILE)
        
        generated = master.generate_dataset(250000)

        np.savez(
            ".../zebra_training_dataV250k.npz",
            kpts=generated["kpts"],
            kpts_3d=generated["kpts_3d"],
            kpts_full=generated["kpts_full"],
            kpts_3d_full=generated["kpts_3d_full"],
            yaw=generated["yaw"],
        )
        print("Success! Dataset saved with keys: 'kpts', 'kpts_3d', 'kpts_full', 'kpts_3d_full', 'yaw'.")

       
    except Exception as e:
        print(f"Script failed: {e}")
        import traceback
        traceback.print_exc()