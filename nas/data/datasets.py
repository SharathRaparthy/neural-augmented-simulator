import os
import numpy as np
import h5py
from torch.utils.data import Dataset

from nas.data import RECORDINGS_PATH


class RealRecordingsV1(Dataset):
    def __init__(self, dtype, path = None):
        super().__init__()
        assert dtype in ["01", "02", "10"]
        self.type = dtype
        print(f'Variant : {self.type}')
        
        if path is None:
            path = os.path.join(RECORDINGS_PATH, "real-recordings-01-02-10-v1-comp.hdf5")

        f = h5py.File(path, 'r')
        self.real_posvel = f.get(f"/{dtype}/real_posvel")
        self.actions = f.get(f"/{dtype}/actions")
        self.next_real_posvel = f.get(f"/{dtype}/next_real_posvel")
        self.next_sim_posvel = f.get(f"/{dtype}/next_sim_posvel")

    def __len__(self):
        return len(self.real_posvel)

    def format_data(self, idx):
        return (
            np.hstack((self.real_posvel[idx], self.actions[idx], self.next_sim_posvel[idx])),
            self.next_real_posvel[idx] - self.next_sim_posvel[idx]
        )

    def __getitem__(self, idx):
        x, y = self.format_data(idx)
        return {"x": x, "y": y}


if __name__ == '__main__':
    ds = RealRecordingsV1("02")
    print (len(ds))

    out_x = np.zeros((1120,998,30),dtype=np.float)
    out_y = np.zeros((1120,998,12), dtype=np.float)

    for i in range(len(ds)):
        out_x[i] = ds[i]["x"]
        out_y[i] = ds[i]["y"]

    # just testing the min/max range for sanity
    print (out_x.min(), out_x.max(), out_x.mean())
    print (out_y.min(), out_y.max(), out_y.mean())


