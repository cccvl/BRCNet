from .abstract_dataset import AbstractDataset
from .abstract_dataset_aux import AbstractDataset_aux
from .faceforensics import FaceForensics
from .wild_deepfake import WildDeepfake
from .celeb_df import CelebDF
from .dfdc import DFDC
from .mydata import Mydata
from .mydata import Mydata_Val
from .mydata import Mydata_Test
from .mydata_aux import Mydata_aux
from .mydata_aux import Mydata_Val_aux
from .mydata_aux import Mydata_Test_aux

LOADERS = {
    "FaceForensics": FaceForensics,
    "WildDeepfake": WildDeepfake,
    "CelebDF": CelebDF,
    "DFDC": DFDC,
    "Mydata": Mydata,
    "Mydata_aux": Mydata_aux,
    "Mydata_Val": Mydata_Val,
    "Mydata_Test": Mydata_Test,
    "Mydata_Val_aux": Mydata_Val_aux,
    "Mydata_Test_aux": Mydata_Test_aux,
}


def load_dataset(name="FaceForensics"):
    print(f"Loading dataset: '{name}'...")
    return LOADERS[name]
