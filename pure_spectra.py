import os.path

from renishawWiRE import WDFReader
from bin.open_wdf import PreProcessing
import glob

class PreProcessPureSpectra():
    def __init__(self,pure_spectra_dir):
        self.S_T = self.pack_components(path=pure_spectra_dir)

    def pack_components(self,path):
        S_T = []
        for f in glob.glob(path + "/*"):
            if os.path.isdir(f) == False:
                spectra = PreProcessing.norm_by_unit_vector(PreProcessing.subtract_ALS_all(self,WDFReader(f).spectra))
                mean_spectra = PreProcessing.mean_spectrum(self,spectra)
                S_T.append(mean_spectra)
        return S_T



