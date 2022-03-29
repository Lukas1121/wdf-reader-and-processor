from renishawWiRE import WDFReader,export
import matplotlib.pyplot as plt

class Main:
    def __init__(self,file):
        self.file = file
        self.reader = WDFReader(file)
        self.wn = self.reader.xpos
        self.sp = self.reader.spectra

    def print_nfo(self):
        self.reader.print_info()
        print(self.wn.shape,self.sp.shape)







