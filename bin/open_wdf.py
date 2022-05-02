from renishawWiRE import WDFReader
import scipy.stats
import numpy as np
from numpy.linalg import inv
from bin import ALS
from bin.ExternalFunctions import nice_string_output, add_text_to_ax
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import scipy.optimize as optimization
import glob
import os.path
import os


class OpenWdfFile():
    def __init__(self,file,path=None):
        self.file = file
        self.path = path
        self.reader = WDFReader(file)
        self.wn = self.reader.xdata
        self.sp = self.reader.spectra

class RemoveCosmicRays(OpenWdfFile):
    def __init__(self,file):
        super().__init__(file)

    def z_score(self,sp):
        return np.array(scipy.stats.zscore(sp,ddof=15))

    def unpack_data_cube(self,reader,spectra,plot=False,remove_cosmic_rays=False,threshold=7):
        D = []
        width, height = reader.map_shape
        if remove_cosmic_rays:
            self.remove_bad_pixels(plot=False,threshold=threshold)
            plt.cla()
            plt.close()
        for i in range(height):
            for j in range(width):
                D.append(spectra[i, j, :])
                if plot:
                    plt.plot(self.wn,spectra[i, j, :])
        if plot:
            plt.xlabel(r"$cm^{-1}$", fontsize=20)
            plt.ylabel("Intensity (I)", fontsize=20)
            plt.tight_layout()
            plt.show()
        return np.array(D)

    def als_baseline(self,intensities,deriv_order=11, asymmetry_param=0.05, smoothness_param=1e6,
                     max_iters=20, conv_thresh=1e-7, verbose=False):
        '''Computes the asymmetric least squares baseline.
        * http://www.science.uva.nl/~hboelens/publications/draftpub/Eilers_2005.pdf
        smoothness_param: Relative importance of smoothness of the predicted response.
        asymmetry_param (p): if y > z, w = p, otherwise w = 1-p.
                           Setting p=1 is effectively a hinge loss.
        '''
        smoother = ALS.WhittakerSmoother(intensities, smoothness_param, deriv_order=deriv_order)
        # Rename p for concision.
        p = asymmetry_param
        # Initialize weights.
        w = np.ones(intensities.shape[0])
        for i in range(max_iters):
            z = smoother.smooth(w)
            mask = intensities > z
            new_w = p * mask + (1 - p) * (~mask)
            conv = np.linalg.norm(new_w - w)
            if verbose:
                print(i + 1, conv)
            if conv < conv_thresh:
                break
            w = new_w
        # print('ALS did not converge in %d iterations' % max_iters)
        return z

    def remove_bad_pixels(self,threshold=7,plot=True):
        width, height = self.reader.map_shape
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))
        for i in range(height):
            for j in range(width):
                z_scores = self.z_score(sp=self.sp[i, j, :])
                spikes = abs(z_scores) > threshold
                axes[0].plot(self.wn,spikes)
                if 1 in spikes:
                    self.sp[i,j] = self.cosmic_ray_removal(i,j)
                    axes[1].plot(self.wn, self.sp[i,j,:])
                else:
                    axes[1].plot(self.wn, self.sp[i, j, :])
        if plot:
            axes[1].set_ylabel("Cosmic rays to be filtered", fontsize=20)
            axes[1].set_xlabel(r"$cm^{-1}$", fontsize=20)
            axes[0].set_ylabel("z_scores", fontsize=20)
            axes[1].set_xlabel(r"$cm^{-1}$", fontsize=20)
            plt.xlabel(r"$cm^{-1}$", fontsize=20)
            plt.ylabel("Intensity (I)", fontsize=20)
            plt.tight_layout()
            plt.show()

    def cosmic_ray_removal(self,i,j):
        width, height = self.reader.map_shape
        mean = np.zeros(len(self.sp[i,j,:]))
        if j >=(width-1):
            if i >=(height-1):
                for n in range(len(mean)):
                    mean[n] = 2*self.sp[i-1, j, n]/6 + 2*self.sp[i, j-1, n]/6 + 2*self.sp[i-1, j-1, n]/6
            else:
                for n in range(len(mean)):
                    mean[n] = self.sp[i, j - 1, n] / 3 + self.sp[i+1, j, n] / 3 + \
                              self.sp[i-1, j, n] / 3
        if i >=(height - 1):
            if j >=(width - 1):
                for n in range(len(mean)):
                    mean[n] = self.sp[i-1, j, n]/3 + self.sp[i, j-1, n]/3 + self.sp[i-1, j-1, n]/3
            else:
                for n in range(len(mean)):
                    mean[n] = self.sp[i, j - 1, n] / 3 + self.sp[i, j+1, n] / 3 + \
                              self.sp[i-1, j, n] / 3
        if j < (width-1):
            if i < (height-1):
                for n in range(len(mean)):
                    mean[n] = self.sp[i, j - 1, n] / 4 + self.sp[i + 1, j, n] / 4 + \
                              self.sp[i - 1, j, n] / 4 + self.sp[i, j+1, n] / 4
        else:
             print("something went wrong at index [{0},{1}]".format(i,j))
        return mean

class PreProcessing(RemoveCosmicRays):
    def __init__(self,file,remove_cosmic_rays=False):
        super().__init__(file)
        if file == None:
            self.D = self.sp
        else:
            self.D = RemoveCosmicRays.unpack_data_cube(self, reader=self.reader,spectra=self.sp,
                                                               remove_cosmic_rays=remove_cosmic_rays,
                                                               threshold=15)

    def subtract_ALS_all(self,spectrum,deriv_order=14):
        for i in range(len(spectrum)):
            spectrum[i] = self.als_baseline(spectrum[i],deriv_order=deriv_order,max_iters=10)
        return spectrum

    def mean_spectrum(self,spectrum):
        return np.mean(spectrum,axis=0)

    def normalized(self,a, axis=-1, order=2):
        l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
        l2[l2 == 0] = 1
        return a / np.expand_dims(l2, axis)

    def norm_by_unit_vector(self,x):
        # norm1 = x/np.linalg.norm(x,ord=1)
        norm2 = self.normalized(x)
        return norm2

class CLS(PreProcessing):
    def __init__(self,file,c=None,pure_spectra_path=None,remove_cosmic_rays=False):
        if pure_spectra_path != None:
            self.S_T,_ = self.unpack_components(path=pure_spectra_path)
        else:
            self.S_T = np.load("data/Blind_test/raw/numpy_arr/S_T_K_means.npy",allow_pickle=True)
        self.c = c
        super().__init__(file)
        self.D = self.unpack_data_cube(reader=WDFReader(file),spectra=WDFReader(file).spectra,
                                       remove_cosmic_rays=remove_cosmic_rays,threshold=15)
        self.D = self.norm_by_unit_vector(self.subtract_ALS_all(self.D))

    def return_D_with_wn(self):
        return self.D,self.wn

    def return_width_height(self):
        return WDFReader(self.file).map_shape

    def unpack_components(self,path,save_S_T=False):
        S_T = []
        filenames = []
        for f in glob.glob(path + "/*"):
            if not os.path.isdir(f):
                if os.path.basename(f[-4:-1]) != ".np":
                    spectra = self.unpack_data_cube(reader=WDFReader(f),spectra=WDFReader(f).spectra)
                    spectra = self.norm_by_unit_vector(self.subtract_ALS_all(spectra))
                    mean_spectrum = self.mean_spectrum(spectra)
                    S_T.append(mean_spectrum)
                    filenames.append(f)
        if save_S_T:
            S_t = S_T
            S_T[2],S_T[3] = S_t[3],S_t[2]
            np.save("data/Blind_test/raw/numpy_arr/S_T_K_means.npy",np.array(S_T))
        return np.array(S_T),filenames

    def least_sq(self,D, S_T):
        # Sample_spectrum (unknown spectrum): array of w values.
        # Components (known spectra): array of n (number of components) columns with w values.
        # This def returns an array of n values. Each value is the similarity score for the
        # sample_spectrum and a component spectrum.
        similarity = np.dot(inv(np.dot(S_T, S_T.T)), np.dot(S_T, D))
        return similarity

    def nn_least_sq(self,D,b):
        temp = [0,0,0,0]
        for i in range(len(temp)):
            val,er = optimization.nnls(b,D[i])
            print(val)
            temp[i] = val
        return np.array(temp)

    def least_sq2(self,c,D):
        def func(params, xdata, ydata):
            return (ydata - np.dot(xdata.T, params)) ** 2
        x0 = np.ones(len(c))
        return optimization.leastsq(func, x0, args=(c, D))[0]

    def refold_scores(self,D_1d_i,S_T,s_i):
        width,height = self.reader.map_shape
        D_2d = self.sp
        A = np.zeros(len(D_1d_i)).reshape(height, width)
        D_1d_i.resize(height,width)
        for (idx) in np.ndenumerate(A):
            i = int(idx[0][0])
            j = int(idx[0][-1])
            similarity = self.least_sq(self.norm_by_unit_vector(
                self.als_baseline(D_2d[i,j]))[-1,:],S_T)
            # print(np.sum(similarity))
            A[i,j] = ((D_1d_i[i,j]+similarity[s_i])/2)*100
        return A.T

    def refold_map(self,A,S_T,it_range=5, n=0):
        D = self.reader.spectra
        height, width = self.reader.map_shape
        while True:
            if n == it_range:
                return map
            map = A.reshape(height, width)
            # A.resize(height,width)
            for (idx) in np.ndenumerate(map):
                i = int(idx[0][0])
                j = int(idx[0][-1])
                similarity = self.least_sq(self.norm_by_unit_vector(
                    self.als_baseline(D[i, j])).T, S_T)
                map[i, j] = 1/(map[i, j] + D[i,j])/(map[i,j]*D[i,j])
                # print(similarity)
            n += 1

class Plotting(CLS):
    def __init__(self,file,*args):
        for arg in args:
            self.arg = arg
        super().__init__(file=file)

    def quick_plot(self,x,*args):
        fig, axes = plt.subplots()
        for arg in args:
            axes.plot(x,arg,linestyle="--")
        # axes.plot(self.wn,self.D,label="big map mean spectra",linewidth=3)
        plt.legend()
        plt.show()

    def plot_maps_from_arr(self,z, titles, ticks):
        fig, axs = plt.subplots(2, 2, figsize=(8, 8))
        fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.05)
        for title, z_i, ax, tick in zip(titles, z, axs.ravel(), ticks):
            levels = MaxNLocator(nbins=20).tick_values(tick[0], tick[-1])
            cm = plt.get_cmap("spring")

            d = {"mean": np.mean(z_i.flatten()),
                 "std": np.std(z_i.flatten())}

            text = nice_string_output(d, extra_spacing=2, decimals=2)
            add_text_to_ax(0.02, 0.97, text, ax, fontsize=16)

            norm = BoundaryNorm(levels, ncolors=cm.N, clip=True)
            im = ax.pcolormesh(z_i, cmap=cm, norm=norm)
            ax.set_title("{0}".format(title), fontsize=20)
            fig.colorbar(im, ax=ax)
        plt.tight_layout()
        return fig, axs
