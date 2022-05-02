import os.path
from bin import open_wdf
import numpy as np
import matplotlib.pyplot as plt


file = "data/Blind_test/raw/map5_Hery.wdf"
path_pure_spectra = "data/Blind_test/raw/pure_spectra"
path_blind_maps = "data/Blind_test/raw"

c = np.array([0.1987,0.3982,0.1013,0.3018])

obj = open_wdf.CLS(file=file,c=c,pure_spectra_path=path_pure_spectra,remove_cosmic_rays=True)
D,wn = obj.return_D_with_wn()
# D = obj.mean_spectrum(D)
# np.save("data/Blind_test/raw/numpy_arr/big_map_K_mean.npy",np.array((wn, D),dtype=object))
# D_wn = np.load("data/Blind_test/raw/numpy_arr/big_map_K_mean.npy",allow_pickle=True)
S_T,_ = obj.unpack_components(path_pure_spectra,save_S_T=True)
# S_T = np.load("data/Blind_test/raw/numpy_arr/S_T_K_means.npy",allow_pickle=True)

def least_sq_loop(D,S_T,similarities_dict={"apifil":[],"butamben":[],"capryol":[],"transcutol":[]}):
    sum_lst = [[],[],[],[]]
    for i in range(len(D)):
        similarities = obj.least_sq(D[i], S_T)
        # print("butamben pixel nr. {0} = {1} for CLS ".format(i,similarities[1]))
        similarities_dict["apifil"].append(similarities[0])
        similarities_dict["butamben"].append(similarities[1])
        similarities_dict["capryol"].append(similarities[2])
        similarities_dict["transcutol"].append(similarities[3])
        sum_lst[0].append(similarities[0])
        sum_lst[1].append(similarities[1])
        sum_lst[2].append(similarities[2])
        sum_lst[3].append(similarities[3])
    return similarities_dict, np.array(sum_lst)

similarities_dict,sum_lst = least_sq_loop(D,S_T)
print("Apifil: {0} \n Butamben: {1} \n Capryol: {2} \n Transecutol: {3}".format(np.sum(sum_lst[0])/len(sum_lst[0]),np.sum(sum_lst[1])/len(sum_lst[1]),np.sum(sum_lst[2])/len(sum_lst[2]),np.sum(sum_lst[3])/len(sum_lst[3])))

A_apifil = open_wdf.CLS.refold_scores(obj,D_1d_i=np.array(similarities_dict["apifil"]),S_T=S_T,s_i=0)
A_butamben = open_wdf.CLS.refold_scores(obj,D_1d_i=np.array(similarities_dict["butamben"]),S_T=S_T,s_i=1)
A_capryol = open_wdf.CLS.refold_scores(obj,D_1d_i=np.array(similarities_dict["capryol"]),S_T=S_T,s_i=2)
A_transcutol = open_wdf.CLS.refold_scores(obj,D_1d_i=np.array(similarities_dict["transcutol"]),S_T=S_T,s_i=3)
A_butamben = open_wdf.CLS.refold_map(obj,A_butamben,S_T)
A_apifil = open_wdf.CLS.refold_map(obj,A_apifil,S_T)
A_transcutol = open_wdf.CLS.refold_map(obj,A_transcutol,S_T)
A_capryol = open_wdf.CLS.refold_map(obj,A_capryol,S_T)


fig,axes = open_wdf.Plotting.plot_maps_from_arr(obj,z=[A_butamben,A_capryol,A_transcutol,A_apifil],
                   titles=["Butamben","Capryol","Transcutol","Apifil"],
                   ticks=[(0,55),(-25,30),(0,40),(0,100)])
# plt.savefig("data/Blind_test/raw/plots/{0}.jpg".format(os.path.basename(file)[0:-4]))
plt.show()