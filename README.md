# wdf-reader-test
The code takes measured pure spectra from Raman samples, specifically Apifil, Butamben, Transcutol, and Capryol in the current implementation. It then subtracts these pure spectra from Raman maps obtained from a mixed sample. This process generates color-coded maps that visually highlight the concentration of each individual species within the sample.

First, the raw maps undergo a series of preprocessing operations designed to filter out cosmic rays. Following this, the spectra are normalized, and an asymmetric least squares baseline subtraction is applied to further clean and refine the data.

Finally, the cleaned spectra are compared with the spectra of the pure samples, resulting in the generation of 2D maps that depict the concentration of each substance within the sample.

NOTICE!
This piece of code is not currently user friendly. There are also some changes in  one of the packages utilized that is causing issues with one of the main functions of the code and therefore needs to be resolved in order for the code to work.

