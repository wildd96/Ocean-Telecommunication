# Ocean_Telecommunication
This project is for Fall 2022 Discovery research program conducted through the Central Sierra Snow Lab.

Project by Andrew Wilder, Queeny Chan, and Chris Alvarez.

The goal of this project is to use historical ocean temperature and anomaly data to predict precipitation levels in the Tahoe region of California. El Nino Southern Oscillation (ENSO), Pacific Decadal Oscillation (PDO), and Arctic Oscillation (AO) cause the amplitude of the jetstream to effect precipitation patterns directly north or directly south of this region, but it is unclear whether or not this affects the Tahoe region. This project uses a Long Short Term Memory (LSTM) Neural Network to predict precipitation levels solely using the ocean data, and is able to do so with remarkable accuracy.

![el_vs_la_jetstream](https://user-images.githubusercontent.com/90016387/195717825-2abaa5a1-3e51-43a9-ad17-e0bbaa2e4f37.jpeg)

Data was collected from the following sources.
SNOTEL: https://wcc.sc.egov.usda.gov/nwcc/tabget?state=CA
ENSO: https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices
PDO: https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat
AO: https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/monthly.ao.index.b50.current.ascii.table

The data is pulled on a given interval and that will ensure that our model is always up to date on historical precipitation levels.
Autocorrelation and Seasonal Decomposition were used to find correlation between the oscillation data and the accumulated precipitation levels in the SNOTEL dataset.

This was presented in poster format through the Berkeley Discovery Research Program, and attached below is the poster.

![poster](https://user-images.githubusercontent.com/90016387/208290765-559fe23b-7db2-45f5-90aa-ed60287dd891.jpg)
