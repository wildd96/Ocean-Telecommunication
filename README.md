# Ocean_Telecommunication
This project is for Fall 2022 Discovery research program conducted through the Central Sierra Snow Lab.

Project by Andrew Wilder, Queeny Chan, and Chris Alvarez.

The goal of this project is to use historical ocean temperature data to predict precipitation levels in the Tahoe region of California. El Nino and La Nina cause the jetstream to either effect precipitation patterns directly north or directly south of this region, but it is unclear whether or not this affects the Tahoe region and this project is using a neural network to try and predict precipitation levels.

![el_vs_la_jetstream](https://user-images.githubusercontent.com/90016387/195717825-2abaa5a1-3e51-43a9-ad17-e0bbaa2e4f37.jpeg)

Our project began by collecting SNOTEL data from https://wcc.sc.egov.usda.gov/nwcc/tabget?state=CA and creating a data stream using PySpark. This will ensure that our model is always up to date on historical precipitation levels, and if there is a correlation it will make our model more accurate as it gets more data. 
