

![](logo.png)

# Welcome to SPI2Py

SPI2 stands for the Spatial Packaging of Interconnected Systems with Physical Interactions.

The SPI2 framework packages components, routes interconnects, and performs multiphysics simulations simultaneously.

At this point in time, we are working on the initial release so many features are missing/untested. We plan to 
demonstrate an early working version in mid-November of 2022.

For more information regarding the SPI2 Strategic Research Initiative see the following.

Website: 

https://spi2.illinois.edu/

A quick note on the setup:

SnapPy version 3.1b1 (not released) is required to calculate the Yamada polynomials of a given spatial topology. 
SnapPy version 3.0.3 is the latest pip-installable release on PyPI. Version 3.1b1 is planned for release around January
of 2023. In the meantime, most people can likely omit this dependency. If you need it, then you must build it from the 
source. Chad can give better details if needed.
1. Setup and activate your virtual environment. With the activated venv do the following.
2. Install cython with "pip install cython"
3. Choose a directory to install snappy
4. Command "git clone https://github.com/3-manifolds/SnapPy.git"
5. Command "cd SnapPy" to into the SnapPy directory.
6. Command "pip install --upgrade ." (space and period included)

## Publications

Satya R T Peddada, Kai A James, James T Allison. 
'A Novel Two-Stage Design Framework for 2d Spatial Packing of Interconnected Components.' 
ASME Journal of Mechanical Design, Dec 2020.
DOI: [10.1115/1.4048817](https://dx.doi.org/10.1115/1.4048817)
```
@article{peddada2021novel,
  title={A novel two-stage design framework for two-dimensional spatial packing of interconnected components},
  author={Peddada, Satya RT and James, Kai A and Allison, James T},
  journal={Journal of Mechanical Design},
  volume={143},
  number={3},
  year={2021},
  publisher={American Society of Mechanical Engineers Digital Collection}
}
```

TODO Add the new SPI2 JMD article once published.
