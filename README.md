**CAFNet**
======

## Description

**CAFNet** This repository provides the implementation of CAFNet, a robust unsupervised deep learning framework for denoising distributed acoustic sensing (DAS) data without requiring clean labels. CAFNet is a two-stage cascaded network consisting of a coarse module followed by a refined module. Each stage is built with fully connected layers, KAN network-based learnable activation functions, and attention mechanisms to adaptively capture coherent seismic signals.

## Reference
    Chen et al. (2025). Towards robust DAS denoising via unsupervised deep learning: The FORGE, Arcata–Eureka, and SAFOD examples, TBD.
    
BibTeX:

	@article{CAFNet,
	  title={Towards robust DAS denoising via unsupervised deep learning: The FORGE, Arcata–Eureka, and SAFOD examples},
	  author={Chen et al.},
	  journal={TBD},
	  volume={TBD},
	  number={TBD},
	  issue={TBD},
	  pages={TBD},
	  year={2026}
	}

## Scientific Application

CAFNet is designed for scientific workflows involving distributed acoustic sensing (DAS), where high‐resolution seismic information is often obscured by strong random and coherent noise. The proposed unsupervised framework enables researchers to denoise large‐scale DAS recordings without the need for curated clean datasets, making it particularly suitable for real-world observational studies.

By enhancing signal clarity and preserving coherent seismic phases, CAFNet facilitates multiple downstream scientific tasks, including:
	•	Earthquake detection and arrival picking
	•	Microseismic monitoring in carbon storage, geothermal, and oil & gas applications
	•	Ambient noise analysis and interferometry
	•	Distributed strain-rate and ground motion characterization
	•	Time-lapse subsurface imaging using DAS

-----------
## Copyright
    CSDL developing team, 2024-present
-----------

## License
    MIT License 

-----------

## Install
Using the latest version

    git clone https://github.com/chen-gui/CAFNet
    cd CAFNet
    pip install -v -e .

-----------
## Examples
    The "Demos" directory contains all runable scripts to demonstrate DAS denoising applications of CAFNet. 
    
-----------

-----------
## Dependence Packages
* scipy 
* numpy
* torch 
* matplotlib

-----------
## Development
    The development team welcomes voluntary contributions from any open-source enthusiast. 
    If you want to make contribution to this project, feel free to contact the development team. 

-----------
## Contact
    Regarding any questions, bugs, developments, or collaborations, please contact      
    Gui Chen
	chenguicup@163.com
