from setuptools import setup, find_packages
import os


VERSION = '2.2.14'
DESCRIPTION = """
Paper Title: ErrorEval: an Open-Source Worst-Case-Error Evaluation Framework for Approximate Computing

Short Description: The open-source toolchain that proposes a methodology called ErrorEval which relies on 
SMT (Satisfiability Modulo Theories) solvers.

Authors: 
Morteza Rezaalipour, Università della Svizzera italiana (USI), Lugano, Switzerland
Lorenzo Ferretti, Micron Technology, San Jose, USA
Ilaria Scarabottolo, Università della Svizzera italiana (USI), Lugano, Switzerland
George A. Constantinides, Imperial College London, London, UK
and Laura Pozzi, Università della Svizzera italiana (USI), Lugano, Switzerland

Event name: (Computing Frontiers - Workshop on Open-Source Hardware) CF23-OSHW 23, May 9-11, 2023, Bologna, Italy
DOI: https://doi.org/10.1145/3587135.3591438
"""
INSTALL_REQUIREMENTS_FILE = 'install_requiremnets.txt'
INSTALL_REQUIREMENTS = []
with open(INSTALL_REQUIREMENTS_FILE, 'r') as IR:
    lines = IR.readlines()
    for line in lines:
        INSTALL_REQUIREMENTS.append(line.replace('\n', ""))


# Setting up
setup(
    name="z3log",
    version=VERSION,
    author="Morteza Rezaalipour (MorellRAP)",
    author_email="<rezaalipour.usi@gmail.com>",
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=INSTALL_REQUIREMENTS,
    keywords=['python', 'verilog', 'circuits', 'synthesis'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
) 
