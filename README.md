# Parallel-COVID-Computations

### Project Description
The application of this project is to effectively analyze and interpret COVID-19 metrics for countries
around the world in order to classify countries based on the severity of their current situations. On a
high-level, the project aims to solve this problem efficiently by using parallelization techniques in CUDA
to read, analyze, and display results for input COVID data read from a CSV file. In order to verify the
computational speedup of the parallel solution, the application is implemented both parallel and
sequentially and the execution times are compared. The CSV file used by the application is available from: https://github.com/owid/covid-19-data/tree/master/public/data/latest 

The primary goal of this project was to achieve a non-trivial speedup for a COVID-19 data analysis tool,
along with expanding our current knowledge of parallel computing. Throughout the project we were
successful in creating speed-ups from a sequential version of the code to a parallel version of the code.
The primary speed-ups we observed were in the data analysis and data output sections.


### System Overview
For this project all tests and development were done on the Pitzer and Owens clusters of the Ohio
Supercomputer Center. Through course work at the University of Cincinnati, students are given access to
the system. The two clusters have similar statistics that allow for fast parallel computations. The Pitzer
cluster is “a 10,240-core Dell Intel Gold 6148 + 19,104-core Dual Intel Xeon 8268 machine” while
Owens is “a 23,392-core Dell Intel Xeon E5-2680 v4 machine” (OSC). On the remote sessions for the
Pitzer desktop, this project used the standard 48 cores and a single node. The Owens cluster desktop has a
standard of 28 cores that can be increased if necessary. Both provide a node that includes an NVIDIA
Tesla P100 GPU allowing for CUDA computations.


### Configuration
The main project file is _project.cu_. 
No external libraries or packages should be needed to run the program. Book.h and the Data directory are both used as imports.

Since the program is made up of three sections (reading, analysis, and results) files are also provided for each individual section. 
For each section, both a sequential and parallel implementation are included.
