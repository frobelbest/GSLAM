# GSLAM: Initialization-robust Monocular Visual SLAM via Global Structure-from-Motion

For more information see
[https://frobelbest.github.io/gslam](https://frobelbest.github.io/gslam)

### 1. Related Papers
* **GSLAM: Initialization-robust Monocular Visual SLAM via Global Structure-from-Motion**, *C. Tang, O. Wang, P. Tan*, In 3DV,2017
* **Global Structure-from-Motion by Similarity Averaging**, *Z. Cui, P. Tan*, In ICCV, 2015

Get some datasets from [https://vision.in.tum.de/mono-dataset](https://vision.in.tum.de/mono-dataset) .

### 2. Installation

	git clone https://github.com/frobelbest/GSLAM.git

#### 2.1 Required Dependencies

##### Theia Vision Library (required for global rotation averaging).
Install from [http://www.theia-sfm.org](http://www.theia-sfm.org)
##### Ceres Solver (required for local and global bundle adjustment).
Install from [http://ceres-solver.org](http://ceres-solver.org)
##### CLP (required for global scale and translation averaging).
Install from [https://projects.coin-or.org/Clp](https://projects.coin-or.org/Clp)
##### OpenCV (required for image processing).
Install from [https://opencv.org](https://opencv.org)
##### Pangolin (required for visualization).
Install from [https://github.com/stevenlovegrove/Pangolin](https://github.com/stevenlovegrove/Pangolin)

#### 2.3 Build
Currently, only the xcode project is supplied. You can write your own code to compile on other platforms or wait for future update.

### 3 Usage
Run on a dataset from [https://vision.in.tum.de/mono-dataset](https://vision.in.tum.de/mono-dataset) using
GSLAM [sequence_path] [vocabulary_path], for example  GSLAM ./bear ./Vocabulary/ORBvoc.txt

#### 3.1 Dataset Format.


### Notes

#### Real-time KLT with AVX Acceleration
Except the method proposed in the paper, this project also featured in a highly optimized KLT Tracker which can track more than 4000 points on a 1080p video in real-time.

#### ToDo
The main bottleneck for this project is the feature tracking, which can be further improved by the paper "Better feature tracking through subspace constraints".

### 5 License
GSLAM was developed at the Simon Fraser University and Adobe.
The open-source version is licensed under the GNU General Public License
Version 3 (GPLv3).For commercial purposes, please contact [cta73@sfu.ca](cta73@sfu.ca) or [pingtan@sfu.ca](pingtan@sfu.ca)
