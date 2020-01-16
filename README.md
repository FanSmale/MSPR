MSPR
***
Based on the paper: Liu-Ying Wen, Chao-Guang Luo, Wei-Zhi Wu, Fan Min. Multi-label symbolic value partitioning through random walks. Neurocomputing, 2020.

https://doi.org/10.1016/j.neucom.2020.01.046

Liu-Ying Wen et al. propose the multi-label symbolic value partitioning through random walks algorithm with two stages. In the first stage, an undirected weighted graph is constructed for each attribute. Each node corresponds to an attribute value and the weight of each edge corresponds to the similarity between two nodes. Similarity is defined based on the attribute value distribution for each label. In the second stage, a random walk algorithm is used to cluster attribute values. The average weight serves as the separation operator to sharpen the inter-cluster edges.

***
Usage:

1. Download the repository;
2. Open the repository with "Eclipse" software;
3. Right click the project name "MSPR", and select "Build Path" --> "Add External Archives", left click;
4. Select all files with ".lib" extension in the "MSPR \ lib" folder
5. The main function is stored in "RandomwalkSymbolicValuePartition.java" file under "mspr" package;
6. Modify data file path in main function;
7. If it does not exist, create a new folder named "Result" in the root directory of the "MSPR" project;
8. If it does not exist, create a new folder with the same name as the data set name;
9. Run "RandomwalkSymbolicValuePartition.java" file as java application.