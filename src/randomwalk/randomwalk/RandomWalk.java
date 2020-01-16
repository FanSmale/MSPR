package randomwalk.randomwalk;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.Arrays;

import randomwalk.clustering.*;
import randomwalk.matrix.CompressedMatrix;
import randomwalk.matrix.CompressedSymmetricMatrix;
import randomwalk.matrix.Triple;


public class RandomWalk {
	CompressedMatrix originalMatrix;
	
	
	public RandomWalk(String paraFilename , int[][][] paraNewIndex, int paraAttribute) {
		originalMatrix = new CompressedMatrix(paraFilename, paraNewIndex, paraAttribute);
		// System.out.println("The original matrix is: " + originalMatrix);
	}//Of the second constructor
	public RandomWalk(String paraFilename ) {
		originalMatrix = new CompressedMatrix(paraFilename);
		// System.out.println("The original matrix is: " + originalMatrix);
	}//Of the second constructor
/*	
	public RandomWalk(String paraFilename, int paraMinNeighbors) {
		originalMatrix = new CompressedMatrix(paraFilename, paraMinNeighbors);
		System.out.println("The original matrix is: " + originalMatrix);
	}//Of the second constructor
	*/
	/**
	 *********************
	 * The main algorithm.
	 * 
	 * @param paraFilename
	 *            The name of the decision table, or triple file.
	 * @param paraNumRounds
	 *            The rounds for random walk, each round update the weights,
	 *            however does not change the topology.
	 * @param paraK
	 *            The maximal times for matrix multiplex.
	 * @param paraMinNeighbors
	 *            For converting decision system into matrix only.
	 * @param paraCutThreshold
	 *            For final clustering from the result matrix. Links smaller
	 *            than the threshold will break.
	 *********************
	 */
	public void randomWalk(int paraNumRounds, int paraK, 
			double paraCutThreshold) {
		// Step 1. Read data
		CompressedMatrix tempMultiplexion, tempCombinedTransitionMatrix;

		// Step 2. Run a number of rounds to obtain new matrices
		for (int i = 0; i < paraNumRounds; i++) {
			// Step 2.1 Compute probability matrix
			CompressedMatrix tempProbabilityMatrix = originalMatrix.computeTransitionProbabilities();
			// System.out.println("\r\nThe probability matrix is:" + tempProbabilityMatrix);
			// Make a copy
			tempMultiplexion = new CompressedMatrix(tempProbabilityMatrix);

			// Step 2.2 Multiply and add
			// Reinitialize
			tempCombinedTransitionMatrix = new CompressedMatrix(tempProbabilityMatrix);
			for (int j = 2; j <= paraK; j++) {
				// System.out.println("j = " + j);
				tempMultiplexion = CompressedMatrix.multiply(tempMultiplexion, tempProbabilityMatrix);
				tempCombinedTransitionMatrix = CompressedMatrix.add(tempCombinedTransitionMatrix, tempMultiplexion);
			} // Of for j

			// System.out.println("Find the error!" + originalMatrix);

			// Step 2.3 Distance between adjacent nodes
			for (int j = 0; j < originalMatrix.matrix.length; j++) {
				Triple tempCurrentTriple = originalMatrix.matrix[j].next;
				while (tempCurrentTriple != null) {
					// Update the weight
					tempCurrentTriple.weight = tempCombinedTransitionMatrix.neighborhoodSimilarity(j,
							tempCurrentTriple.column, paraK);

					tempCurrentTriple = tempCurrentTriple.next;
				} // Of while
			} // Of for i
		} // Of for i

		// System.out.println("The new matrix is:" + originalMatrix);

		// Step 3. Depth-first clustering and output
		//originalMatrix.depthFirstClustering(paraCutThreshold);
		
		// Step 3'. Width-first clustering and output
		try {
			originalMatrix.widthFirstClustering(paraCutThreshold);
		} catch (Exception ee) {
			System.out.println("Error occurred in random walk: " + ee);
		}//Of try
	}// Of randomWalk
	/**
	 *********************
	 * 获取所有节点的权重值.
	 * 
	 * @param paraOriginalMatrix
	 *            原始矩阵信息.
	 *@return 权重值数组
	 *********************
	 */
	public double[] getAllWeight(CompressedMatrix paraOriginalMatrix ){
		Triple tempTriple;
		int countLength = 0;
		for (int i = 0; i < paraOriginalMatrix.matrix.length; i++) {
			tempTriple = paraOriginalMatrix.matrix[i].next;
			while (tempTriple != null) {
				//System.out.println("当前权重值：" + tempTriple.weight);
				countLength++;
				tempTriple = tempTriple.next;
			} // Of while
		} // Of for i
		
		double[] result = new double[countLength];
		int index = 0;
		for (int i = 0; i < paraOriginalMatrix.matrix.length; i++) {
			tempTriple = paraOriginalMatrix.matrix[i].next;
			while (tempTriple != null) {
				//System.out.println("当前权重值：" + tempTriple.weight);
				countLength++;
				result[index] = tempTriple.weight;
				tempTriple = tempTriple.next;
				index++;
			} // Of while
		} // Of for i
		return result;
	}//end of getAllWeight
	
	
	/**
	 *********************
	 * The main algorithm.
	 * 
	 * @param paraFilename
	 *            The name of the decision table, or triple file.
	 * @param paraNumRounds
	 *            The rounds for random walk, each round update the weights,
	 *            however does not change the topology.
	 * @param paraK
	 *            The maximal times for matrix multiplex.
	 * @param paraMinNeighbors
	 *            For converting decision system into matrix only.
	 * @param paraCutThreshold
	 *            For final clustering from the result matrix. Links smaller
	 *            than the threshold will break.
	 *@return 聚类结果
	 *********************
	 */
	public int[] randomWalkMy(int paraNumRounds, int paraK, 
			double paraCutThreshold) {
		// Step 1. Read data
		CompressedMatrix tempMultiplexion, tempCombinedTransitionMatrix;

		// Step 2. Run a number of rounds to obtain new matrices
		for (int i = 0; i < paraNumRounds; i++) {
			
			// System.out.println("------------------");
			// System.out.println("paraNumRounds = " + i);
			
			// Step 2.1 Compute probability matrix
			CompressedMatrix tempProbabilityMatrix = originalMatrix.computeTransitionProbabilities();
			// System.out.println("\r\nThe probability matrix is:" + tempProbabilityMatrix);
			// Make a copy
			tempMultiplexion = new CompressedMatrix(tempProbabilityMatrix);

			// Step 2.2 Multiply and add
			// Reinitialize
			tempCombinedTransitionMatrix = new CompressedMatrix(tempProbabilityMatrix);
			for (int j = 2; j <= paraK; j++) {
				// System.out.println("j = " + j);
				tempMultiplexion = CompressedMatrix.multiply(tempMultiplexion, tempProbabilityMatrix);
				tempCombinedTransitionMatrix = CompressedMatrix.add(tempCombinedTransitionMatrix, tempMultiplexion);
			} // Of for j

			//System.out.println("Find the error!" + originalMatrix);

			// Step 2.3 Distance between adjacent nodes
			for (int j = 0; j < originalMatrix.matrix.length; j++) {
				Triple tempCurrentTriple = originalMatrix.matrix[j].next;
				while (tempCurrentTriple != null) {
					// Update the weight
					tempCurrentTriple.weight = tempCombinedTransitionMatrix.neighborhoodSimilarity(j,
							tempCurrentTriple.column, paraK);

					tempCurrentTriple = tempCurrentTriple.next;
				} // Of while
			} // Of for i
		} // Of for i

		System.out.println("The new matrix is:" + originalMatrix);
		int[] clusterResult = null;
		
		// Step 3. Depth-first clustering and output
		clusterResult = originalMatrix.depthFirstClustering(paraCutThreshold);
		
		// Step 3'. Width-first clustering and output
		//int[] clusterResult = null;
		/*
		try {
			clusterResult = originalMatrix.widthFirstClustering(paraCutThreshold);
		} catch (Exception ee) {
			System.out.println("Error occurred in random walk: " + ee);
		}//Of try
		*/
		return clusterResult;
	}// Of randomWalk
	
	/**
	 *************************** 
	 * Compress an double array so that no duplicate elements, no redundant elemnts
	 * exist, and it is in an ascendent order. <br>
	 * 
	 * @param paraIntArray
	 *            The given double array.
	 * @param paraLength
	 *            The effecitive length of the given double array.
	 * @return The constructed array.
	 *************************** 
	 */
	public double[] compressAndSortDoubleArray(double[] paraIntArray,
			int paraLength) {
		double[] noDuplicateArray = new double[paraLength];
		int realLength = 0;
		double currentLeast = 100000;
		int currentLeastIndex = 0;
		for (int i = 0; i < paraLength; i++) {
			if (paraIntArray[i] == Integer.MAX_VALUE) {
				continue;
			}

			currentLeast = paraIntArray[i];
			currentLeastIndex = i;

			for (int j = i + 1; j < paraLength; j++) {
				if (paraIntArray[j] < currentLeast) {
					currentLeast = paraIntArray[j];
					currentLeastIndex = j;
				}// Of if
			}// Of for j

			// Swap. The element of [i] should be stored in another place.
			paraIntArray[currentLeastIndex] = paraIntArray[i];

			noDuplicateArray[realLength] = currentLeast;
			realLength++;

			// Don't process this data any more.
			for (int j = i + 1; j < paraLength; j++) {
				if (paraIntArray[j] == currentLeast) {
					paraIntArray[j] = Integer.MAX_VALUE;
				}// Of if
			}// Of for j
		}// Of for i

		double[] compressedArray = new double[realLength];
		for (int i = 0; i < realLength; i++) {
			compressedArray[i] = noDuplicateArray[i];
		}// Of for i

		return compressedArray;
	}// Of compressAndSortIntArray
	
	/**
	 *********************
	 * The main algorithm.
	 * 
	 * @param paraNumRounds
	 *            The rounds for random walk, each round update the weights,
	 *            however does not change the topology.
	 * @param paraK
	 *            The maximal times for matrix multiplex.
	 *@return 聚类结果
	 *********************
	 */
	public int[] randomWalkAdaptive(int paraNumRounds, int paraK) {
		// Step 1. Read data
		CompressedMatrix tempMultiplexion, tempCombinedTransitionMatrix;

		// Step 2. Run a number of rounds to obtain new matrices
		for (int i = 0; i < paraNumRounds; i++) {
			
			// System.out.println("------------------");
			// System.out.println("paraNumRounds = " + i);
			
			// Step 2.1 Compute probability matrix
			CompressedMatrix tempProbabilityMatrix = originalMatrix.computeTransitionProbabilities();
			// System.out.println("\r\nThe probability matrix is:" + tempProbabilityMatrix);
			// Make a copy
			tempMultiplexion = new CompressedMatrix(tempProbabilityMatrix);

			// Step 2.2 Multiply and add
			// Reinitialize
			tempCombinedTransitionMatrix = new CompressedMatrix(tempProbabilityMatrix);
			for (int j = 2; j <= paraK; j++) {
				// System.out.println("j = " + j);
				tempMultiplexion = CompressedMatrix.multiply(tempMultiplexion, tempProbabilityMatrix);
				tempCombinedTransitionMatrix = CompressedMatrix.add(tempCombinedTransitionMatrix, tempMultiplexion);
			} // Of for j

			//System.out.println("Find the error!" + originalMatrix);

			// Step 2.3 Distance between adjacent nodes
			for (int j = 0; j < originalMatrix.matrix.length; j++) {
				Triple tempCurrentTriple = originalMatrix.matrix[j].next;
				while (tempCurrentTriple != null) {
					// Update the weight
					tempCurrentTriple.weight = tempCombinedTransitionMatrix.neighborhoodSimilarity(j,
							tempCurrentTriple.column, paraK);

					tempCurrentTriple = tempCurrentTriple.next;
				} // Of while
			} // Of for i
		} // Of for i

		//System.out.println("The new matrix is:" + originalMatrix);
		int[] clusterResult = null;
		
		double[] weightArray = getAllWeight(originalMatrix);
	    // System.out.println("权重值列表：");
		// for(int i = 0; i < weightArray.length; i++)
			// System.out.println(weightArray[i]);
		
		double[] weightArrayAfterSorted = compressAndSortDoubleArray(weightArray, weightArray.length);
		
		double averageWeight = 0;
		double total = 0;
		for(int i = 0; i < weightArrayAfterSorted.length; i++){
			total += weightArrayAfterSorted[i];
		}
		averageWeight = total/weightArrayAfterSorted.length;
		
		/*
		double threshold = 0;
		
		if(weightArrayAfterSorted.length == 1 || weightArrayAfterSorted.length == 2)
			threshold = weightArrayAfterSorted[weightArrayAfterSorted.length - 1];
		else
			threshold = weightArrayAfterSorted[2];
		*/	
		/*
		System.out.println("经过压缩排序后的权重值列表：");
		for(int i = 0; i < weightArrayAfterSorted.length; i++)
			System.out.println(weightArrayAfterSorted[i]);
		
		System.out.println("随机游走阈值为： " + weightArrayAfterSorted[0]);
		*/
		
		// Step 3. Depth-first clustering and output
		clusterResult = originalMatrix.depthFirstClustering(averageWeight);
		
		// Step 3'. Width-first clustering and output
		//int[] clusterResult = null;
		/*
		try {
			clusterResult = originalMatrix.widthFirstClustering(paraCutThreshold);
		} catch (Exception ee) {
			System.out.println("Error occurred in random walk: " + ee);
		}//Of try
		*/
		return clusterResult;
	}// Of randomWalk
	
	public static void main(String args[]) throws FileNotFoundException {
		
		// File test = new File("./Output.txt");
	//	PrintStream out = new PrintStream(new FileOutputStream(test));
		//System.setOut(out);
		System.out.println("The new matrix is: \r\n" );
		System.out.println("Let's randomly walk!");
		
		
		// KMeans tempMeans = new
		// KMeans("D:/workplace/randomwalk/data/iris.arff");
		// KMeans tempMeans = new
		// KMeans("D:/workspace/randomwalk/data/iris.arff");
		// Walk tempWalk = new Walk("D:/workspace/randomwalk/data/iris.arff");
		// int[] tempIntArray = {1, 2};

		// tempMeans.kMeans(3, KMeans.MANHATTAN);
		// tempMeans.kMeans(3, KMeans.EUCLIDEAN);
		// tempWalk.computeVkS(tempIntArray, 3);
		// double[][] originalMatrix = tempWalk.computeTransitionProbabilities();
		// double[][] tempTransition =
		// tempWalk.computeKStepTransitionProbabilities(100);
		// double[][] tempTransition =
		// tempWalk.computeAtMostKStepTransitionProbabilities(5);

		// double[][] tempNewGraph = tempWalk.ngSeparate(3);

		// System.out.println(Arrays.deepToString(originalMatrix));

		// System.out.println("The new graph is:\r\n" +
		// Arrays.deepToString(tempNewGraph));

		// CompressedSymmetricMatrix originalMatrix = new
		// CompressedSymmetricMatrix("D:/workspace/randomwalk/data/iris.arff",
		// 3);
		// CompressedSymmetricMatrix originalMatrix2 =
		// CompressedSymmetricMatrix.multiply(originalMatrix, originalMatrix);
		// CompressedSymmetricMatrix originalMatrix2 =
		// CompressedSymmetricMatrix.weightMatrixToTransitionProbabilityMatrix(originalMatrix);

		// System.out.println("The new matrix is: \r\n" + originalMatrix2);
		// System.out.println("The accuracy is: " + tempMeans.computePurity());

		// new
		// RandomWalk().randomWalk("D:/workspace/randomwalk/data/example21.arff",
		// 1, 3);
		RandomWalk tempWalk = new RandomWalk("D:/Java/randomwalk/data/example21.arff");
	
		// RandomWalk tempWalk = new RandomWalk("D:/Java/MultiLabel/Result/scene/scene_attribute0_Graph.arff");
		tempWalk.randomWalkAdaptive( 4, 3);
	}// Of main
}// Of class RandomWalk
