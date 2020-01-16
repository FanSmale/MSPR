package mspr;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.Date;
import java.util.logging.Level;
import java.util.logging.Logger;

import coser.common.SimpleTool;
import coser.datamodel.decisionsystem.RoughDecisionSystem;
import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import mulan.examples.CrossValidationExperiment;
import randomwalk.randomwalk.RandomWalk;
import weka.core.Instances;
import weka.core.Utils;

/**
 * Summary: Nominal decision system with attribute value taxonomy. Implement the
 * following algorithms:<br>
 * 1) built undirected weighted graph;<br>
 * 2) Clustering the attribute value using random walk algorithm;<br>
 * <p>
 * Author: <b>Liu ying Wen</b> wenliuying1983@163.com
 * <p>
 * Copyright: The source code and all documents are open and free. PLEASE keep
 * this header while revising the program. <br>
 * Organization: SMALE, Southwest Petroleum University, Chengdu 610500,
 * China.<br>
 * Project: The multi-label learning project.
 * <p>
 * Progress: half way.<br>
 * Written time: April 10, 2018. <br>
 * Last modify time: April 10, 2018.
 */
public class RandomwalkSymbolicValuePartition
{

	/**
	 * original data set.
	 */
	MultiLabelInstances originalMldataset;
	/**
	 * undirected weighted graph construct for all attribute.
	 */
	private double[][][] undirectedWeightedGraph;

	/**
	 * clustering results.
	 */
	private int[][] clusteringResult;

	/**
	 * attribute values selection status.
	 */
	private boolean[][] selectStatus;


	public RandomwalkSymbolicValuePartition(String paraFilename, String ParaXmlFilename)
	{
		try
		{
			originalMldataset = new MultiLabelInstances(paraFilename, ParaXmlFilename);
		}
		catch (InvalidDataFormatException e)
		{
			e.printStackTrace();
		}
	}// Of the first constructor

	/**
	 ********************************** 
	 * construct undirected weighted graph
	 * 
	 * @param paraMldataset The original mldataset that need to compute
	 * @param paraDataSet   The instances that need to compute
	 * @param paraAttribute The index of the given attribute.
	 * @param paraLabel     The index of the given label.
	 * @author Liuying Wen 2018/4/11
	 ********************************** 
	 */
	public void coumputeGraph(MultiLabelInstances paraMldataset, Instances paraDataSet, int paraAttribute,
			int paraLabel)
	{

		int numOfInstance = paraMldataset.getNumInstances();

		int countZeroNum = 0;
		for (int i = 0; i < numOfInstance; i++)
		{
			if (paraDataSet.instance(i).value(paraLabel) == 0.0)
				countZeroNum++;
		} // end of for i

		int[] tempCurrentDataZero = new int[countZeroNum];
		int[] tempCurrentDataOne = new int[numOfInstance - countZeroNum];

		int countZero = 0;
		int countOne = 0;
		// Step 1. split data set
		for (int i = 0; i < numOfInstance; i++)
		{
			if ((int) (paraDataSet.instance(i).value(paraLabel)) == 0)
			{
				tempCurrentDataZero[countZero] = (int) (paraDataSet.instance(i).value(paraAttribute));
				countZero++;
			}
			else
			{
				tempCurrentDataOne[countOne] = (int) (paraDataSet.instance(i).value(paraAttribute));
				countOne++;
			} // end of else
		} // end of for i

		coumputeWeightLightByProbability(paraDataSet, paraAttribute, tempCurrentDataOne);
		
	}// end of coumputeGraph

	/**
	 ********************************** 
	 * calculate the weight between nodes
	 * 
	 * @param paraDataSet   The original dataset.
	 * @param paraAttribute The attribute need to be processed.
	 * @param paraZero      The set of label value 0.
	 * @param paraOne       The set of label value 1.
	 * @author Liuying Wen 2018/4/11
	 ********************************** 
	 */
	public void coumputeWeight(Instances paraDataSet, int paraAttribute, int[] paraZero, int[] paraOne)
	{

		int numOfAttValue = paraDataSet.attribute(paraAttribute).numValues();

		int[] oneAttibuteValue = SimpleTool.compressAndSortIntArray(paraOne, paraOne.length);
		
		// Step 3. calculate the weight between nodes whose label value is equal 1.
		for (int i = 0; i < oneAttibuteValue.length; i++)
		{
			for (int j = oneAttibuteValue.length - 1; j >= 0; j--)
			{
				int index = 0;
				if (j == i)
				{
					continue;
				}

				if (oneAttibuteValue[j] < oneAttibuteValue[i])
					index = (numOfAttValue - 1) * oneAttibuteValue[i] + oneAttibuteValue[j];
				else
					index = (numOfAttValue - 1) * oneAttibuteValue[i] + oneAttibuteValue[j] - 1;
				
				undirectedWeightedGraph[paraAttribute][index][0] = oneAttibuteValue[i];
				undirectedWeightedGraph[paraAttribute][index][1] = oneAttibuteValue[j];
				undirectedWeightedGraph[paraAttribute][index][2]++;

			} // end of for j
		} // end of for i
	}// end of coumputeWeight

	/**
	 ********************************** 
	 * average value of a array.
	 * 
	 * @param paraArray array.
	 * @return average value.
	 * @author Liuying Wen 2018/6/19
	 ********************************** 
	 */
	public int computeAverageCountNumber(int[] paraArray)
	{
		int result;
		int total = 0;
		for (int i = 0; i < paraArray.length; i++)
		{
			total += paraArray[i];
		}
		result = total / paraArray.length;
		return result;
	}// end of computeAverageCountNumber

	/**
	 ********************************** 
	 * calculate the weight between nodes(select nodes whose counts is greater than a threshold==average value, and calculate)
	 * 
	 * @param paraDataSet   The original dataset.
	 * @param paraAttribute The attribute need to be processed.
	 * @param paraOne       The set of label value 1.
	 * @author Liuying Wen 2018/6/19
	 ********************************** 
	 */
	public void coumputeWeightLight(Instances paraDataSet, int paraAttribute, int[] paraOne)
	{

		int numOfAttValue = paraDataSet.attribute(paraAttribute).numValues();
		int[] tempOneValue = SimpleTool.copyIntArray(paraOne);
		// Step 1. compressing and sorting clusters.
		int[] oneAttibuteValue = SimpleTool.compressAndSortIntArray(tempOneValue, tempOneValue.length);
		// Step 2. get count.
		int[] countNumber = SimpleTool.countNumerOfEachElement(paraOne, oneAttibuteValue);
		int averageCountNum = computeAverageCountNumber(countNumber);
		int[][] countNumberAfterSorted = SimpleTool.sortIntArray(countNumber);

		SimpleTool.printIntArray(countNumber);

		// Step 3. label attribute values whose count is greater than a threshold==average.
		// boolean[] mark = new boolean[oneAttibuteValue.length];
		if (oneAttibuteValue.length <= 2)
		{
			for (int i = 0; i < oneAttibuteValue.length; i++)
			{
				selectStatus[paraAttribute][oneAttibuteValue[i]] = true;
			} // end of for j
		}
		else
		{
			int count = 0;
			for (int i = 0; i < oneAttibuteValue.length; i++)
			{
				if (countNumber[i] >= averageCountNum)
				{
					selectStatus[paraAttribute][oneAttibuteValue[i]] = true;
					count++;
				} // end of if
			} // end of for j
			if (count <= 1)
			{
				selectStatus[paraAttribute][oneAttibuteValue[countNumberAfterSorted[1][0]]] = true;
				selectStatus[paraAttribute][oneAttibuteValue[countNumberAfterSorted[1][1]]] = true;
			}
		}

		// Step 3. calculate the weight between attribute values whose label is equal 1.
		for (int i = 0; i < oneAttibuteValue.length; i++)
		{
			if (selectStatus[paraAttribute][oneAttibuteValue[i]] == false)
				continue;
			for (int j = oneAttibuteValue.length - 1; j >= 0; j--)
			{
				if (selectStatus[paraAttribute][oneAttibuteValue[j]] == false)
					continue;
				int index = 0;
				if (j == i)
				{
					continue;
				}

				if (oneAttibuteValue[j] < oneAttibuteValue[i])
					index = (numOfAttValue - 1) * oneAttibuteValue[i] + oneAttibuteValue[j];
				else
					index = (numOfAttValue - 1) * oneAttibuteValue[i] + oneAttibuteValue[j] - 1;

				undirectedWeightedGraph[paraAttribute][index][0] = oneAttibuteValue[i];
				undirectedWeightedGraph[paraAttribute][index][1] = oneAttibuteValue[j];
				undirectedWeightedGraph[paraAttribute][index][2]++;

			} // end of for j
		} // end of for i
	}// end of coumputeWeightLight

	/**
	 ********************************** 
	 * calculate the weight between nodes(select nodes whose count is greater than a threshold==average, and calculate)
	 * 
	 * @param paraDataSet   The original dataset.
	 * @param paraAttribute The attribute need to be processed.
	 * @param paraOne       The set of label value 1.
	 * @author Liuying Wen 2018/6/19
	 ********************************** 
	 */
	public void coumputeWeightLightByProbability(Instances paraDataSet, int paraAttribute, int[] paraOne)
	{

		int numOfAttValue = paraDataSet.attribute(paraAttribute).numValues();
		int[] tempOneValue = SimpleTool.copyIntArray(paraOne);
		// Step 1. compressing and sorting clusters.
		int[] oneAttibuteValue = SimpleTool.compressAndSortIntArray(tempOneValue, tempOneValue.length);
		// Step 2. get probabilities of each node, and sort.
		double[] countProbability = SimpleTool.countProbabilityent(paraOne, oneAttibuteValue);
		// int averageCountNum = computeAverageCountNumber(countNumber);
		double[][] countProbabilityAfterSorted = SimpleTool.sortDoubleArray(countProbability);

		if (oneAttibuteValue.length <= 2)
		{
			for (int i = 0; i < oneAttibuteValue.length; i++)
			{
				selectStatus[paraAttribute][oneAttibuteValue[i]] = true;
			} // end of for j

			if (oneAttibuteValue.length == 2)
			{
				for (int i = 0; i < 2; i++)
				{
					for (int j = 1; j >= 0; j--)
					{
						int index = 0;
						if (j == i)
						{
							continue;
						}

						if (oneAttibuteValue[(int) countProbabilityAfterSorted[1][j]] < oneAttibuteValue[(int) countProbabilityAfterSorted[1][i]])
							index = (numOfAttValue - 1) * oneAttibuteValue[(int) countProbabilityAfterSorted[1][i]]
									+ oneAttibuteValue[(int) countProbabilityAfterSorted[1][j]];
						else
							index = (numOfAttValue - 1) * oneAttibuteValue[(int) countProbabilityAfterSorted[1][i]]
									+ oneAttibuteValue[(int) countProbabilityAfterSorted[1][j]] - 1;

						undirectedWeightedGraph[paraAttribute][index][0] = (double) oneAttibuteValue[(int) countProbabilityAfterSorted[1][i]];
						undirectedWeightedGraph[paraAttribute][index][1] = (double) oneAttibuteValue[(int) countProbabilityAfterSorted[1][j]];
						undirectedWeightedGraph[paraAttribute][index][2] += countProbabilityAfterSorted[0][i]
								* countProbabilityAfterSorted[0][j];
					} // end of for j
				} // end of for i
			}
		}
		else
		{
			for (int i = 0; i < 3; i++)
			{
				selectStatus[paraAttribute][oneAttibuteValue[(int) countProbabilityAfterSorted[1][i]]] = true;
			}

			for (int i = 0; i < 3; i++)
			{
				for (int j = 2; j >= 0; j--)
				{
					int index = 0;
					if (j == i)
					{
						continue;
					}

					if (oneAttibuteValue[(int) countProbabilityAfterSorted[1][j]] < oneAttibuteValue[(int) countProbabilityAfterSorted[1][i]])
						index = (numOfAttValue - 1) * oneAttibuteValue[(int) countProbabilityAfterSorted[1][i]]
								+ oneAttibuteValue[(int) countProbabilityAfterSorted[1][j]];
					else
						index = (numOfAttValue - 1) * oneAttibuteValue[(int) countProbabilityAfterSorted[1][i]]
								+ oneAttibuteValue[(int) countProbabilityAfterSorted[1][j]] - 1;

					undirectedWeightedGraph[paraAttribute][index][0] = (double) oneAttibuteValue[(int) countProbabilityAfterSorted[1][i]];
					undirectedWeightedGraph[paraAttribute][index][1] = (double) oneAttibuteValue[(int) countProbabilityAfterSorted[1][j]];
					undirectedWeightedGraph[paraAttribute][index][2] += countProbabilityAfterSorted[0][i]
							* countProbabilityAfterSorted[0][j];
				} // end of for j
			} // end of for i
		}

	}// end of coumputeWeightLightByProbability

	/**
	 ********************************** 
	 * initialize undirected weighted graph of an attribute
	 * 
	 * @param paraAttribute         present attribute.
	 * @param paraAttributeValueNum count of attribute values.
	 * @author Liuying Wen 2018/4/16
	 ********************************** 
	 */
	public void initializeGraph(int paraAttribute, int paraAttributeValueNum)
	{

		for (int i = 0; i < paraAttributeValueNum; i++)
		{
			for (int j = paraAttributeValueNum - 1; j >= 0; j--)
			{
				int index = 0;
				if (j == i)
				{
					continue;
				}

				if (j < i)
					index = (paraAttributeValueNum - 1) * i + j;
				else
					index = (paraAttributeValueNum - 1) * i + j - 1;

				undirectedWeightedGraph[paraAttribute][index][0] = i;
				undirectedWeightedGraph[paraAttribute][index][1] = j;
				undirectedWeightedGraph[paraAttribute][index][2] = 0;

			} // end of for j
		} // end of for i

	}// end of initializeGraph

	/**
	 ********************************** 
	 * construct undirected weighted graph
	 * 
	 * @param paraMldataset The set of multi-label dataset.
	 * @author Liuying Wen 2018/4/19
	 ********************************** 
	 */
	public void constructUndirectedWeightGraph(MultiLabelInstances paraMldataset)
	{

		Instances dataSet = paraMldataset.getDataSet();
		int numOfAttribute = paraMldataset.getFeatureAttributes().size();
		int numOfLabel = paraMldataset.getLabelAttributes().size();
		// int numOfInstance = dataSet.numInstances();

		// initialize undirectedWeightedGraph
		undirectedWeightedGraph = new double[numOfAttribute][][];
		selectStatus = new boolean[numOfAttribute][];

		// iterate each attribute and all labels.
		for (int i = 0; i < numOfAttribute; i++)
		{
			int numOfCurrentAttValue = dataSet.attribute(i).numValues();

			undirectedWeightedGraph[i] = new double[numOfCurrentAttValue * (numOfCurrentAttValue - 1)][3];
			selectStatus[i] = new boolean[numOfCurrentAttValue];
			// Step 1. initialize undirected weighted graph
			initializeGraph(i, numOfCurrentAttValue);

			// Step 2. construct undirected weight graph.
			for (int j = 0; j < numOfLabel; j++)
			{
				coumputeGraph(paraMldataset, dataSet, i, numOfAttribute + j);
			} // end of for j
		} // end of for i

	}// end of constructUndirectedWeightGraph

	/**
	 ********************************** 
	 * store decision system to file according to matrix.
	 * 
	 * @param paramlDataset    original decision system.
	 * @param paraAttribute    present attribute
	 * @param paraMatrixOfData
	 * @return new decision system.
	 ********************************** 
	 */
	public RoughDecisionSystem restoreToArff(MultiLabelInstances paramlDataset, int paraAttribute,
			double[][] paraMatrixOfData) throws Exception
	{

		StringBuffer sb = new StringBuffer();
		String filename = paramlDataset.getDataSet().relationName() + "_attribute" + paraAttribute + "_Graph.arff";

		paramlDataset.getDataSet();
		sb.append(Instances.ARFF_RELATION + " "
				+ Utils.quote(paramlDataset.getDataSet().relationName() + "_attribute" + paraAttribute + "_Graph")
				+ "\r\n");
		sb.append("@attribute  i");
		sb.append(" numeric\r\n");

		sb.append("@attribute  j");
		sb.append(" numeric\r\n");

		sb.append("@attribute  w");
		sb.append(" numeric\r\n");

		paramlDataset.getDataSet();
		// @DATA
		sb.append("\r\n" + Instances.ARFF_DATA + "\r\n");

		for (int i = 0; i < paraMatrixOfData.length; i++)
		{
			sb.append(paraMatrixOfData[i][0]);
			for (int j = 1; j < paraMatrixOfData[i].length; j++)
			{
				sb.append(", ").append(paraMatrixOfData[i][j]);
			} // Of for j
			sb.append("\r\n");
		} // Of for i

		try (DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(
				new FileOutputStream("./Result/" + paramlDataset.getDataSet().relationName() + "/" + filename))))
		{
			dos.writeBytes(sb.toString());
		}
		catch (IOException e)
		{
			e.printStackTrace();
		} // Of try

		RoughDecisionSystem currentDs = null;

		// Read the arff file and return.
		try
		{
			FileReader fileReader = new FileReader(
					"./Result/" + paramlDataset.getDataSet().relationName() + "/" + filename);
			currentDs = new RoughDecisionSystem(fileReader);
			currentDs.setArffFilename("./Result/" + paramlDataset.getDataSet().relationName() + "/" + filename);
			fileReader.close();
		}
		catch (IOException e)
		{
			e.printStackTrace();
		} // Of try
		return currentDs;

	}// Of restoreToArff

	/**
	 ********************************** 
	 * store decision system to file according to matrix.
	 * 
	 * @param paraOriginalData original decision system.
	 * @param paraNewData      decision information after updated.
	 ********************************** 
	 */
	public void restoreToArff(MultiLabelInstances paraOriginalData, Instances paraNewData) throws Exception
	{

		StringBuffer sb = new StringBuffer();
		String filename = ".\\Result\\" + paraOriginalData.getDataSet().relationName() + "_newData.arff";

		paraOriginalData.getDataSet();
		sb.append(Instances.ARFF_RELATION + " " + Utils.quote(paraOriginalData.getDataSet().relationName() + "_newData")
				+ "\r\n");

		Instances originalData = paraOriginalData.getDataSet();
		int numOfAttribute = paraOriginalData.getFeatureAttributes().size();
		int numOfLabel = paraOriginalData.getLabelAttributes().size();
		int numOfInstance = originalData.numInstances();

		for (int i = 0; i < numOfAttribute; i++)
		{
			sb.append(paraNewData.attribute(i));
			sb.append("\r\n");
		} // end of for i

		for (int i = 0; i < numOfLabel; i++)
		{
			sb.append(paraNewData.attribute(numOfAttribute + i));
			sb.append("\r\n");
		} // end of for i

		paraOriginalData.getDataSet();
		// @DATA
		sb.append("\r\n" + Instances.ARFF_DATA + "\r\n");

		/* set file path */

		File file = new File(filename);
		try
		{
			file.createNewFile();

			/* write the result to the file */
			FileWriter fileWriter;
			fileWriter = new FileWriter(file);
			fileWriter.write(sb.toString());
			sb = null;
			fileWriter.close();

		}
		catch (IOException e)
		{

			e.printStackTrace();
		}

		// Store instances data.
		for (int i = 0; i < numOfInstance; i++)
		{

			sb = new StringBuffer();
			sb.append(paraNewData.instance(i));
			sb.append("\r\n");

			try
			{
				FileWriter fileWriter = new FileWriter(new File(filename), true);
				fileWriter.write(sb.toString());
				sb = null;
				fileWriter.close();
			}
			catch (IOException e)
			{

				e.printStackTrace();
			}

		} // Of for i

	}// Of normalize

	/**
	 ********************************** 
	 * clustering attribute values through random walks algorithm 
	 * 
	 * @param paramlDataset multi-label data set
	 * @param paraGraph     undirected weighted graph
	 * @author Liu ying Wen 2018/4/20
	 * @throws Exception
	 ********************************** 
	 */
	public void clusteringByRandomWalk(MultiLabelInstances paramlDataset, double[][][] paraGraph, int paraNumRounds,
			int paraK, double paraCutThreshold) throws Exception
	{

		int numOfAttribute = paramlDataset.getFeatureAttributes().size();
		clusteringResult = new int[numOfAttribute][];

		for (int i = 0; i < paraGraph.length; i++)
		{
			RoughDecisionSystem currentSystem = restoreToArff(paramlDataset, i, paraGraph[i]);

			RandomWalk tempWalk = new RandomWalk(currentSystem.getArffFilename());
			clusteringResult[i] = tempWalk.randomWalkMy(paraNumRounds, paraK, paraCutThreshold);
		} // end of for i
	}// end of clusteringByRandomWalk

	/**
	 ********************************** 
	 * clustering attribute values through random walks algorithm 
	 * 
	 * @param paramlDataset multi-label data set
	 * @param paraGraph     undirected weighted graph
	 * @author Liuying Wen 2018/4/20
	 * @throws Exception
	 ********************************** 
	 */
	public void clusteringByRandomWalk(MultiLabelInstances paramlDataset, double[][][] paraGraph, int paraNumRounds,
			int paraK) throws Exception
	{

		int numOfAttribute = paramlDataset.getFeatureAttributes().size();
		clusteringResult = new int[numOfAttribute][];

		for (int i = 0; i < paraGraph.length; i++)
		{
			RoughDecisionSystem currentSystem = restoreToArff(paramlDataset, i, paraGraph[i]);
			RandomWalk tempWalk = new RandomWalk(currentSystem.getArffFilename());
			clusteringResult[i] = tempWalk.randomWalkAdaptive(paraNumRounds, paraK);
		} // end of for i
	}// end of clusteringByRandomWalk

	/**
	 ********************************** 
	 * update decision table according attribute values clustering results.
	 * 
	 * @param paraOriginalMlDataset original decision system.
	 * @param paraClusteringResult  selection status of breakpoint set
	 * @return newDataset new decision system.
	 ********************************** 
	 */
	public Instances updateDS(MultiLabelInstances paraOriginalMlDataset, int[][] paraClusteringResult)
	{

		Instances originalDataSet = paraOriginalMlDataset.getDataSet();
		Instances newDataset = originalDataSet;

		int numOfAttribute = paraOriginalMlDataset.getFeatureAttributes().size();
		int numOfInstance = originalDataSet.numInstances();
		int rawData = 0;
		int newData = 0;

		int[][][] selectAttributeValueSet = new int[numOfAttribute][][];
		int index = 0;
		for (int i = 0; i < numOfAttribute; i++)
		{
			index = 0;
			selectAttributeValueSet[i] = new int[paraClusteringResult[i].length][2];
			int numOfCurrentAttValue = originalDataSet.attribute(i).numValues();
			for (int j = 0; j < numOfCurrentAttValue; j++)
			{

				if (selectStatus[i][j] == true)
				{
					selectAttributeValueSet[i][index][0] = index;
					selectAttributeValueSet[i][index][1] = j;
					index++;
				} // end of if
			} // end of for j
		} // end of for i

		for (int i = 0; i < numOfAttribute; i++)
		{
			System.out.println("attribute" + i + "attribute values selection");
			SimpleTool.printIntMatrix(selectAttributeValueSet[i]);
		}

		// update decision system
		for (int i = 0; i < numOfAttribute; i++)
		{
			for (int j = 0; j < numOfInstance; j++)
			{
				rawData = (int) originalDataSet.instance(j).value(i);

				if (selectStatus[i][rawData] == false)
				{
					newData = paraClusteringResult[i].length;
				}
				else
				{
					for (int k = 0; k < paraClusteringResult[i].length; k++)
					{
						if (selectAttributeValueSet[i][k][1] != rawData)
							continue;
						else
						{
							newData = paraClusteringResult[i][k] - 1;
							break;
						}

					} // end of for k
				} // end of else
				newDataset.instance(j).setValue(i, newData);
			} // end of for j
		} // end of for i
		
		return newDataset;
	}// end of updateDS

	/**
	 ********************************** 
	 * store calculation results to file.
	 * 
	 * @param paraData      the precision value to store
	 * @param paraStorePath file path
	 * @author Liu-Ying Wen 2014/12/8
	 ********************************** 
	 */
	public void toStoreDataInFile(double paraData, String paraStorePath)
	{
		byte[] buff = new byte[] {};
		try
		{
			String tempWriteString = " ";

			tempWriteString = Double.toString(paraData);
			tempWriteString += "\r\n";
			buff = tempWriteString.getBytes();
			FileOutputStream out = new FileOutputStream(paraStorePath, true);
			out.write(buff, 0, buff.length);
			out.close();
		}
		catch (FileNotFoundException e)
		{
			e.printStackTrace();
		}
		catch (IOException e)
		{
			e.printStackTrace();
		} // Of try
	}// Of toStoreAccuracyInFile

	/**
	 ********************************** 
	 * compressed matrix
	 * 
	 * @param paraOriginalMatrix original matrix
	 * @return compressed matrix
	 * @author Liu-Ying Wen 2014/12/8
	 ********************************** 
	 */
	public double[][][] compressMatrix(double[][][] paraOriginalMatrix)
	{

		double[][][] tempResult = new double[paraOriginalMatrix.length][][];
		int[] countColumnNum = new int[paraOriginalMatrix.length];

		int column = 0;
		for (int i = 0; i < paraOriginalMatrix.length; i++)
		{
			tempResult[i] = new double[paraOriginalMatrix[i].length][3];
			for (int j = 0; j < paraOriginalMatrix[i].length; j++)
			{
				if (paraOriginalMatrix[i][j][2] != 0)
				{

					tempResult[i][column][0] = paraOriginalMatrix[i][j][0];
					tempResult[i][column][1] = paraOriginalMatrix[i][j][1];
					tempResult[i][column][2] = paraOriginalMatrix[i][j][2];
					column++;
				}
				else
				{
					continue;
				}
			} // end of for j
			countColumnNum[i] = column;
			column = 0;
		} // end of for i

		double[][][] result = new double[paraOriginalMatrix.length][][];

		for (int i = 0; i < tempResult.length; i++)
		{
			result[i] = new double[countColumnNum[i]][3];
			for (int j = 0; j < countColumnNum[i]; j++)
			{
				result[i][j][0] = tempResult[i][j][0];
				result[i][j][1] = tempResult[i][j][1];
				result[i][j][2] = tempResult[i][j][2];
			} // end of for j
		} // end of for i

		return result;

	}// end of compressMatrix

	/**
	 ********************************** 
	 * the start function based on the random walk of the attribute values.
	 * 
	 * @author Liu-Ying Wen 2018/5/10
	 * @throws Exception
	 ********************************** 
	 */
	public void startRSVP(int paraNumRounds, int paraK, double paraCutThreshold) throws Exception
	{

		long startTime, startTime1;
		long endTime;

		startTime = new Date().getTime();

		// Step 1. construct undirected weighted graph
		constructUndirectedWeightGraph(originalMldataset);

		// set output file path
		File test = new File("./Result/" + originalMldataset.getDataSet().relationName() + "/Output.txt");
		PrintStream out = new PrintStream(new FileOutputStream(test));
		System.setOut(out);

		endTime = new Date().getTime();
		long constructRunningTime = endTime - startTime;
		toStoreDataInFile(constructRunningTime,
				"./Result/" + originalMldataset.getDataSet().relationName() + "/constructTime.txt");

		startTime1 = new Date().getTime();

		double[][][] newComprressedGraph = compressMatrix(undirectedWeightedGraph);
		double[][][] newIndex = computeNewIndex(originalMldataset, newComprressedGraph);
		double[][][] newUndirectedGraph = computeNewUndirectedGrahp(newComprressedGraph, newIndex);

		// Step 2. random walks for clustering attribute values.
		clusteringByRandomWalk(originalMldataset, newUndirectedGraph, paraNumRounds, paraK, paraCutThreshold);

		endTime = new Date().getTime();
		long clusterRunningTime = endTime - startTime1;
		toStoreDataInFile(clusterRunningTime,
				"./Result/" + originalMldataset.getDataSet().relationName() + "/clusterTime.txt");

		System.out.println("--------------clustering result:--------------");
		SimpleTool.printIntMatrix(clusteringResult);

		long totalRunningTime = endTime - startTime;
		toStoreDataInFile(totalRunningTime,
				"./Result/" + originalMldataset.getDataSet().relationName() + "/totalTime.txt");

		System.out.println("--------------update decision table--------------");
		// Step 3. update decision table
		Instances newData = updateDS(originalMldataset, clusteringResult);

		// Step 4. store new decision table
		restoreToArff(originalMldataset, newData);
		// SimpleTool.printIntMatrix(clusteringResult);
	}// end of startRSVP

	/**
	 ********************************** 
	 * the start function based on the random walk partition of the attribute values.
	 * 
	 * @author Liu-Ying Wen 2018/5/10
	 * @throws Exception
	 ********************************** 
	 */
	public void startRSVPWithoutCutThreshold(int paraNumRounds, int paraK) throws Exception
	{

		long startTime, startTime1;
		long endTime;

		startTime = new Date().getTime();

		// Step 1. construct undirected weighted graph
		constructUndirectedWeightGraph(originalMldataset);

		// set output file path
		File test = new File("./Result/" + originalMldataset.getDataSet().relationName() + "/Output.txt");
		PrintStream out = new PrintStream(new FileOutputStream(test));
		System.setOut(out);

		endTime = new Date().getTime();
		long constructRunningTime = endTime - startTime;
		toStoreDataInFile(constructRunningTime,
				"./Result/" + originalMldataset.getDataSet().relationName() + "/constructTime.txt");

		startTime1 = new Date().getTime();

		double[][][] newComprressedGraph = compressMatrix(undirectedWeightedGraph);
		double[][][] newIndex = computeNewIndex(originalMldataset, newComprressedGraph);
		double[][][] newUndirectedGraph = computeNewUndirectedGrahp(newComprressedGraph, newIndex);

		// Step 2. random walks for clustering attribute values.
		clusteringByRandomWalk(originalMldataset, newUndirectedGraph, paraNumRounds, paraK);

		endTime = new Date().getTime();
		long clusterRunningTime = endTime - startTime1;
		toStoreDataInFile(clusterRunningTime,
				"./Result/" + originalMldataset.getDataSet().relationName() + "/clusterTime.txt");

		System.out.println("--------------clustering results:--------------");
		SimpleTool.printIntMatrix(clusteringResult);

		long totalRunningTime = endTime - startTime;
		toStoreDataInFile(totalRunningTime,
				"./Result/" + originalMldataset.getDataSet().relationName() + "/totalTime.txt");

		System.out.println("--------------update decision table--------------");
		// Step 3. update decision table
		Instances newData = updateDS(originalMldataset, clusteringResult);

		// Step 4. store new decision table.
		restoreToArff(originalMldataset, newData);
	}// end of startRSVP

	/**
	 ********************************** 
	 * construct new undirected weighted graph.
	 * 
	 * @param paraOriginalGrahph original undirected weighted graph.
	 * @param paraNewIndex       index array of new and old attribute values
	 * @return new undirected weighted graph.
	 * @author Liu-Ying Wen 2018/6/27
	 ********************************** 
	 */
	public double[][][] computeNewUndirectedGrahp(double[][][] paraOriginalGraph, double[][][] paraNewIndex)
	{
		double[][][] newUndirectedGraph = new double[paraOriginalGraph.length][][];

		for (int i = 0; i < paraOriginalGraph.length; i++)
		{
			newUndirectedGraph[i] = new double[paraOriginalGraph[i].length][3];
			for (int j = 0; j < paraOriginalGraph[i].length; j++)
			{
				newUndirectedGraph[i][j][2] = paraOriginalGraph[i][j][2];
				for (int k = 0; k < paraNewIndex[i].length; k++)
				{
					if (paraNewIndex[i][k][1] == paraOriginalGraph[i][j][0])
					{
						newUndirectedGraph[i][j][0] = k;
					}
					if (paraNewIndex[i][k][1] == paraOriginalGraph[i][j][1])
					{
						newUndirectedGraph[i][j][1] = k;
					}
				} // end of for k
			} // end of for j
		} // end of for i

		return newUndirectedGraph;

	}// end of computeNewUndirectedGrahp

	/**
	 ********************************** 
	 * calculate index of new attribute index
	 * 
	 * @param paramlDataset original data set.
	 * @param paraGraph     attribute undirected weighted graph.
	 * @return index array of new and old attribute values.
	 * @author Liu-Ying Wen 2018/6/27
	 ********************************** 
	 */
	public double[][][] computeNewIndex(MultiLabelInstances paramlDataset, double[][][] paraGraph)
	{
		int numOfAttribute = paraGraph.length;
		Instances originalDataSet = paramlDataset.getDataSet();

		double[][][] selectAttributeValueSet = new double[numOfAttribute][][];
		int index = 0;
		for (int i = 0; i < numOfAttribute; i++)
		{
			int count = 0;
			for (int j = 0; j < selectStatus[i].length; j++)
			{
				if (selectStatus[i][j] == true)
					count++;
			} // end of for j
			index = 0;
			selectAttributeValueSet[i] = new double[count][2];
			int numOfCurrentAttValue = originalDataSet.attribute(i).numValues();

			for (int j = 0; j < numOfCurrentAttValue; j++)
			{
				if (selectStatus[i][j] == true)
				{
					selectAttributeValueSet[i][index][0] = index;
					selectAttributeValueSet[i][index][1] = j;
					index++;
				} // end of if
			} // end of for j
		} // end of for i
		
		return selectAttributeValueSet;
	}// end of computeNewIndex

	/**
	 * Get rank number.
	 */
	public static int rankOfAttribute(MultiLabelInstances multiLabelInstances, int indexOfAttribute)
	{
		int rankValue = 0;

		Instances instances = multiLabelInstances.getDataSet();

		int numberOfInstances = instances.numInstances();

		String[] allValueOfAttribute = new String[numberOfInstances];

		int index = 0;

		for (int i = 0; i < numberOfInstances; ++i)
		{
			String[] strings = instances.instance(i).toString().split(",");
			allValueOfAttribute[index] = strings[indexOfAttribute];
			++index;
		}

		Arrays.sort(allValueOfAttribute);

		int numberOfDistinctValue = 0;
		String lastString = null;

		for (int i = 0; i < allValueOfAttribute.length; ++i)
		{
			if (numberOfDistinctValue == 0)
			{
				lastString = allValueOfAttribute[i];
				++numberOfDistinctValue;
			}
			else if (allValueOfAttribute[i].compareTo(lastString) > 0)
			{
				lastString = null;
				lastString = allValueOfAttribute[i];
				++numberOfDistinctValue;
			}
			else
			{
				continue;
			}
		}

		rankValue = numberOfDistinctValue;

		return rankValue;
	}

	// all rank of attribute value.
	public static int allRankOfAttribute(MultiLabelInstances multiLabelInstances)
	{
		int sumOfRank = 0;

		int numberOfAttribute = multiLabelInstances.getFeatureAttributes().size();

		for (int i = 0; i < numberOfAttribute; ++i)
		{
			sumOfRank += rankOfAttribute(multiLabelInstances, i);
		}

		return sumOfRank;
	}

	// the average rank
	public static double averageRank(int oldSumOfRank, int newSumOfRank)
	{
		double average = 0;
		average = (double) newSumOfRank / oldSumOfRank;
		return average;
	}

	/**
	 * Main function, the only entry of MSPR method.
	 * @param args console arguments
	 */
	public static void main(String[] args)
	{
		try
		{
			/**
			 * data set name
			 */
			String datasetName = "flags";
			
			/**
			 * Where the data set is.
			 */
			String arffFilename = ".\\data\\ew\\" + datasetName + ".arff";
			String xmlFilename = ".\\data\\xml\\" + datasetName + ".xml";
			
			/**
			 * Where the new decision table is.
			 */
			String newArffFilename = ".\\Result\\" + datasetName + "_newData.arff";

			/**
			 * Get start time.
			 */
			long startTime = System.currentTimeMillis();

			RandomwalkSymbolicValuePartition mlSystem = new RandomwalkSymbolicValuePartition(arffFilename, xmlFilename);

			/**
			 * Get original rank number.
			 */
			int oldSumOfRank = allRankOfAttribute(mlSystem.originalMldataset);

			/**
			 * parameter of random walks, random walks rounds and random walks k.
			 */
			mlSystem.startRSVPWithoutCutThreshold(3, 3);

			
			/**
			 * Get new rank number.
			 */
			int newSumOfRank = allRankOfAttribute(new MultiLabelInstances(newArffFilename, xmlFilename));
			
			/**
			 * The output information is in ".\Result\$DatasetName$\Output.txt" file.
			 */
			System.out.println("old sum of rank = " + oldSumOfRank);
			System.out.println("new sum of rank = " + newSumOfRank);
			System.out.println("average percent of rank compressed = " + averageRank(oldSumOfRank, newSumOfRank));
			
			/**
			 * Get end time.
			 */
			long endTime = System.currentTimeMillis();
			
			System.out.println("run time = " + (endTime - startTime));

		}
		catch (Exception ex)
		{
			Logger.getLogger(CrossValidationExperiment.class.getName()).log(Level.SEVERE, null, ex);
		}
	}

}
