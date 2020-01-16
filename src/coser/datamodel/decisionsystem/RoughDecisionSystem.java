package coser.datamodel.decisionsystem;

import java.io.*;
import java.text.*;
import java.util.Date;

import weka.core.*;
import coser.common.*;

/**
 * Decision systems for rough sets however without algorithm implementation. The
 * following data are claimed:<br>
 * 1) basic information of the decision system;<br>
 * 2) basic information for nominal and numeric decision systems;<br>
 * 3) reduct related information; and<br>
 * 4) subreduct related information.<br>
 * No algorithm is implemented. Some methods are essentially abstract, so does
 * this class.
 * <p>
 * Author: <b>Fan Min</b> minfanphd@163.com
 * <p>
 * Copyright: The source code and all documents are open and free. PLEASE keep
 * this header while revising the program. <br>
 * Organization: <a href=http://grc.fjzs.edu.cn/>Lab of Granular Computing</a>,
 * Zhangzhou Normal University, Fujian 363000, China.<br>
 * Project: The cost-sensitive rough sets project.
 * <p>
 * Progress: Done. New methods may be added in the future.<br>
 * Written time: August 12, 2011. <br>
 * Last modify time: February 15, 2012.
 */
public class RoughDecisionSystem extends Instances {

	/**
	 * For serialization.
	 */
	private static final long serialVersionUID = 9142315159496665406L;

	// ////Part 1, basic information of the decision system//////
	/**
	 * The file for the basic decision system.
	 */
	protected String arffFilename;

	/**
	 * Number of conditional attributes. This variable is used to avoid invoking
	 * numAttributes() - 1, which might be time-consuming.
	 */
	public  int numberOfConditions;

	/**
	 * Number of instances. This variable is used to avoid invoking
	 * numInstances(), which might be time-consuming.
	 */
	protected int numberOfInstances;

	/**
	 * The number of values of the class attribute.
	 */
	protected int classSize;

	/**
	 * Is it nominal? It is true only if all attributes are nominal.
	 */
	private boolean isItNominal;

	/**
	 * Is it numeric? It is true only if all attributes except the decision
	 * class are numeric.
	 */
	private boolean isItNumeric;

	/**
	 * Has it missing values?
	 */
	private boolean hasItMissingValue;

	/**
	 * Is it normalized?
	 */
	private boolean isItNormalized;

	/**
	 * The set of all conditional attributes. An array with only true values. It
	 * is used to save time while computing.
	 */
	protected boolean[] allConditions;

	// ////Part 2, basic information for purely nominal and numeric decision
	// systems//////
	/**
	 * Evaluate measure for attribute significance or the whole decision system.
	 * It can be POSITIVE_REGION, CONDITIONAL_ENTROPY, or MAJORITY. It is set to
	 * private to avoid unauthorized access. It can be accessed only through
	 * getMeasure() and setMeasure().
	 */
	private int measure;

	/**
	 * Positive region based, for attribute reduct definition.
	 */
	public static final int POSITIVE_REGION = 0;

	/**
	 * Conditional entropy based, for attribute reduct definition.
	 */
	public static final int CONDITIONAL_ENTROPY = 1;

	/**
	 * Majority based, for attribute reduct definition.
	 */
	public static final int MAJORITY = 2;

	/**
	 * The consistency computed for the current measure. It is comparable to
	 * totalEntropy, totalPositiveRegion and totalMajority.
	 */
	protected int consistency;

	/**
	 * The total entropy of the decision system;
	 */
	protected double totalEntropy;

	/**
	 * The total positive region of the decision system;
	 */
	protected int totalPositiveRegion;

	/**
	 * The total majority of the decision system;
	 */
	protected int totalMajority;

	/**
	 * Used to compute entropy reciprocal.
	 */
	private static int ENTROPY_RECIPROCAL_CONSTANT = 1000;

	// ////Part 3, reduct related information//////
	/**
	 * The file for all reducts.
	 */
	protected String allReductsFilename;

	/**
	 * The set of all reducts. This class only support reading reducts from a
	 * file. The computation of reducts is not supported because we do not know
	 * the characteristics of the decision system (numeric, nominal, etc).
	 */
	protected boolean[][] allReducts;

	/**
	 * The current reduct represented by an boolean array. It is more useful in
	 * subclasses.
	 */
	protected static boolean[] currentReduct;

	/**
	 * Currently selected attributes. Used in the process of reduction. Useful
	 * in subclasses.
	 */
	protected boolean[] currentSelectedAttributes;

	/**
	 * The array that indicate the set of core attributes. Useful in subclasses.
	 */
	protected boolean[] core;

	/**
	 * The size of the minimal reduct.
	 */
	protected int minimalReductSize;

	/**
	 * The optimal reduct. More useful in subclasses.
	 */
	protected boolean[] optimalReduct;

	/**
	 * The number of candidate reducts.
	 */
	protected int numberOfCandidateReducts;

	/**
	 * Candidate reduct array, useful in the process of reduction.
	 */
	protected long[] candidateReductArray;

	// ////Part 4, subreduct related information//////
	/**
	 * The file for all subreducts. Now the concept of subreduct is only
	 * supported by nominal decision systems. In the further, however, we may
	 * move it to general decision systems. In that case, the information
	 * entropy is not applicable.
	 */
	protected String allSubreductsFilename;

	/**
	 * The set of all subreducts. Similar to the case of allReducts, it can be
	 * read, however not computed in this class.
	 */
	protected boolean[][] allSubreducts;

	/**
	 * Number of candidate sub-reducts
	 */
	protected int numberOfCandidateSubreducts;

	/**
	 * Candidate sub-reduct array
	 */
	protected long[] candidateSubreductArray;

	/**
	 * Candidate sub-reduct consistency array
	 */
	protected int[] candidateSubreductConsistencyArray;

	// ////Part 5, algorithm run time information//////
	/**
	 * Reduction start time, only useful in subclasses.
	 */
	protected long startTime;

	/**
	 * Reduction end time, only useful in subclasses.
	 */
	protected long endTime;

	/**
	 * Backtrack steps, only useful in subclasses.
	 */
	protected long backtrackSteps;

	/**
	 * Runtime information, only useful in subclasses.
	 */
	protected String runTimeInformation;

	/**
	 ************************* 
	 * Construct the rough decision system. Simply adopt everything of the super
	 * class. This is the most often invoked constructor.
	 * 
	 * @param paraReader
	 *            the input with the form of a Reader. We can generate a
	 *            FileReader given an .arff filename.
	 ************************* 
	 */
	public RoughDecisionSystem(Reader paraReader) throws IOException {
		super(paraReader);
		initialize();
	}// Of the first constructor

	/**
	 ************************* 
	 * Construct the decision system, essentially clone a decision system.
	 * Simply adopt everything of the super class.
	 * 
	 * @param paraDataset
	 *            the initial dataset with the form of Instance.
	 ************************* 
	 */
	public RoughDecisionSystem(Instances paraDataset) {
		super(paraDataset);
		initialize();
	}// Of the second constructor

	/**
	 ************************* 
	 * Construct the decision system. Simply adopt everything of the super
	 * class.
	 ************************* 
	 */
	public RoughDecisionSystem(Instances paraDataset, int paraCapacity) {
		super(paraDataset, paraCapacity);
		initialize();
	}// Of the third constructor

	/**
	 ************************* 
	 * Construct the decision system. Simply adopt everything of the super
	 * class.
	 ************************* 
	 */
	public RoughDecisionSystem(Instances source, int first, int toCopy) {
		super(source, first, toCopy);
		initialize();
	}// Of the fourth constructor

	/**
	 ************************* 
	 * Construct the decision system. Simply adopt everything of the super
	 * class.
	 ************************* 
	 */
	public RoughDecisionSystem(String name, FastVector attInfo, int capacity) {
		super(name, attInfo, capacity);
		initialize();
	}// Of the fifth constructor

	/**
	 ************************* 
	 * Construct a new sub decision system with selected attributes.
	 * 
	 * @param paraDataset
	 *            the initial dataset with the form of Instance.
	 * @param paraSelectedAttributes
	 *            selected attributes, others are removed.
	 ************************* 
	 */
	public RoughDecisionSystem(Instances paraDataset,
			boolean[] paraSelectedAttributes) {
		super(paraDataset);
		for (int i = paraSelectedAttributes.length - 1; i >= 0; i--) {
			if (!paraSelectedAttributes[i]) {
				deleteAttributeAt(i);
			}// Of if
		}// Of for i
		initialize();
	}// Of the sixth constructor

	/**
	 ************************* 
	 * Initialize to default values. This method is always invoked in
	 * constructors.
	 ************************* 
	 */
	private void initialize() {
		// Part 1
		arffFilename = null;
		numberOfConditions = numAttributes() - 1;
		numberOfInstances = numInstances();
		setClassIndex(numberOfConditions);
		classSize = classAttribute().numValues();
		/*isItNominal = isNominalJudge();
		isItNumeric = isNumericJudge();
		hasItMissingValue = hasMissingValueJudge();
		isItNormalized = isNormalizedJudge();
*/
		isItNominal = false;
		isItNumeric = true;
		hasItMissingValue = false;
		isItNormalized = false;
		
		allConditions = new boolean[numberOfConditions];
		for (int i = 0; i < numberOfConditions; i++) {
			allConditions[i] = true;
		}// Of for i

		// Part 2
		measure = POSITIVE_REGION;
		consistency = Integer.MIN_VALUE + 1;
		totalEntropy = Integer.MIN_VALUE;
		totalPositiveRegion = Integer.MIN_VALUE;
		totalMajority = Integer.MIN_VALUE;

		// Part 3
		allReductsFilename = null;
		allReducts = null;
		currentReduct = null;
		currentSelectedAttributes = null;
		core = null;
		minimalReductSize = Integer.MAX_VALUE;
		optimalReduct = null;
		numberOfCandidateReducts = Integer.MAX_VALUE;
		candidateReductArray = null;

		// Part 4
		allSubreductsFilename = null;
		allSubreducts = null;
		numberOfCandidateSubreducts = Integer.MIN_VALUE;
		candidateSubreductArray = null;
		candidateSubreductConsistencyArray = null;

		// Part 5
		startTime = Integer.MIN_VALUE;
		endTime = Integer.MIN_VALUE;
		backtrackSteps = Integer.MAX_VALUE;
		runTimeInformation = null;
	}// Of initialize

	/**
	 ************************* 
	 * Copy from the given decision system. This method is often invoked in
	 * subclasses.
	 * 
	 * @param paraDecisionSystem
	 *            the decision system that copy from.
	 ************************* 
	 */
	protected void copyForClone(RoughDecisionSystem paraDecisionSystem) {
		// Part 1
		arffFilename = paraDecisionSystem.arffFilename;
		numberOfConditions = paraDecisionSystem.numberOfConditions;
		numberOfInstances = paraDecisionSystem.numberOfInstances;
		setClassIndex(numberOfConditions);
		classSize = paraDecisionSystem.classSize;
		isItNominal = paraDecisionSystem.isItNominal;
		isItNumeric = paraDecisionSystem.isItNumeric;
		hasItMissingValue = paraDecisionSystem.hasItMissingValue;
		isItNormalized = paraDecisionSystem.isItNormalized;
		allConditions = new boolean[numberOfConditions];
		for (int i = 0; i < numberOfConditions; i++) {
			allConditions[i] = paraDecisionSystem.allConditions[i];
		}// Of for i

		// Part 2
		measure = paraDecisionSystem.measure;
		consistency = paraDecisionSystem.consistency;
		totalEntropy = paraDecisionSystem.totalEntropy;
		totalPositiveRegion = paraDecisionSystem.totalPositiveRegion;
		totalMajority = paraDecisionSystem.totalMajority;

		// Part 3
		allReductsFilename = new String(paraDecisionSystem.allReductsFilename);
		allReducts = paraDecisionSystem.allReducts;
		currentReduct = paraDecisionSystem.currentReduct;
		currentSelectedAttributes = paraDecisionSystem.currentSelectedAttributes;
		core = paraDecisionSystem.core;
		minimalReductSize = paraDecisionSystem.minimalReductSize;
		optimalReduct = paraDecisionSystem.optimalReduct;
		numberOfCandidateReducts = paraDecisionSystem.numberOfCandidateReducts;
		candidateReductArray = paraDecisionSystem.candidateReductArray;

		// Part 4
		allSubreductsFilename = paraDecisionSystem.allSubreductsFilename;
		allSubreducts = paraDecisionSystem.allSubreducts;
		numberOfCandidateSubreducts = paraDecisionSystem.numberOfCandidateSubreducts;
		candidateSubreductArray = paraDecisionSystem.candidateSubreductArray;
		candidateSubreductConsistencyArray = paraDecisionSystem.candidateSubreductConsistencyArray;

		// Part 5
		startTime = paraDecisionSystem.startTime;
		endTime = paraDecisionSystem.endTime;
		backtrackSteps = paraDecisionSystem.backtrackSteps;
		runTimeInformation = paraDecisionSystem.runTimeInformation;
	}// Of copyForClone

	// //////////////////////Part 1//////////////////////
	/**
	 ************************* 
	 * Set the arff filename such that it can be stored and reloaded next time.
	 * It is not used to read the file this time. At the same time set the
	 * reduct filename and subreduct filename. They can be constructed from the
	 * arff filename
	 * 
	 * @param paraFilename
	 *            the given filename.
	 ************************* 
	 */
	public void setArffFilename(String paraFilename) {
		arffFilename = paraFilename;
		setAllReductsFilename();
		setAllSubreductsFilename();
	}// Of setArffFilename

	/**
	 ************************* 
	 * Get the arff filename of the current dataset.
	 * 
	 * @return arff filename of the current dataset.
	 * @throws Exception
	 *             if the filename is not assigned.
	 ************************* 
	 */
	public String getArffFilename() throws Exception {
		if (arffFilename == null) {
			throw new Exception(
					"Error occurred in RoughDecisionSystem.getArffFilename().\r\n"
							+ "The arff filename is not assigned yet.");
		}// Of if
		return arffFilename;
	}// Of getArffFilename

	/**
	 ************************* 
	 * Get the number of conditions, only invoked by those who have not access
	 * to numberOfConditions.
	 * 
	 * @return the number of conditions.
	 ************************* 
	 */
	public int getNumberOfConditions() {
		return numberOfConditions;
	}// Of getNumberOfConditions

	/**
	 ************************* 
	 * Is the decision system nominal? Set to private such that it can be
	 * invoked only in initialize() once.
	 ************************* 
	 */
	private boolean isNominalJudge() {
		boolean tempIsNominal = true;
		for (int i = 0; i < numberOfConditions; i++) {
			if (!attribute(i).isNominal()) {
				tempIsNominal = false;
				break;
			}// Of if
		}// Of for

		return tempIsNominal;
	}// Of isNominalJudge

	/**
	 ************************* 
	 * Is the decision system nominal?
	 * 
	 * @return true only if all attributes are nominal.
	 ************************* 
	 */
	public boolean isNominal() {
		return isItNominal;
	}// Of isNominal

	/**
	 ************************* 
	 * Is the decision system numeric? Set to private such that it can be
	 * invoked only in initialize() once.
	 ************************* 
	 */
	private boolean isNumericJudge() {
		boolean tempIsNumeric = true;
		for (int i = 0; i < numberOfConditions; i++) {
			if (!attribute(i).isNumeric()) {
				tempIsNumeric = false;
				break;
			}// Of if
		}// Of for

		return tempIsNumeric;
	}// Of isNumericJudge

	/**
	 ************************* 
	 * Is the decision system numeric?
	 * 
	 * @return true only if all attributes except the decision class are
	 *         numeric.
	 ************************* 
	 */
	public boolean isNumeric() {
		return isItNumeric;
	}// Of isNumeric

	/**
	 ************************* 
	 * Is the decision system numeric? Set to private such that it can be
	 * invoked only in this class and runs only one time.
	 ************************* 
	 */
	private boolean hasMissingValueJudge() {
		hasItMissingValue = false;
		for (int i = 0; i < numberOfInstances; i++) {
			if (instance(i).hasMissingValue()) {
				return true;
			}
		}// Of while
		return hasItMissingValue;
	}// Of hasMissingValueJudge

	/**
	 ************************* 
	 * Has the decision system missing value?
	 * 
	 * @return has missing value or not.
	 ************************* 
	 */
	public boolean hasMissingValue() {
		return hasItMissingValue;
	}// Of hasMissingValue

	/**
	 ************************* 
	 * Is the decision system normalized? Set to private such that it can be
	 * invoked only in this class and runs only one time.
	 * 
	 * @return normalized or not.
	 ************************* 
	 */
	private boolean isNormalizedJudge() {
		// boolean tempIsNormalized = true;
		double tempDouble = 0;
		for (int i = 0; i < numberOfConditions; i++) {
			for (int j = 0; j < numberOfInstances; j++) {
				tempDouble = instance(j).value(i);
				if ((tempDouble < 0) || (tempDouble > 1)) {
					return false;
				}
			}// Of if
		}// Of for

		return true;
	}// Of isNormalizedJudge

	/**
	 ************************* 
	 * Is the decision system normalized?
	 * 
	 * @return true only if all attribute values are in [0, 1].
	 ************************* 
	 */
	public boolean isNormalized() {
		return isItNormalized;
	}// Of isNormalized

	// //////////////////////Part 2//////////////////////
	/**
	 ************************* 
	 * Set the measure for evaluation the consistency of an attribute or a
	 * decision system. At the same time compute the consistency.
	 * 
	 * @param paraMeasure
	 *            the given measure.
	 ************************* 
	 */
	public void setMeasure(int paraMeasure) throws Exception {
		measure = paraMeasure;
		consistency = computeConsistency();
	}// Of setMeasure

	/**
	 ************************* 
	 * Get the measure.
	 * 
	 * @return the measure.
	 ************************* 
	 */
	public int getMeasure() {
		return measure;
	}// Of getMeasure

	/**
	 ************************* 
	 * Compute the attribute set consistency. At the same time set the
	 * consistency variable.
	 * 
	 * @return the attribute set consistency.
	 * @see #computeConsistency(boolean[])
	 ************************* 
	 */
	public int computeConsistency() throws Exception {
		consistency = computeConsistency(allConditions);
		return consistency;
	}// Of computeConsistency

	/**
	 ************************* 
	 * Compute the attribute set consistency.
	 * 
	 * @param paraSelectedAttributesByLong
	 *            the test set represented by an integer.
	 * @return the attribute set consistency.
	 * @throws Exception
	 *             because it might be thrown by
	 *             SimpleTool.longToBooleanArray().
	 * @see #computeConsistency(boolean[])
	 ************************* 
	 */
	public int computeConsistency(long paraSelectedAttributesByLong)
			throws Exception {
		boolean[] selectedAttributes = SimpleTool.longToBooleanArray(
				paraSelectedAttributesByLong, numberOfConditions);
		return computeConsistency(selectedAttributes);
	}// Of consistency

	/**
	 ************************* 
	 * Compute the attribute set consistency.
	 * 
	 * @param paraSelectedAttributes
	 *            selected attributes represented as a boolean array.
	 * @return the attribute set consistency.
	 ************************* 
	 */
	public int computeConsistency(boolean[] paraSelectedAttributes)
			throws Exception {
		int tempConsistency = Integer.MIN_VALUE;
		switch (measure) {
		case POSITIVE_REGION:
			tempConsistency = positiveRegion(paraSelectedAttributes);
			break;
		case CONDITIONAL_ENTROPY:
			tempConsistency = entropyReciprocal(paraSelectedAttributes);
			break;
		case MAJORITY:
			tempConsistency = majority(paraSelectedAttributes);
			break;
		default:
			throw new Exception(
					"Error occurred in RoughDecisionSystem.computeConsistency().\r\n"
							+ "The measure for reduction is not supported.");
		}// Of switch

		return tempConsistency;
	}// Of computeConsistency

	/**
	 ************************* 
	 * Compute the positive region size using all conditional attributes,
	 * essentially an abstract method.
	 * 
	 * @return the number of objects in the positive region.
	 * @see #positiveRegion(boolean[]).
	 ************************* 
	 */
	public int positiveRegion() throws Exception {
		return positiveRegion(allConditions);
	}// Of positiveRegion

	/**
	 ************************* 
	 * Compute the positive region size given the attribute subset, for
	 * overload.
	 * 
	 * @param paraSelectedAttributes
	 *            the attribute subset represented by a long integer.
	 * @return the number of objects in the positive region.
	 * @see #positiveRegion(boolean[])
	 ************************* 
	 */
	public int positiveRegion(long paraSelectedAttributes) throws Exception {
		if (paraSelectedAttributes < 0) {
			throw new Exception(
					"Exception occurred in RoughDecisionSystem.positiveRegion(),\r\n"
							+ "  the long integer specifying the test set should not be negative: "
							+ paraSelectedAttributes + ".");
		}

		boolean[] selectedAttributes = SimpleTool.longToBooleanArray(
				paraSelectedAttributes, numberOfConditions);

		return positiveRegion(selectedAttributes);
	}// Of positiveRegion

	/**
	 ************************* 
	 * Compute the positive region given the attribute subset, should be
	 * overridden.
	 * 
	 * @param paraSelectedAttributes
	 *            selected attributes represented as a boolean array.
	 * @return the number of objects in the positive region.
	 * @throws Exception
	 *             for overridden by subclasses.
	 ************************* 
	 */
	public int positiveRegion(boolean[] paraSelectedAttributes)
			throws Exception {
		return Integer.MIN_VALUE;
	}// Of positiveRegion

	/**
	 ************************* 
	 * Compute the entropy reciprocal. It is defined by
	 * (ENTROPY_RECIPROCAL_CONSTANT / entropy).
	 * 
	 * @return the entropy reciprocal.
	 * @throws Exception
	 *             from conditionalEntropy().
	 ************************* 
	 */
	public int entropyReciprocal() throws Exception {
		return (int) (ENTROPY_RECIPROCAL_CONSTANT / conditionalEntropy());
	}// Of entropyReciprocal

	/**
	 ************************* 
	 * Compute the entropy reciprocal.
	 * 
	 * @param paraSelectedAttributesByLong
	 *            the test set represented by an integer.
	 * @return the entropy reciprocal.
	 * @throws Exception
	 *             from conditionalEntropy(int).
	 ************************* 
	 */
	public int entropyReciprocal(long paraSelectedAttributesByLong)
			throws Exception {
		return (int) (ENTROPY_RECIPROCAL_CONSTANT / conditionalEntropy(paraSelectedAttributesByLong));
	}// Of entropyReciprocal

	/**
	 ************************* 
	 * Compute the entropy reciprocal.
	 * 
	 * @param paraSelectedAttributes
	 *            selected attributes represented as a boolean array.
	 * @return the entropy reciprocal.
	 * @throws Exception
	 *             for overridden by subclasses.
	 ************************* 
	 */
	public int entropyReciprocal(boolean[] paraSelectedAttributes)
			throws Exception {
		return (int) (ENTROPY_RECIPROCAL_CONSTANT / conditionalEntropy(paraSelectedAttributes));
	}// Of entropyReciprocal

	/**
	 ************************* 
	 * Compute the entropy reciprocal.
	 * 
	 * @param paraEntropy
	 *            the given entropy.
	 * @return the entropy reciprocal.
	 * @throws Exception
	 *             for overridden by subclasses.
	 ************************* 
	 */
	public int entropyReciprocal(double paraEntropy) throws Exception {
		if (paraEntropy < 1e-6) {
			return Integer.MAX_VALUE;
		}// Of if

		return (int) (ENTROPY_RECIPROCAL_CONSTANT / paraEntropy);
	}// Of entropyReciprocal

	/**
	 ************************* 
	 * Compute the entropy reciprocal for object subsets.
	 * 
	 * @param paraInstancesArray
	 *            instances array.
	 * @return the entropy reciprocal for object subsets.
	 ************************* 
	 */
	public int entropyReciprocal(Instances[] paraInstancesArray)
			throws Exception {
		double tempEntropy = conditionalEntropy(paraInstancesArray);
		if (tempEntropy < 1e-6) {
			return Integer.MAX_VALUE;
		}// Of if

		return (int) (ENTROPY_RECIPROCAL_CONSTANT / tempEntropy);
	}// Of entropyReciprocal

	/**
	 ************************* 
	 * Computes the conditional entropy of this dataset, for overload.
	 * 
	 * @return the conditional entropy of all attributes.
	 * @see #conditionalEntropy(boolean[])
	 * @throws Exception
	 *             for overridden by subclasses.
	 ************************* 
	 */
	public double conditionalEntropy() throws Exception {
		return conditionalEntropy(allConditions);
	}// Of conditionalEntropy

	/**
	 ************************* 
	 * Compute the conditional entropy given an attribute subset, for overload.
	 * It supports attribute subsets represented by a long.
	 * 
	 * @param paraSelectedAttributesByLong
	 *            the selected attributes represented by a long integer.
	 * @return the conditional entropy of selected attributes.
	 * @see #conditionalEntropy(boolean[])
	 ************************* 
	 */
	public double conditionalEntropy(long paraSelectedAttributesByLong)
			throws Exception {
		if (paraSelectedAttributesByLong < 0) {
			throw new Exception(
					"Exception occurred in RoughDecisionSystem.conditionalEntropy(),\r\n"
							+ "  the long integer specifying the test set should not be negative: "
							+ paraSelectedAttributesByLong + ".");
		}
		boolean[] selectedAttributes = SimpleTool.longToBooleanArray(
				paraSelectedAttributesByLong, numberOfConditions);
		return conditionalEntropy(selectedAttributes);
	}// Of conditionalEntropy

	

	/**
	 ************************* 
	 * Computes the conditional entropy of a dataset.
	 * 
	 * @param paraSelectedAttributes
	 *            the selected attributes.
	 * @return the conditional entropy of selected attributes.
	 * @see #conditionalEntropy(Instances[])
	 ************************* 
	 */
	public double conditionalEntropy(boolean[] paraSelectedAttributes)
			throws Exception {
		// double entropy = 0;
		/*long startTime;
		long endTime;
		String runTimeInformation;
		startTime = new Date().getTime();*/
		Instances[] splitData = splitData(paraSelectedAttributes);
		/*endTime = new Date().getTime();
		runTimeInformation = "Split data花费的时间: "
				+ (endTime - startTime) + " ms.";

		System.out.println(runTimeInformation);*/
		
		
		return conditionalEntropy(splitData);
	}// Of conditionalEntropy
	
	/**
	 ************************* 
	 * Splits a dataset according to the values of a nominal attribute. Copied
	 * from Id3.java. Only for nominal attributes.
	 * 
	 * @param paraData
	 *            the data which is to be split.
	 * @param paraAttribute
	 *            the attribute to be used for splitting.
	 * @return the sets of instances produced by the split.
	 ************************* 
	 */
	public Instances[] splitData(Instances paraData, Attribute paraAttribute) {

		Instances[] splitData = new Instances[paraAttribute.numValues()];
		for (int j = 0; j < paraAttribute.numValues(); j++) {
			splitData[j] = new Instances(paraData, paraData.numInstances());
		}// Of for

		//System.out.println("----Begin Split data ------");
		for (int i = 0; i < paraData.numInstances(); i++) {
			Instance inst = paraData.instance(i);
			// System.out.println(inst.value(paraAttribute));
			splitData[(int) inst.value(paraAttribute)].add(inst);
		}
		//System.out.println("----End Split data ------");
		for (int i = 0; i < splitData.length; i++) {
			splitData[i].compactify();
			//System.out.println(splitData[i]);
		}
		return splitData;
	}// Of splitData
	
	/**
	 ************************* 
	 * Splits this dataset according to the given attribute subset. There is not
	 * a paraData parameter since it is unnecessary.
	 * 
	 * @param paraSelectedAttributes
	 *            selected attributes.
	 * @return the sets of instances produced by the split. Pure Instances are
	 *         not returned, hence when the returned value is null, the given
	 *         attibute subset is enough for classification.
	 ************************* 
	 */
	public Instances[] splitData(boolean[] paraSelectedAttributes)
			throws Exception {

		Instances[] currentInstances = new Instances[1];
		currentInstances[0] = new Instances(this);
		Instances[] newLevelInstances;
		Attribute currentAttribute;

		for (int i = 0; i < paraSelectedAttributes.length; i++) {
			if (!paraSelectedAttributes[i]) {
				continue;// Simply skip this attribute
			}// of if
			currentAttribute = attribute(i);

			Instances[] currentLevelInstances = new Instances[500000];
			int currentLevelInstancesArrayLength = 0;
			for (int j = 0; j < currentInstances.length; j++) {
				if (classEntropy(currentInstances[j]) < 1e-6) {
					// No need to split when the class is pure.
					continue;
				} else {
					newLevelInstances = splitData(currentInstances[j],
							currentAttribute);
					for (int k = 0; k < newLevelInstances.length; k++) {
						if (newLevelInstances[k].numInstances() <= 1)
							continue;
						if (classEntropy(newLevelInstances[k]) < 1e-6) {
							// No need to split when the class is pure.
							continue;
						}// Of if
						currentLevelInstances[currentLevelInstancesArrayLength] = newLevelInstances[k];
						currentLevelInstancesArrayLength++;
					}// Of k
				}// Of if
			}// Of for j

			if (currentLevelInstancesArrayLength == 0) {
				// All object subsets are pure
				return null;
			}// Of if

			// Copy to currentInstances
			currentInstances = new Instances[currentLevelInstancesArrayLength];
			for (int j = 0; j < currentLevelInstancesArrayLength; j++) {
				currentInstances[j] = currentLevelInstances[j];
			}// Of for j
		}// Of for i

		return currentInstances;
	}// Of splitData
	
	/**
	 ************************* 
	 * Computes the conditional entropy of object subsets.
	 * 
	 * @param paraInstancesArray
	 *            object subsets.
	 * @return the conditional entropy of object subsets.
	 ************************* 
	 */
	public double conditionalEntropy(Instances[] paraInstancesArray)
			throws Exception {
		/*long startTime;
		long endTime;
		String runTimeInformation;
		
		startTime = new Date().getTime();*/
		if (paraInstancesArray == null) {
			return 0;
		}// Of if
		double entropy = 0;

		for (int j = 0; j < paraInstancesArray.length; j++) {
			if (paraInstancesArray[j].numInstances() > 0) {
				entropy += ((double) paraInstancesArray[j].numInstances() / numberOfInstances)
						* classEntropy(paraInstancesArray[j]);
			}// Of if
		}// Of for j
		/*endTime = new Date().getTime();
		runTimeInformation = "真正计算信息熵花费的时间: "
				+ (endTime - startTime) + " ms.";

		System.out.println(runTimeInformation);*/
		
		return entropy;
	}// Of conditionalEntropy

	/**
	 ************************* 
	 * Computes the entropy of a dataset by considering the class attribute.
	 * This is valid for all data since the decision attribute is always
	 * nominal.
	 * 
	 * @return the entropy of the decision system's class distribution
	 * @see #classEntropy(Instances)
	 ************************* 
	 */
	public double classEntropy() {
		return classEntropy(this);
	}// Of classEntropy

	/**
	 ************************* 
	 * Computes the entropy of a dataset by considering the class attribute. It
	 * is valid for all data since the decision attribute is always nominal.
	 * Copied from Id3.java.
	 * 
	 * @param paraData
	 *            the data for which entropy is to be computed
	 * @return the entropy of the data's class distribution
	 ************************* 
	 */
	public double classEntropy(Instances paraData) {
		double[] classCounts = new double[paraData.numClasses()];
		for (int i = 0; i < paraData.numInstances(); i++) {
			classCounts[(int) paraData.instance(i).classValue()]++;
		}

		double entropy = 0;
		for (int i = 0; i < paraData.numClasses(); i++) {
			if (classCounts[i] > 0) {
				entropy -= classCounts[i] * Utils.log2(classCounts[i]);
			}
		}// Of for j
		entropy /= (double) paraData.numInstances();
		entropy += Utils.log2(paraData.numInstances());// important!

		return entropy;
	}// Of classEntropy

	/**
	 ************************* 
	 * Compute the majority number of instances, for overload.
	 * 
	 * @return the majority number of instances.
	 * @see #majority(boolean[])
	 ************************* 
	 */
	public int majority() throws Exception {
		return majority(allConditions);
	}// Of majority

	/**
	 ************************* 
	 * Compute the majority number of instances, for overload.
	 * 
	 * @param paraSelectedAttributesByLong
	 *            the test set represented by an integer.
	 * @return the majority number of instances.
	 * @throws Exception
	 *             for negative number. Maybe not initialized.
	 * @see #majority(boolean[])
	 ************************* 
	 */
	public int majority(long paraSelectedAttributesByLong) throws Exception {
		if (paraSelectedAttributesByLong < 0) {
			throw new Exception(
					"Exception occurred in RoughDecisionSystem.majority(),\r\n"
							+ "   the integer specifying the test set should not be negative.");
		}

		long currentTestSetLong = paraSelectedAttributesByLong;
		boolean[] selectedAttributes = SimpleTool.longToBooleanArray(
				currentTestSetLong, numberOfConditions);
		return majority(selectedAttributes);
	}// Of majority

	/**
	 ************************* 
	 * Compute the majority number of instances, should be overridden.
	 * 
	 * @param paraSelectedAttributes
	 *            selected attributes represented as a boolean array.
	 * @return the majority number of instances.
	 ************************* 
	 */
	public int majority(boolean[] paraSelectedAttributes) throws Exception {
		return Integer.MIN_VALUE;
	}// Of majority

	
	// //////////////////////Part 3//////////////////////
	/**
	 ************************* 
	 * Initialize for reduction. All attributes are marked as unselected.
	 * Missing values are not supported.
	 * 
	 * @throws Exception
	 *             missing values exists.
	 ************************* 
	 */
	public void initializeForReduction() throws Exception {
		
		if (hasMissingValue()) {
			throw new Exception(
					"Error occurred in RoughDecisionSystem.initializeForReduction(), "
							+ "missing values not supported.");
		}// Of if
		currentSelectedAttributes = new boolean[numberOfConditions];
		for (int i = 0; i < numberOfConditions; i++) {
			currentSelectedAttributes[i] = false;
		}// Of for i
	}// Of initializeForReduction

	/**
	 ************************* 
	 * Get the optimal reduct. "Optimal" has different definitions in definition
	 * environment.<br>
	 * For a decision system, an optimal reduct is a minimal reduct.<br>
	 * For a test-cost-senstive decision system, an optimal reduct is a minimal
	 * test-cost reduct.<br>
	 * For a both-cost-senstive decision system, an optimal reduct is a minimal
	 * cost reduct.<br>
	 * Therefore, this method is essentially an abstract method.
	 * 
	 * @return the optimal reduct.
	 ************************* 
	 */
	public boolean[] getOptimalReduct() {
		return optimalReduct;
	}// Of getOptimalReduct

	/**
	 ************************* 
	 * Compute the core, essentially an abstract method.
	 * 
	 * @return the core with the form of boolean vector.
	 * @see #computeCore(boolean[])
	 ************************* 
	 */
	public boolean[] computeCore() throws Exception {
		return computeCore(allConditions);
	}// Of computeCore

	/**
	 ************************* 
	 * Find a core with available attributes, should be overridden.
	 * 
	 * @param paraAvailableAttributes
	 *            available attributes
	 * @return the core with the form of boolean vector.
	 ************************* 
	 */
	public boolean[] computeCore(boolean[] paraAvailableAttributes)
			throws Exception {
		core = new boolean[numberOfConditions];
		return core;
	}// Of computeCore

	/**
	 ************************* 
	 * Set the red filename such that it can be stored for and reloaded next
	 * time. It is not used to read the file this time.
	 * 
	 * @param paraFilename
	 *            the given filename.
	 ************************* 
	 */
	public void setAllReductsFilename(String paraFilename) {
		allReductsFilename = paraFilename.substring(0,
				paraFilename.length() - 5) + ".reds";
	}// Of setAllReductsFilename

	/**
	 ************************* 
	 * Get the sred filename.
	 * 
	 * @return the filename of all reducts
	 * @throws Exception
	 *             to avoid the null pointer exception.
	 ************************* 
	 */
	public String getAllReductsFilename() throws Exception {
		if (allReductsFilename == null) {
			throw new Exception(
					"Error occurred in DecisionSystem.getAllReductsFilename().\r\n"
							+ "The all reducts filename is not assigned yet.");
		}// Of if
		return allReductsFilename;
	}// Of getAllReductsFilename

	/**
	 ************************* 
	 * Read all reducts from the default reduct file, whose filename only
	 * differs from the arff file by the suffix.
	 * 
	 * @throws IOException
	 *             if an error occurr while reading the file. It is thrown
	 *             directly without processing.
	 * @throws Exception
	 *             if an attribute index exceeds the bound.
	 ************************* 
	 */
	public void readAllReducts() throws IOException, Exception {
		readAllReducts(allReductsFilename);
	}// Of readAllReducts

	/**
	 ************************* 
	 * Read all reducts from a text file with the following format:<br>
	 * 2<br>
	 * 1,3,6<br>
	 * 1,4,5,7<br>
	 * That is, the first line indicates the number of reducts. Each line
	 * corresponds with a reduct, and each element corresponds with the index of
	 * the attribute.
	 * 
	 * @param paraFilename
	 *            the filename of the reduct file.
	 * @throws IOException
	 *             if an error occurr while reading the file.
	 * @throws Exception
	 *             if an attribute index exceeds the bound.
	 ************************* 
	 */
	public void readAllReducts(String paraFilename) throws IOException,
			Exception {

		File firstTempFile = new File(paraFilename);

		if (!firstTempFile.exists()) {
			throw new Exception(
					"Exception occurred in RoughDecisionSystem.readAllReducts(),\r\n"
							+ paraFilename + " does not exist.");
		}// Of if

		RandomAccessFile tempFile = new RandomAccessFile(paraFilename, "r");

		// How many reducts? Ignore blank lines in the head.
		int tempNumberOfReducts = 0;
		String currentLine = tempFile.readLine().trim();
		while (currentLine.equals("")) {
			currentLine = tempFile.readLine().trim();
		}// Of while
		tempNumberOfReducts = Integer.parseInt(currentLine);
		if (tempNumberOfReducts < 1) {
			tempFile.close();
			throw new Exception(
					"Exception occurred in RoughDecisionSystem.readAllReducts(),\r\n"
							+ paraFilename + " contains no reduct.");
		}// Of if

		allReducts = new boolean[tempNumberOfReducts][numberOfConditions];
		// Read these reducts
		currentLine = tempFile.readLine();
		int currentIndex = 0;
		int[] currentReductInt;
		while (currentLine != null) {
			currentLine = currentLine.trim();
			if (currentLine.length() > 0) {
				currentReductInt = SimpleTool.parseIntArray(currentLine);
				for (int i = 0; i < currentReductInt.length; i++) {
					allReducts[currentIndex][currentReductInt[i]] = true;
				}// Of for i
				currentIndex++;
			}// Of if
			currentLine = tempFile.readLine();
		}// Of while

		// Obtain the minimal reduct size
		int tempNumberOfAttributes = 0;
		for (int i = 0; i < allReducts.length; i++) {
			tempNumberOfAttributes = 0;
			for (int j = 0; j < allReducts[i].length; j++) {
				if (allReducts[i][j])
					tempNumberOfAttributes++;
			}// Of for j
			if (tempNumberOfAttributes < minimalReductSize) {
				minimalReductSize = tempNumberOfAttributes;
			}// Of minimalReductSize
		}// Of for i

		tempFile.close();
	}// Of readAllReducts

	/**
	 ************************* 
	 * Print all reducts. Only for testing.<br>
	 ************************* 
	 */
	public void printAllReducts() throws Exception {
		System.out.println(getAllReductsString());
	}// Of printAllReducts

	/**
	 ************************* 
	 * Write all reducts to the default reduct file, whose filename only differs
	 * from the arff file by the suffix.
	 * 
	 * @throws Exception
	 *             For file writing.
	 ************************* 
	 */
	public void writeAllReducts() throws Exception {
		String tempFilename = arffFilename.substring(0,
				arffFilename.length() - 5) + ".reds";

		try {
			File tempFile = new File(tempFilename);
			if (tempFile.exists()) {
				tempFile.delete();
			}// Of if

			RandomAccessFile tempRandomFile = new RandomAccessFile(
					tempFilename, "rw");
			tempRandomFile.writeBytes(getAllReductsString());
			tempRandomFile.close();
		} catch (Exception ee) {
			throw new Exception(
					"Exception occurred in RoughDecisionSystem.writeAllReducts(),\r\n"
							+ "while trying to write to " + tempFilename
							+ "\r\n" + ee);
		}// Of try
	}// Of writeAllReducts

	/**
	 ************************* 
	 * Get all reducts reprensented by a string, which can be written to a .reds
	 * file directly.
	 * 
	 * @return a string indicating all reducts.
	 ************************* 
	 */
	public String getAllReductsString() {
		String tempString = "" + allReducts.length + "\r\n";
		for (int i = 0; i < allReducts.length; i++) {
			boolean firstInLine = true;
			for (int j = 0; j < allReducts[0].length; j++) {
				if (allReducts[i][j]) {
					if (!firstInLine) {
						tempString += ", ";
					}
					tempString += "" + j;
					firstInLine = false;
				}// Of if
			}// Of for j
			tempString += "\r\n";
		}// Of for i
		return tempString;
	}// Of getAllReductsString

	/**
	 ************************* 
	 * Set the reds filename such that it can be stored for and reloaded next
	 * time. It is based on the arff filename.
	 ************************* 
	 */
	public void setAllReductsFilename() {
		setAllReductsFilename(arffFilename);
	}// Of setAllReductsFilename

	/**
	 ************************* 
	 * Set the current reduct.
	 * 
	 * @param paraCurrentReduct
	 *            current reduct with a form of long.
	 ************************* 
	 */
	public void setCurrentReduct(long paraCurrentReduct) {
		currentReduct = SimpleTool.longToBooleanArray(paraCurrentReduct,
				numberOfConditions);
	}// Of setCurrentReduct

	/**
	 ************************* 
	 * Set the current reduct. The current reduct is a clone of the given array
	 * to avoid conflict.
	 * 
	 * @param paraCurrentReduct
	 *            current reduct with a form of boolean array.
	 ************************* 
	 */
	public void setCurrentReduct(boolean[] paraCurrentReduct) {
		currentReduct = SimpleTool.copyBooleanArray(paraCurrentReduct);
	}// Of setCurrentReduct

	/**
	 ************************* 
	 * Get the current reduct.
	 * 
	 * @return the reduct with the form of boolean vector.
	 ************************* 
	 */
	public boolean[] getCurrentReduct() {
		return currentReduct;
	}// Of getCurrentReduct

	/**
	 ************************* 
	 * Get the index of the current index. Although the current reduct is
	 * obtained in subclasses, this method will not be overridden.
	 * 
	 * @return the index of the current reduct.
	 * @throws Exception
	 *             if the reduct or the set of all reducts not computed yet, or
	 *             the current reduct is not valid.
	 ************************* 
	 */
	public int getCurrentReductIndex() throws Exception {
		int matchedIndex = -1;
		if (allReducts == null) {
			throw new Exception(
					"Error occurred in DecisionSystem.getCurrentReductIndex():\r\n"
							+ "The set of all reducts not obtained (read or computed) yet.");
		}// Of if

		if (currentReduct == null) {
			throw new Exception(
					"Error occurred in DecisionSystem.getCurrentReductIndex(): "
							+ "The current reduct not computed yet.");
		}// Of if

		// Match one reduct?
		boolean matches = true;
		for (int i = 0; i < allReducts.length; i++) {
			matches = true;
			for (int j = 0; j < allReducts[i].length; j++) {
				if (currentReduct[j] != allReducts[i][j]) {
					matches = false;
					break;
				}// Of if
			}// Of for j

			if (matches) {
				matchedIndex = i;
				break;
			}// Of if
		}// Of for i

		if (matchedIndex == -1) {
			String errorMessage = "Error occurred in DecisionSystem.getCurrentReductIndex(): "
					+ getReductString() + " does not match any reduct.";
			throw new Exception(errorMessage);
		}// Of if

		return matchedIndex;
	}// Of getCurrentReductIndex

	/**
	 ************************* 
	 * Get the current reduct with the form of String.
	 * 
	 * @return the reduct with the form of String.
	 ************************* 
	 */
	public String getReductString() throws Exception {
		if (currentReduct == null) {
			throw new Exception(
					"Error occurred in RoughDecisionSystem.getReductString():\r\n"
							+ "Current reduct not computed yet.");
		}
		String reductString = "";
		for (int i = 0; i < currentReduct.length; i++) {
			if (currentReduct[i]) {
				reductString += "" + i + ",";
			}// Of if
		}// Of for i
		return reductString;
	}// Of getReductString

	// //////////////////////Part 4//////////////////////
	/**
	 ************************* 
	 * Get the candidate subreduct entropy array. May be used somewhere else
	 * such as AllSubreductsDialog.java.
	 ************************* 
	 */
	public int[] getCandidateSubreductConsistencyArray() {
		return candidateSubreductConsistencyArray;
	}// Of getCandidateSubreductConsistencyArray

	/**
	 ************************* 
	 * Set the sred filename such that it can be stored for and reloaded next
	 * time. It is based on the arff filename.
	 ************************* 
	 */
	public void setAllSubreductsFilename() {
		setAllSubreductsFilename(arffFilename);
	}// Of setAllSubreductsFilename

	/**
	 ************************* 
	 * Set the sred filename such that it can be stored for and reloaded next
	 * time. It is not used to read the file this time.
	 * 
	 * @param paraFilename
	 *            the given filename.
	 ************************* 
	 */
	public void setAllSubreductsFilename(String paraFilename) {
		allSubreductsFilename = paraFilename.substring(0,
				paraFilename.length() - 5) + ".sred";
	}// Of setAllSubreductsFilename

	/**
	 ************************* 
	 * Get the sred filename.
	 * 
	 * @return the filename of all subreducts
	 ************************* 
	 */
	public String getAllSubreductsFilename() throws Exception {
		if (allSubreductsFilename == null) {
			throw new Exception(
					"Error occurred in DecisionSystem.getAllReductsFilename().\r\n"
							+ "The all subreducts filename is not assigned yet.");
		}// Of if
		return allSubreductsFilename;
	}// Of getSubreductsName

	/**
	 ************************* 
	 * Read all subreducts from the default subreduct file, whose filename only
	 * differs from the arff file by the suffix. The entropy of each subreduct
	 * is stored, therefore it is inappropriate to employ positive region based
	 * approaches while reading the file.
	 * 
	 * @throws IOException
	 *             if an error occurr while reading the file.
	 * @throws Exception
	 *             if an attribute index exceeds the bound.
	 * @see #readAllSubreducts(String)
	 ************************* 
	 */
	public void readAllSubreducts() throws IOException, Exception {
		File tempFile = new File(allSubreductsFilename);
		if (tempFile.exists()) {
			readAllSubreducts(allSubreductsFilename);
		}// Of if
	}// Of readAllSubreducts

	/**
	 ************************* 
	 * Read all reducts from a text file with the following format:<br>
	 * 2<br>
	 * 1,3,6: 0.0135<br>
	 * 1,4,5,7: 0<br>
	 * That is, the first line indicates the number of subreducts. Each line
	 * corresponds with a subreduct, the real number correspond with the
	 * conditional entropy of the subreduct.
	 * 
	 * @param paraFilename
	 *            the filename of the reduct file.
	 * @throws IOException
	 *             if an error occurr while reading the file.
	 * @throws Exception
	 *             if an attribute index exceeds the bound.
	 ************************* 
	 */
	// @SuppressWarnings("resource")
	public void readAllSubreducts(String paraFilename) throws IOException,
			Exception {
		RandomAccessFile tempSubreductFile = new RandomAccessFile(paraFilename,
				"r");

		// How many reducts?
		int tempNumberOfSubreducts = 0;
		String currentLine = tempSubreductFile.readLine().trim();
		// Skip blank lines
		while (currentLine.equals("")) {
			currentLine = tempSubreductFile.readLine().trim();
		}// Of while

		// The first line serves for the measure
		int tempMeasure = Integer.parseInt(currentLine);
		if (tempMeasure != measure) {
			tempSubreductFile.close();
			throw new Exception(
					"Error occurred in RoughDecisionSystem.readAllSubreducts()\r\n"
							+ "  The measure of the file (" + tempMeasure
							+ ") does not match that of the current system ("
							+ measure + ")");
		}
		// The second line servers for the number of subreducts
		currentLine = tempSubreductFile.readLine().trim();
		tempNumberOfSubreducts = Integer.parseInt(currentLine);

		// Allocate space
		allSubreducts = new boolean[tempNumberOfSubreducts][numberOfConditions];
		candidateSubreductConsistencyArray = new int[tempNumberOfSubreducts];

		// Read these subreducts
		currentLine = tempSubreductFile.readLine();
		int currentIndex = 0;
		int[] currentReductInt;
		while (currentLine != null) {
			currentLine = currentLine.trim();
			if (currentLine.length() > 0) {
				currentReductInt = SimpleTool.parseIntArray(currentLine);
				for (int i = 0; i < currentReductInt.length; i++) {
					allSubreducts[currentIndex][currentReductInt[i]] = true;
				}// Of for i
				candidateSubreductConsistencyArray[currentIndex] = SimpleTool
						.parseIntValueAfterColon(currentLine);
				currentIndex++;
			}// Of if
			currentLine = tempSubreductFile.readLine();
		}// Of while

		tempSubreductFile.close();
	}// Of readAllSubreducts

	// //////////////////////Part 5//////////////////////
	/**
	 ************************* 
	 * Get the reduction run time.
	 * 
	 * @return the reduction run time.
	 ************************* 
	 */
	public long getReductionTime() {
		return (endTime - startTime);
	}// Of getReductionTime

	/**
	 ************************* 
	 * Get the run time information.
	 * 
	 * @return the run time informaion in a string.
	 ************************* 
	 */
	public String getRunTimeInformation() {
		if (runTimeInformation.equals("")) {
			runTimeInformation = "Time used: " + (endTime - startTime) + " ms";
		}// Of if
		return runTimeInformation;
	}// Of getRunTimeInformation

	/**
	 ************************* 
	 * Get the backtrack steps. Many subclasses need backtrack algorithms.
	 * 
	 * @return the backtrack steps.
	 ************************* 
	 */
	public long getBacktrackSteps() {
		return backtrackSteps;
	}// Of getBacktrackSteps

	// //////////////////////Others//////////////////////
	/**
	 ************************* 
	 * Delete elements marked FALSE in a boolean array.
	 * 
	 * @param paraIndices
	 *            The indices marked with false are deleted. Although not
	 *            checked, the length of paraIndices should be
	 *            numberOfInstances.
	 ************************* 
	 */
	public void delete(boolean[] paraIndices) {
		for (int i = numberOfInstances - 1; i >= 0; i--) {
			if (!paraIndices[i]) {
				delete(i);
			}// Of if
		}// Of for i
	}// Of delete

	
	
	
		
		
		
	/**
	 ************************* 
	 * Divide the decision system in two according to the given percentage.
	 * Obtain two subsets for training and testing classifiers.
	 * 
	 * @param paraPercentage
	 *            The percentage of instances in the first subset.
	 * @return Two decision systems in an array.
	 ************************* 
	 */
	public RoughDecisionSystem[] divideInTwo(double paraPercentage)
			throws Exception {
		//System.out.println("\r\n numberOfInstances=" + numberOfInstances);
		boolean[] firstInclusionArray = SimpleTool
				.generateBooleanArrayForDivision(numberOfInstances,
						paraPercentage);
		//System.out.print("firstInclusionArray=[");
		//for (int i = 0; i < firstInclusionArray.length; i++) {
			//System.out.print(firstInclusionArray[i] + ",");
		//}
		//System.out.print("]\r\n");

		RoughDecisionSystem firstDecisionSystem = new RoughDecisionSystem(this);
		firstDecisionSystem.delete(firstInclusionArray);

		boolean[] secondInclusionArray = SimpleTool
				.revertBooleanArray(firstInclusionArray);
		RoughDecisionSystem secondDecisionSystem = new RoughDecisionSystem(this);
		secondDecisionSystem.delete(secondInclusionArray);

		RoughDecisionSystem[] subsets = new RoughDecisionSystem[2];
		subsets[0] = firstDecisionSystem;
		subsets[1] = secondDecisionSystem;
		

		return subsets;
	}// Of divideInTwo

	/**
	 ************************* 
	 * Normalize the current decision system and write to an arff file. Symbolic
	 * attributes are also normalized. Do not normalize the class. Missing
	 * values are set to 0.5. After normalization, this decision system is not
	 * modified.
	 * 
	 * @return the normalized decision system.
	 * @throws Exception
	 *             for I/O exceptions.
	 ************************* 
	 */
	public RoughDecisionSystem normalize() throws Exception {
		double[][] dataMatrix = new double[numberOfInstances][numberOfConditions];
		String filename = arffFilename.substring(0, arffFilename.length() - 5)
				+ "_norm.arff";
		// Step 1. Normalize the data.
		for (int i = 0; i < numberOfConditions; i++) {
			int tempType = attribute(i).type();
			if (tempType == 1) {
				// The attribute is symbolic
				int numberOfDistinctValues = numDistinctValues(i);
				for (int j = 0; j < numInstances(); j++) {
					dataMatrix[j][i] = (instance(j).value(i) + 0.0)
							/ (numberOfDistinctValues - 1);
				}// Of for j
			} else if (tempType == 0) {
				// The attribute is numeric
				double maxValue = Double.MIN_VALUE;
				double minValue = Double.MAX_VALUE;
				for (int j = 0; j < numberOfInstances; j++) {
					if (maxValue < instance(j).value(i)) {
						maxValue = instance(j).value(i);
					} else if (minValue > instance(j).value(i)) {
						minValue = instance(j).value(i);
					}// Of if
				}// Of for j

				// Only one value for the attribute
				if (maxValue == minValue) {
					for (int j = 0; j < numberOfInstances; j++) {
						dataMatrix[j][i] = 0.5;
					}// Of for j
				} else {
					// Normalize
					for (int j = 0; j < numberOfInstances; j++) {
						dataMatrix[j][i] = (instance(j).value(i) - minValue)
								/ (maxValue - minValue);
					}// Of for j
				}// Of if equals
			} else {
				throw new Exception(
						"Error occurred in DecisionSystem.nomalize()"
								+ "Currently only numeric (0) and symbolic (1) data are supported."
								+ " The given type is " + tempType);
			}// Of for if
		}// Of for i

		// Step 2. Construct the file text.
		String messageNormalized = "";
		NumberFormat numFormat = NumberFormat.getNumberInstance();
		numFormat.setMaximumFractionDigits(5);
		for (int i = 0; i < numberOfInstances; i++) {
			for (int j = 0; j < numberOfConditions; j++) {
				if (instance(i).isMissing(j)) {
					dataMatrix[i][j] = 0.5;
				}// Of if missing value
				messageNormalized += numFormat.format(dataMatrix[i][j]) + ", ";
			}// Of for j
			messageNormalized += classAttribute().value(
					(int) instance(i).value(numberOfConditions));
			messageNormalized += "\r\n";// including class attribute
		}// Of for i

		StringBuffer text = new StringBuffer();
		// @RELATION The name of decision system
		text.append(ARFF_RELATION + " " + Utils.quote(relationName()) + "\r\n");
		// Conditional attributes
		for (int i = 0; i < numberOfConditions; i++) {
			text.append("@ATTRIBUTE " + attribute(i).name());
			text.append(" real\r\n");
		}// Of for i

		// The class attribute and all its values
		text.append("@ATTRIBUTE class{");
		for (int i = 0; i < numClasses(); i++) {
			text.append(classAttribute().value(i));
			if (i < numClasses() - 1) {
				text.append(",");
			} else {
				text.append("}\r\n");
			}
		}// Of for i

		// @DATA
		text.append("\r\n" + ARFF_DATA + "\r\n");
		// All instances
		text.append(messageNormalized);

		// Step 3. Write to an arff file.
		try {
			String s = text.toString();
			FileWriter fw = new FileWriter(filename);
			fw.write(s, 0, s.length());
			fw.flush();
			fw.close();
		} catch (Exception ee) {
			throw new Exception(
					"Error occurred in DecisionSystem.nomalize() while writing to a file.\r\n"
							+ ee);
		}// Of try

		// Step 4. Read the arff file and return.
		RoughDecisionSystem currentDs = null;
		try {
			FileReader fileReader = new FileReader(filename);
			currentDs = new RoughDecisionSystem(fileReader);
			currentDs.setClassIndex(numberOfConditions);
			fileReader.close();
		} catch (Exception ee) {
			throw new Exception(
					"Error occurred in DecisionSystem.nomalize() while reading from a file.\r\n"
							+ ee);
		}// Of try
		return currentDs;
	}// Of normalize

	/**
	 ********************************** 
	 * Get the arff source.
	 * 
	 * @return arff source as a string, which can be wrote to an arff file
	 *         directly.
	 ********************************** 
	 */
	public String getArffSource() {
		return super.toString();
	}// Of getArffSource

	/**
	 ********************************** 
	 * For toString(). See interface toString().
	 * 
	 * @return object status.
	 ********************************** 
	 */
	public String toString() {
		String reportString = "Summary:\r\n";
		reportString += "The .arff filename is: " + arffFilename + "\r\n";
		reportString += "Current decision system has " + numAttributes()
				+ " attributes (including the decision) and " + numInstances()
				+ " objects.\r\n";
		reportString += "==================\r\n";
		reportString += "Basic data of the decision system:\r\n"
				+ super.toString();
		return reportString;
	}// Of toString
	
	
	/**
	 ************************* 
	 * Find a (possibly) minimal reduct using the information entropy based
	 * approach.
	 * 
	 * @return the reduct with the form of boolean vector.
	 ************************* 
	 */
	public boolean[] entropyBasedReduction() throws Exception {
		initializeForReduction();
		
		double currentEntropy = 100;
		totalEntropy = conditionalEntropy();

		// Step 1. Compute the core
		computeCore();
		/*
		System.out.println("核属性为：");
		for(int i = 0; i < core.length; i++){
			System.out.println(core[i]);
		}
		System.out.println("-------------");
		*/
		// Step 2. Add attributes
		// Step 2.1 Copy core attributes
		int numberOfCoreAttributes = 0;
		for (int i = 0; i < numberOfConditions; i++) {
			currentSelectedAttributes[i] = core[i];
			if (currentSelectedAttributes[i]) {
				numberOfCoreAttributes++;
			}// Of if
		}// Of for i

		
		if (numberOfCoreAttributes == 0) {
			currentEntropy = 100;
		} else {
			currentEntropy = conditionalEntropy(currentSelectedAttributes);
		}
		int currentBestAttribute = -1;
		
		// Step 2.2 Add attributes one by one
		while (Math.abs(currentEntropy - totalEntropy) > 1e-6) {
			double currentBestEntropy = 10000;
			for (int i = 0; i < numberOfConditions; i++) {
				// Ignore selected attributes
				if (currentSelectedAttributes[i])
					continue;

				// Try this attribute
				currentSelectedAttributes[i] = true;
				currentEntropy = conditionalEntropy(currentSelectedAttributes);
				if (currentBestEntropy > currentEntropy) {
					currentBestEntropy = currentEntropy;
					currentBestAttribute = i;
				}
				// Set back
				currentSelectedAttributes[i] = false;
			}// Of for i
				// Really add it
			currentSelectedAttributes[currentBestAttribute] = true;
			currentEntropy = conditionalEntropy(currentSelectedAttributes);
		}// Of while
		
		// Step 3. Remove redundant attributes
		for (int i = 0; i < numberOfConditions; i++) {
			// Ignore core attributes and not selected attributes
			if ((core[i]) || (!currentSelectedAttributes[i])) {
				continue;
			}// Of if

			// Try to remove this attribute
			currentSelectedAttributes[i] = false;
			currentEntropy = conditionalEntropy(currentSelectedAttributes);
			if (Math.abs(currentEntropy - totalEntropy) > 1e-6) {
				// Set back
				currentSelectedAttributes[i] = true;
			} else {
				// This rarely happens, so output this message directly.
				//System.out.println("attribute #" + i + " can be removed.");
			}// Of if
		}// Of for i
	
		currentReduct = currentSelectedAttributes;

		// int currentReductIndex = getCurrentReductIndex();
		// if (currentReductIndex == -1) {
			// String errorMessage = "Error occurred in NominalDecisionSystem.entropyBasedReduction(): "
					// + getReductString() + " is not a reduct";
			// throw new Exception(errorMessage);
		// }// Of if
		// System.out.println("约简结果");
		
		for(int i = 0; i < currentReduct.length; i++){
			System.out.println(currentReduct[i]);
		}
		
		return currentReduct;
	}// Of entropyBasedReduction
	
	/**
	 ************************* 
	 * The main function.
	 ************************* 
	 */
	public static void main(String args[]) {

		String arffFilename = "D:/Java/MultiLabel/Result/test_attribute0_Graph.arff";

		try{
			FileReader fileReader = new FileReader(arffFilename);
			RoughDecisionSystem decisionSystem = new RoughDecisionSystem(fileReader);
			decisionSystem.setClassIndex(decisionSystem.numberOfConditions);
			decisionSystem.setArffFilename(arffFilename);
			fileReader.close();
	
			System.out.println(decisionSystem);
			
			// decisionSystem.entropyBasedReduction();
			
			// SimpleTool.printBooleanArray(currentReduct);
			
			
		} catch (Exception ee) {
			System.out.println("Error occurred while trying to read \'"
					+ arffFilename + "\' in CoserProject.readArffFile().\r\n"
					+ ee);
		}// Of try
	}// Of main

}// class RoughDecisionSystem
