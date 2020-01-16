/*
 * @(#)SimpleTool.java
 *
 */

package coser.common;

import java.util.*;

import weka.core.Utils;
import weka.core.*;
import java.util.LinkedList;
import java.util.List;

/**
 * Frequently used methods to convert strings, integers, codings, etc.
 * <p>
 * Author: <b>Fan Min</b> minfanphd@163.com <br>
 * Copyright: The source code and all documents are open and free. PLEASE keep
 * this header while revising the program. <br>
 * Organization: <a href=http://grc.fjzs.edu.cn/>Lab of Granular Computing</a>,
 * Zhangzhou Normal University, Fujian 363000, China.<br>
 * Project: The cost-sensitive rough sets project.
 * <p>
 * Progress: OK. Copied from Hydrosimu.<br>
 * Written time: March 09, 2011. <br>
 * Last modify time: March 09, 2011.
 */
public class SimpleTool extends Object {
	/**
	 * Maximal length of the parsed array (for int and double).
	 */
	public static final int MAX_PARSE_ARRAY_LENGTH = 100;

	/**
	 * A global random object.
	 */
	public static final Random random = new Random();

	/**
	 ********************************** 
	 * Convert a double value into a shorter string.
	 * 
	 * @param paraDouble
	 *            the double value to be converted.
	 ********************************** 
	 */
	public static String shorterDouble(double paraDouble) {
		return shorterDouble(paraDouble, 7);
	}// Of shorterDouble

	/**
	 ********************************** 
	 * Convert a double value into a shorter string.
	 * 
	 * @param paraDouble
	 *            the double value to be converted.
	 * @param paraLength
	 *            the length of reserved double.
	 ********************************** 
	 */
	public static String shorterDouble(double paraDouble, int paraLength) {
		// double absValue = Math.abs(paraDouble);
		// double least = 0.001;
		// for (int i = 2; i < paraLength; i++)
		// least *= 0.1;
		// if (absValue < least)
		// return "0.0";

		String shorterString = new Double(paraDouble).toString();
		// For uncontrollable data, just output itself
		if ((paraDouble < 0.001) && (-0.001 < paraDouble)
				|| (paraDouble < -9999) || (9999 < paraDouble))
			return shorterString;

		if ((paraDouble > 0) && (shorterString.length() > paraLength))
			shorterString = shorterString.substring(0, paraLength);
		if ((paraDouble < 0) && (shorterString.length() > paraLength + 1))
			shorterString = shorterString.substring(0, paraLength + 1);

		return shorterString;
	}// Of shorterDouble

	/**
	 ********************************** 
	 * Convert a string with commas into a string array, blanks adjacent with
	 * commas are deleted. If the string bewteen two adjacent commas are blank
	 * or contains only space char ' ', then an exception will be thrown.<br>
	 * For example, "a, bc, def, g" will be converted into a string array with 4
	 * elements "a", "bc", "def" and "g".
	 * 
	 * @param prmString
	 *            The source string
	 * @return A string array.
	 * @throws Exception
	 *             Exception for illegal data.
	 * @see #stringArrayToString(java.lang.String[])
	 ********************************** 
	 */
	public static String[] stringToStringArray(String prmString)
			throws Exception {
		/*
		 * Convert this string into an array such that another method could be
		 * invoked.
		 */
		int tempCounter = 1;
		for (int i = 0; i < prmString.length(); i++) {
			if (prmString.charAt(i) == ',') {
				tempCounter++;
			}// Of if
		}// Of for i

		String[] theStringArray = new String[tempCounter];

		String remainString = new String(prmString) + ",";
		for (int i = 0; i < tempCounter; i++) {
			theStringArray[i] = remainString.substring(0,
					remainString.indexOf(",")).trim();
			if (theStringArray[i].equals("")) {
				throw new Exception(
						"Error occurred in common.SimpleTool.stringToStringArray()."
								+ "\n\tBlank attribute or data is not allowed as a data. "
								+ "\n\tThe string is:" + prmString);
			}// Of if
				// Common.println(theStringArray[i]);
			remainString = remainString
					.substring(remainString.indexOf(",") + 1);
			// Common.println("remainString: " + remainString);
		}// Of for i

		return theStringArray;
	}// Of stringToStringArray

	/**
	 ********************************** 
	 * Convert a string array into a string, elements are separated by commas.
	 * 
	 * @param prmStringArray
	 *            The source string array
	 * @return converted string.
	 * @see #stringToStringArray(java.lang.String)
	 ********************************** 
	 */
	public static String stringArrayToString(String[] prmStringArray) {
		String newString = "";
		for (int i = 0; i < prmStringArray.length; i++) {
			newString += new String(prmStringArray[i]) + ",";
		}// Of for

		// Delete the last comma
		newString = newString.substring(0, newString.length() - 1);

		return newString;
	}// Of stringArrayToString

	/**
	 ********************************** 
	 * Add single quotes for a string (may be an array, elements are separated
	 * by commas). This is needed in some SQL statements. For example, "ab,c,d"
	 * will be converted into "ab','c','d".
	 * 
	 * @param prmStringArray
	 *            The source string
	 * @return converted string.
	 ********************************** 
	 */
	public static String addSingleQuotes(String prmStringArray) {
		int tempIndexInString = prmStringArray.indexOf(",");
		while (tempIndexInString > -1) {
			prmStringArray = prmStringArray.substring(0, tempIndexInString)
					+ "\';\'"
					+ prmStringArray.substring(tempIndexInString + 1,
							prmStringArray.length());
			tempIndexInString = prmStringArray.indexOf(",");
		}// Of while

		prmStringArray = "\'" + prmStringArray.replace(';', ',') + "\'";

		return prmStringArray;
	}// Of addSingleQuotes

	/**
	 ********************************** 
	 * If the string quisi-array (elements are separated by commas) contains
	 * respective attribute.<br>
	 * For exampe "abc, de" contains "de", but it does not contain "d".
	 * 
	 * @param prmStringArray
	 *            The string quisi-array (elements are separated by commas).
	 * @param prmString
	 *            Respect string.
	 * @return If it is contained.
	 * @see #stringToStringArray(java.lang.String)
	 ********************************** 
	 */
	public static boolean stringArrayContainsString(String prmStringArray,
			String prmString) {
		String[] realStringArray = null;
		try {
			realStringArray = stringToStringArray(prmStringArray);
		} catch (Exception ee) {
			return false;
		}// Of try

		for (int i = 0; i < realStringArray.length; i++) {
			if (realStringArray[i].equals(prmString)) {
				return true;
			}// Of if
		}// Of for
		return false;
	}// Of stringArrayContainsString

	/**
	 ********************************** 
	 * Join two attribute strings, separated by a comma.<br>
	 * 
	 * @param prmFirstString
	 *            The first attribute string
	 * @param prmSecondString
	 *            The second attribute string
	 * @return concated string.
	 ********************************** 
	 */
	public static String joinString(String prmFirstString,
			String prmSecondString) {
		if (prmFirstString.equals(""))
			return prmSecondString + "";
		if (prmSecondString.equals(""))
			return prmFirstString + "";

		return prmFirstString + "," + prmSecondString;
	}// Of joinString

	/**
	 ********************************** 
	 * Convert a string with delimiters (such as commas or semi-commas) into a
	 * string array.<br>
	 * This method is more generalized than stringToStringArray because the
	 * latter only permits commas to be delimiters. For more detail please
	 * contact <A href="mailto:qiheliu@uestc.edu.cn">Liu Qihe</A>
	 * 
	 * @param prmString
	 *            The given string
	 * @param prmDelimiter
	 *            The given delimiter
	 * @param prmReturnTokens
	 *            Is the delimiter permitted after convertion.
	 * @return string array separated by commas
	 ********************************** 
	 */
	public static String[] parseString(String prmString, String prmDelimiter,
			boolean prmReturnTokens) {
		String[] returnString = null;
		StringTokenizer token;
		if (prmString != null) {
			token = new StringTokenizer(prmString.trim(), prmDelimiter,
					prmReturnTokens);
			returnString = new String[token.countTokens()];
			int i = 0;
			while (token.hasMoreTokens()) {
				returnString[i] = (String) token.nextToken();
				i++;
			}// Of while
		}// Of if

		return returnString;
	}// end of parseString

	/**
	 ********************************** 
	 * Remove string from a string array, and return a string. For more detail
	 * please contact <A href="mailto:qiheliu@uestc.edu.cn">Liu Qihe</A>
	 * 
	 * @param prmString
	 *            字符串
	 * @param index
	 *            去掉串的位\uFFFD
	 * @return 字符串，属性值之间用逗号分隔.
	 ***********************************/
	public static String generateString(String[] prmString, int index) {
		String result = "";
		for (int i = 0; i < prmString.length; i++) {
			if (i == index)
				continue;
			result += prmString[i] + ",";
		}
		result = result.substring(0, result.length() - 1);
		return result;

	}// of generateString

	/**
	 ********************************* 
	 * GB2312 to UNICODE. Use this one if Chinese characters is a mess.
	 * 
	 * @param paraString
	 *            a GB2312 string
	 * @return a UNICODE string.
	 * @see #UNICODEToGB2312(java.lang.String)
	 ********************************* 
	 */
	public static String GB2312ToUNICODE(String paraString) {
		char[] tempCharArray = paraString.toCharArray();
		int tempLength = tempCharArray.length;
		byte[] tempByteArray = new byte[tempLength];
		for (int i = 0; i < tempLength; i++) {
			tempByteArray[i] = (byte) tempCharArray[i];
		}// Of for.

		String returnString = new String(tempByteArray);
		return returnString;
	}// Of GB2312ToUNICODE.

	/**
	 ********************************* 
	 * UNICODE to GB2312. Use this one if Chinese characters is a mess.
	 * 
	 * @param paraString
	 *            a UNICODE string.
	 * @return a GB2312 string.
	 * @see #GB2312ToUNICODE(java.lang.String)
	 ********************************* 
	 */
	public static String UNICODEToGB2312(String paraString) {
		// Convert the string into a byte array.
		byte[] byteArray = paraString.getBytes();

		int arrayLength = byteArray.length;

		// Store converted char array.
		char[] charArray = new char[arrayLength];

		// Convert chars one by one.
		for (int i = 0; i < arrayLength; i++) {
			// Add an all 0 byte.
			charArray[i] = (char) byteArray[i];
		}// Of for.

		// Get a new string according to the converted array.
		String convertedString = new String(charArray);
		return convertedString;
	}// Of UNICODEToGB2312.

	/**
	 ********************************** 
	 * One string minus another, essentially corresponding string arrays.<br>
	 * E.g., "ab,cd,efa" minus "ab,efa" gets "cd"
	 * 
	 * @param prmFirstString
	 *            The first string.
	 * @param prmSecondString
	 *            The second string.
	 * @return A result string.
	 * @throws Exception
	 *             if the first string does not contain an element of the second
	 *             string, e.g., "ab, cd" minus "ab, de". #see
	 *             scheme.SymbolicPartition
	 *             .computeOptimalPartitionReduct(java.lang.String,
	 *             java.lang.String, int)
	 ********************************** 
	 */
	public static String stringMinusString(String prmFirstString,
			String prmSecondString) throws Exception {
		String[] firstArray = stringToStringArray(prmFirstString);
		String[] secondArray = stringToStringArray(prmSecondString);

		boolean[] includeArray = new boolean[firstArray.length];
		for (int i = 0; i < includeArray.length; i++) {
			includeArray[i] = true;
		}// Of for i

		// Given an element of secondArray, is there an identical element of
		// firstArray?
		boolean found = false;
		for (int i = 0; i < secondArray.length; i++) {
			found = false;
			for (int j = 0; j < firstArray.length; j++) {
				if (secondArray[i].equals(firstArray[j])) {
					includeArray[j] = false;
					found = true;
					break;
				}
			}// Of for j
			if (!found)
				throw new Exception(
						"Error occured in SimpleTool.stringMinusString(), \n"
								+ "\t" + secondArray[i] + "is not included in "
								+ prmFirstString);
		}// Of for i

		String returnString = "";
		for (int i = 0; i < includeArray.length; i++) {
			if (includeArray[i])
				returnString += firstArray[i] + ",";
		}// Of for i

		if (returnString.length() > 0)
			returnString = returnString.substring(0, returnString.length() - 1);
		return returnString;
	}// Of stringMinusString

	/**
	 ********************************** 
	 * One string union another, essentially corresponding string sets.<br>
	 * E.g., "ab,cd,efa" union "ab,ee" gets "ab,cd,efa, ee"
	 * 
	 * @param prmFirstString
	 *            The first string.
	 * @param prmSecondString
	 *            The second string.
	 * @return A result string. #see
	 *         scheme.Reduction.computeOptimalMReductByEntropy(java.lang.String,
	 *         java.lang.String, java.lang.String)
	 ********************************** 
	 */
	public static String stringUnionString(String prmFirstString,
			String prmSecondString) throws Exception {
		if (prmFirstString.equals(""))
			return prmSecondString + "";
		if (prmSecondString.equals(""))
			return prmFirstString + "";

		String[] firstArray = stringToStringArray(prmFirstString);
		String[] secondArray = stringToStringArray(prmSecondString);

		String unionString = new String(prmFirstString);

		boolean found = false;
		for (int i = 0; i < secondArray.length; i++) {
			found = false;
			for (int j = 0; j < firstArray.length; j++) {
				if (secondArray[i].equals(firstArray[j])) {
					found = true;
					break;
				}
			}// Of for j
			if (!found)
				if (unionString.equals(""))
					unionString += secondArray[i];
				else
					unionString += "," + secondArray[i];
		}// Of for i

		return unionString;
	}// Of stringUnionString

	/**
	 *************************** 
	 * Judge whether or not the given string is null/empty.
	 *************************** 
	 */
	public static boolean isEmptyStr(String paraString) {
		if (paraString == null)
			return true;
		if (paraString.equals(""))
			return true;
		return false;
	}// Of isEmptyStr

	/**
	 *************************** 
	 * Judge whether or not the given string is specified.
	 *************************** 
	 */
	public static boolean isUnspecifiedStr(String paraString) {
		if (isEmptyStr(paraString))
			return true;
		if (paraString.equals(Common.unspecifiedString))
			return true;
		return false;
	}// Of isEmptyStr

	/**
	 *************************** 
	 * Read an integer array from a given string. Integers are separated by
	 * separators. Author Fan Min.
	 * 
	 * @param paraString
	 *            The given string.
	 * @param paraNumberOfInts
	 *            The number of integer to read from the string. If there are
	 *            more integers, just ignore them.
	 * @param paraSeparator
	 *            The separator of data, blank and commas are most commonly uses
	 *            ones.
	 * @return The constructed array.
	 *************************** 
	 */
	public static int[] parseIntArray(String paraString, int paraNumberOfInts,
			char paraSeparator) throws Exception {
		int[] returnArray = new int[paraNumberOfInts];
		String currentString = null;
		String remainingString = new String(paraString);
		try {
			for (int i = 0; i < paraNumberOfInts - 1; i++) {
				currentString = remainingString.substring(0,
						remainingString.indexOf(paraSeparator)).trim();
				returnArray[i] = Integer.parseInt(currentString);
				remainingString = remainingString.substring(
						remainingString.indexOf(paraSeparator) + 1).trim();
			}// Of for i

			// The last one may have no blank after it.
			if (remainingString.indexOf(paraSeparator) < 0)
				currentString = remainingString;
			else
				currentString = remainingString.substring(0,
						remainingString.indexOf(paraSeparator));

			returnArray[paraNumberOfInts - 1] = Integer.parseInt(currentString);
		} catch (java.lang.StringIndexOutOfBoundsException sie) {
			throw new Exception(
					"Error occurred in common.SimpleTool.parseIntArray.\r\n"
							+ "May caused by the number of int value required exceeds those in the string.\r\n"
							+ sie);
		} catch (java.lang.NumberFormatException nfe) {
			throw new Exception(
					"Error occurred in common.SimpleTool.parseIntArray.\r\n"
							+ "May caused by incorrect separator (e.g., comma, blank).\r\n"
							+ nfe);
		} catch (Exception ee) {
			throw new Exception(
					"Error occurred in common.SimpleTool.parseIntArray.\r\n"
							+ ee);
		}

		return returnArray;
	}// Of parseIntArray

	/**
	 *************************** 
	 * An int to an attribute subset. Author Fan Min.
	 * 
	 * @param paraInt
	 *            The given integer representing an attribute subset.
	 * @return The attribute subset in a string.
	 *************************** 
	 */
	public static String intToAttributeSetString(int paraInt) {
		String resultString = "";
		if (paraInt < 1) {
			return resultString; // No need to throw an exception.
		}

		int currentIndex = 0;
		int tempInt = paraInt;
		while (tempInt > 0) {
			if (tempInt % 2 == 1) {
				resultString += currentIndex + ",";
			}

			tempInt /= 2;
			currentIndex++;
		}// Of while

		resultString = resultString.substring(0, resultString.length() - 1);
		return resultString;
	}// Of intToAttributeSetString

	/**
	 *************************** 
	 * An int array to an attribute subset. Author Fan Min.
	 * 
	 * @param paraIntArray
	 *            The given integer array an attribute subset.
	 * @param paraLength
	 *            The size of the subset.
	 * @return The attribute subset in a string.
	 *************************** 
	 */
	public static String intArrayToAttributeSetString(int[] paraIntArray,
			int paraLength) {
		String resultString = "";
		if (paraLength < 1) {
			return resultString; // No need to throw an exception.
		}
		for (int i = 0; i < paraLength; i++) {
			resultString += paraIntArray[i] + ",";

		}// Of for i
		resultString = resultString.substring(0, resultString.length() - 1);
		return resultString;
	}// Of intArrayToAttributeSetString

	/**
	 *************************** 
	 * An int array to an int value. For example, [0, 2] will be binary 101 = 5.
	 * 
	 * @param paraIntArray
	 *            The given integer array indicating which positions is 1.
	 * @return An int value.
	 *************************** 
	 */
	public static int intArrayToInt(int[] paraIntArray) {
		if (paraIntArray == null) {
			return 0;
		}

		int tempValue = 0;
		for (int i = 0; i < paraIntArray.length; i++) {
			tempValue += (int) Math.pow(2, paraIntArray[i]);
		}
		return tempValue;
	}// Of intArrayToInt

	/**
	 *************************** 
	 * An int array to an long value. For example, [0, 2] will be binary 101 =
	 * 5.
	 * 
	 * @param paraIntArray
	 *            The given integer representing an attribute subset.
	 * @return An int value.
	 * @see #intArrayToInt(int[])
	 *************************** 
	 */
	public static long intArrayToLong(int[] paraIntArray) {
		if (paraIntArray == null) {
			return 0;
		}

		long tempValue = 0;
		for (int i = 0; i < paraIntArray.length; i++) {
			tempValue += (long) Math.pow(2, paraIntArray[i]);
		}
		return tempValue;
	}// Of intArrayToLong

	/**
	 *************************** 
	 * A long to an attribute subset.
	 * 
	 * @param paraLong
	 *            The given integer representing an attribute subset.
	 * @return The attribute subset in a string.
	 *************************** 
	 */
	public static String longToAttributeSetString(long paraLong) {
		String resultString = "";
		if (paraLong < 1) {
			return resultString; // No need to throw an exception.
		}

		int currentIndex = 0;
		long tempLong = paraLong;
		while (tempLong > 0) {
			if (tempLong % 2 == 1) {
				resultString += currentIndex + ",";
			}

			tempLong /= 2;
			currentIndex++;
		}// Of while

		resultString = resultString.substring(0, resultString.length() - 1);
		return resultString;
	}// Of longToAttributeSetString

	/**
	 *************************** 
	 * Convert an integer array into a string. Integers are separated by
	 * separators. Author Fan Min.
	 * 
	 * @param paraArray
	 *            The given array.
	 * @param paraSeparator
	 *            The separator of data, blank and commas are most commonly uses
	 *            ones.
	 * @return The constructed String.
	 *************************** 
	 */
	public static String intArrayToString(int[] paraArray, char paraSeparator)
			throws Exception {
		String returnString = "[]";
		if ((paraArray == null) || (paraArray.length < 1))
			return returnString;
		// throw new Exception(
		// "Error occurred in common.SimpleTool. Cannot convert an empty array into a string.");
		returnString = "";
		for (int i = 0; i < paraArray.length - 1; i++) {
			returnString += "" + paraArray[i] + paraSeparator;
		}// Of for i
		returnString += paraArray[paraArray.length - 1];
		returnString = "[" + returnString + "]";

		return returnString;
	}// Of intArrayToString

	/**
	 *************************** 
	 * Read a double array from a given string. Double values are separated by
	 * separators. <br>
	 * Author Fan Min.
	 * 
	 * @param paraString
	 *            The given string.
	 * @param paraNumberOfDoubles
	 *            The number of integer to read from the string. If there are
	 *            more integers, just ignore them.
	 * @param paraSeparator
	 *            The separator of data, blank and commas are most commonly uses
	 *            ones.
	 * @return The constructed array.
	 *************************** 
	 */
	public static double[] parseDoubleArray(String paraString,
			int paraNumberOfDoubles, char paraSeparator) throws Exception {
		double[] returnArray = new double[paraNumberOfDoubles];
		String currentString = null;
		String remainingString = new String(paraString);

		try {
			for (int i = 0; i < paraNumberOfDoubles - 1; i++) {
				currentString = remainingString.substring(0,
						remainingString.indexOf(paraSeparator)).trim();
				returnArray[i] = Double.parseDouble(currentString);
				remainingString = remainingString.substring(
						remainingString.indexOf(paraSeparator) + 1).trim();
			}// Of for i

			// The last one may have no blank after it.
			if (remainingString.indexOf(paraSeparator) < 0)
				currentString = remainingString;
			else
				currentString = remainingString.substring(0,
						remainingString.indexOf(paraSeparator));

			returnArray[paraNumberOfDoubles - 1] = Double
					.parseDouble(currentString);
		} catch (java.lang.StringIndexOutOfBoundsException sie) {
			throw new Exception(
					"Error occurred in common.SimpleTool.parseDoubleArray.\r\n"
							+ "May caused by the number of double value required exceeds those in the string,\r\n"
							+ "or invalid separator.\r\n" + sie);
		} catch (java.lang.NumberFormatException nfe) {
			throw new Exception(
					"Error occurred in common.SimpleTool.parseDoubleArray.\r\n"
							+ "May caused by incorrect separator (e.g., comma, blank).\r\n"
							+ nfe);
		} catch (Exception ee) {
			throw new Exception(
					"Error occurred in common.SimpleTool.parseDoubleArray.\r\n"
							+ ee);
		}

		return returnArray;
	}// Of parseDoubleArray

	/**
	 *************************** 
	 * Read a double array from a given string. The number of doubles are not
	 * given, so I will parse as many double values as possible. The seperator
	 * is not indicated, so comma is the first candidate, and blank is the
	 * second. <br>
	 * Author Fan Min.
	 * 
	 * @param paraString
	 *            The given string.
	 * @return The constructed array.
	 *************************** 
	 */
	public static double[] parseDoubleArray(String paraString) throws Exception {
		char separator = ' ';
		if (paraString.indexOf(',') > 0)
			separator = ',';
		// String tempString = new String(paraString);

		double[] tempArray = new double[MAX_PARSE_ARRAY_LENGTH];
		int arrayLength = 0;

		String currentString = null;
		String remainingString = new String(paraString);

		try {
			while (remainingString.indexOf(separator) > 0) {
				currentString = remainingString.substring(0,
						remainingString.indexOf(separator)).trim();
				tempArray[arrayLength] = Double.parseDouble(currentString);
				arrayLength++;
				if (arrayLength >= MAX_PARSE_ARRAY_LENGTH)
					throw new Exception("The array length should not exceed "
							+ MAX_PARSE_ARRAY_LENGTH);
				remainingString = remainingString.substring(
						remainingString.indexOf(separator) + 1).trim();
			}// Of while

			tempArray[arrayLength] = Double.parseDouble(remainingString);
		} catch (java.lang.StringIndexOutOfBoundsException sie) {
			throw new Exception(
					"Error occurred in common.SimpleTool.parseDoubleArray.\r\n"
							+ "May caused by the number of double value required exceeds those in the string,\r\n"
							+ "or invalid separator.\r\n" + sie);
		} catch (java.lang.NumberFormatException nfe) {
			throw new Exception(
					"Error occurred in common.SimpleTool.parseDoubleArray.\r\n"
							+ "May caused by incorrect separator (e.g., comma, blank).\r\n"
							+ nfe);
		} catch (Exception ee) {
			throw new Exception(
					"Error occurred in common.SimpleTool.parseDoubleArray.\r\n"
							+ ee);
		}

		double[] returnArray = new double[arrayLength + 1];

		// The arrayLength is starting from 0
		for (int i = 0; i <= arrayLength; i++)
			returnArray[i] = tempArray[i];

		return returnArray;
	}// Of parseDoubleArray

	/**
	 *************************** 
	 * Conver a double array into a string. Doubles are separated by blanks.
	 * Author Fan Min.
	 * 
	 * @param paraArray
	 *            The given array.
	 * @return The constructed String.
	 *************************** 
	 */
	public static String doubleArrayToString(double[] paraArray)
			throws Exception {
		return doubleArrayToString(paraArray, ' ');
	}// Of doubleArrayToString

	/**
	 *************************** 
	 * Conver a double array into a string. Doubles are separated by separators.
	 * Author Fan Min.
	 * 
	 * @param paraArray
	 *            The given array.
	 * @param paraSeparator
	 *            The separator of data, blank and commas are most commonly uses
	 *            ones.
	 * @return The constructed String.
	 *************************** 
	 */
	public static String doubleArrayToString(double[] paraArray,
			char paraSeparator) throws Exception {
		if ((paraArray == null) || (paraArray.length < 1))
			throw new Exception(
					"Error occurred in common.SimpleTool. Cannot convert an empty array into a string.");
		String returnString = "";
		for (int i = 0; i < paraArray.length - 1; i++) {
			returnString += "" + paraArray[i] + paraSeparator;
		}// Of for i
		returnString += paraArray[paraArray.length - 1];

		return returnString;
	}// Of doubleArrayToString

	/**
	 *************************** 
	 * Convert a boolean array into a string. Booleans are separated by
	 * separators.
	 * 
	 * @param paraArray
	 *            The given array.
	 * @param paraSeparator
	 *            The separator of data, blank and commas are most commonly uses
	 *            ones.
	 * @return The constructed String.
	 *************************** 
	 */
	public static String booleanArrayToString(boolean[] paraArray,
			char paraSeparator) {
		if ((paraArray == null) || (paraArray.length < 1)) {
			return "[]";
		}// Of if

		String returnString = "[";
		for (int i = 0; i < paraArray.length; i++) {
			returnString += "" + paraArray[i] + paraSeparator;
		}// Of for i
		returnString = returnString.substring(0, returnString.length() - 1);
		returnString += "]";

		return returnString;
	}// Of booleanArrayToString

	/**
	 *************************** 
	 * Convert a string array into a boolean array. Booleans are separated by
	 * separators. Only support 0 and 1 in the string. TRUE and FALSE are not
	 * supported.
	 * 
	 * @param paraString
	 *            The given string.
	 * @param paraSeparator
	 *            The separator of data, blank and commas are most commonly uses
	 *            ones.
	 * @return The constructed String.
	 * @throws Exception
	 *             for wrong format.
	 *************************** 
	 */
	public static boolean[] stringToBooleanArray(String paraString,
			char paraSeparator) throws Exception {
		paraString.trim();
		if ((paraString == null) || (paraString.length() < 1)) {
			throw new Exception(
					"Error occurred in SimpleTool.stringToBooleanArray(String, char)."
							+ "\r\nThe given string is null.");
		}// Of if

		// Scan separators to find the size of the array
		int tempLength = 1;
		for (int i = 0; i < paraString.length(); i++) {
			if (paraString.charAt(i) == paraSeparator) {
				tempLength++;
			}// Of if
		}// Of for i

		// parse boolean values
		boolean[] resultArray = new boolean[tempLength];
		for (int i = 0; i < tempLength; i++) {
			if (paraString.charAt(i * 2) == '0') {
				resultArray[i] = false;
			} else if (paraString.charAt(i * 2) == '1') {
				resultArray[i] = true;
			} else {
				throw new Exception(
						"Error occurred in SimpleTool.stringToBooleanArray(String, char)."
								+ "\r\n Unsupported boolean value '"
								+ paraString.charAt(i * 2)
								+ "'. It should be 0 or 1.");
			}
		}// Of for i
		return resultArray;
	}// Of booleanArrayToString

	/**
	 *************************** 
	 * Conver a boolean array into a string. Booleans are separated by commas.
	 * 
	 * @param paraArray
	 *            The given array.
	 * @return The constructed String.
	 *************************** 
	 */
	public static String booleanArrayToAttributeSetString(boolean[] paraArray) {
		String returnString = "[";
		if ((paraArray == null) || (paraArray.length < 1)) {
			returnString += "]";
			return returnString;
		}// Of if

		for (int i = 0; i < paraArray.length; i++) {
			if (paraArray[i]) {
				returnString += i + ",";
			}
		}// Of for i
		returnString = returnString.substring(0, returnString.length() - 1);
		returnString += "]";

		return returnString;
	}// Of booleanArrayToAttributeSetString

	/**
	 *************************** 
	 * Read a int array from a given string. The number of ints are not given,
	 * so I will parse as many int values as possible. The seperator is not
	 * indicated, so comma is the first candidate, and blank is the second. <br>
	 * Author Fan Min.
	 * 
	 * @param paraString
	 *            The given string.
	 * @return The constructed array.
	 *************************** 
	 */
	public static int[] parseIntArray(String paraString) throws Exception {
		char separator = ' ';
		String tempString = new String(paraString);
		String remainingString = new String(paraString);
		// System.out.println(tempString);

		if (tempString.lastIndexOf(":") != -1) {
			remainingString = tempString.substring(0,
					tempString.lastIndexOf(":"));
			// System.out.println(tempString.indexOf(":"));
		}
		if (paraString.indexOf(',') > 0)
			separator = ',';

		int[] tempArray = new int[MAX_PARSE_ARRAY_LENGTH];
		int arrayLength = 0;

		String currentString = null;
		// System.out.println("Test parseIntArray before");
		try {

			while (remainingString.indexOf(separator) > 0) {
				currentString = remainingString.substring(0,
						remainingString.indexOf(separator)).trim();
				tempArray[arrayLength] = Integer.parseInt(currentString);
				arrayLength++;
				if (arrayLength >= MAX_PARSE_ARRAY_LENGTH)
					throw new Exception("The array length should not exceed "
							+ MAX_PARSE_ARRAY_LENGTH);
				remainingString = remainingString.substring(
						remainingString.indexOf(separator) + 1).trim();
			}// Of while

			tempArray[arrayLength] = Integer.parseInt(remainingString);
		} catch (java.lang.StringIndexOutOfBoundsException sie) {
			throw new Exception(
					"Error occurred in common.SimpleTool.parseIntegerArray.\r\n"
							+ "May caused by the number of int value required exceeds those in the string,\r\n"
							+ "or invalid separator.\r\n" + sie);
		} catch (java.lang.NumberFormatException nfe) {
			throw new Exception(
					"Error occurred in common.SimpleTool.parseIntegerArray.\r\n"
							+ "May caused by incorrect separator (e.g., comma, blank).\r\n"
							+ nfe);
		} catch (Exception ee) {
			throw new Exception(
					"Error occurred in common.SimpleTool.parseIntegerArray.\r\n"
							+ ee);
		}

		int[] returnArray = new int[arrayLength + 1];

		// The arrayLength is starting from 0
		for (int i = 0; i <= arrayLength; i++)
			returnArray[i] = tempArray[i];
		// System.out.println("Test parseIntArray after");
		return returnArray;
	}// Of parseIntArray

	/**
	 *************************** 
	 * Read a double value from a given string after the colon. <br>
	 * 
	 * @param paraString
	 *            The given string.
	 * @return A double value.
	 *************************** 
	 */
	public static double parseDoubleValueAfterColon(String paraString)
			throws Exception {
		// char separator = ' ';
		double tempValue = 0;
		String tempString = new String(paraString);
		String currentString = null;
		// String remainingString = new String(paraString);
		currentString = tempString.substring(tempString.indexOf(":") + 1)
				.trim();
		tempValue = Double.parseDouble(currentString);
		return tempValue;
	}// Of parseDoubleValueAfterColon

	/**
	 *************************** 
	 * Read an int value from a given string after the colon. <br>
	 * 
	 * @param paraString
	 *            The given string.
	 * @return A double value.
	 *************************** 
	 */
	public static int parseIntValueAfterColon(String paraString)
			throws Exception {
		// char separator = ' ';
		int tempValue = 0;
		String tempString = new String(paraString);
		String currentString = null;
		// String remainingString = new String(paraString);
		currentString = tempString.substring(tempString.indexOf(":") + 1)
				.trim();
		tempValue = Integer.parseInt(currentString);
		return tempValue;
	}// Of parseIntValueAfterColon

	/**
	 *************************** 
	 * Compress an int array so that no duplicate elements, no redundant elemnts
	 * exist, and it is in an ascendent order. <br>
	 * 
	 * @param paraIntArray
	 *            The given int array.
	 * @param paraLength
	 *            The effecitive length of the given int array.
	 * @return The constructed array.
	 *************************** 
	 */
	public static int[] compressAndSortIntArray(int[] paraIntArray,
			int paraLength) {
		int[] noDuplicateArray = new int[paraLength];
		int realLength = 0;
		int currentLeast = 0;
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

		int[] compressedArray = new int[realLength];
		for (int i = 0; i < realLength; i++) {
			compressedArray[i] = noDuplicateArray[i];
		}// Of for i

		return compressedArray;
	}// Of compressAndSortIntArray

	
	/**
	 *************************** 
	 * 统计压缩后数组中的每一个元素在原始数组中出现的次数. <br>
	 * 
	 * @param paraOriginalArray
	 *            The original array.
	 * @param paraCompressedArray
	 *            The compressed array.
	 * @return The result array of count number.
	 *************************** 
	 */
	public static  int[] countNumerOfEachElement(int[] paraOriginalArray, int[] paraCompressedArray) {
	
		int[] result = new int[paraCompressedArray.length];
		int count = 0;
		for(int i = 0; i < paraCompressedArray.length; i++){
			count = 0;
			for(int j = 0; j < paraOriginalArray.length; j++){
				if(paraOriginalArray[j] != paraCompressedArray[i]){
					continue;
				}
				else{
					count++;
				}
			}//end of for j
			
			result[i] = count;
			
		}//end of for i
		
		return result;
	}// Of compressAndSortIntArray
	
	/**
	 *************************** 
	 * 统计压缩后数组中的每一个元素在原始数组中出现的次数. <br>
	 * 
	 * @param paraOriginalArray
	 *            The original array.
	 * @param paraCompressedArray
	 *            The compressed array.
	 * @return The result array of count number.
	 *************************** 
	 */
	public static  double[] countProbabilityent(int[] paraOriginalArray, int[] paraCompressedArray) {
	
		double[] result = new double[paraCompressedArray.length];
		int totalNumber = 0;
		int tempCount = 0;
		int[] countEachElementNum = new int[paraCompressedArray.length];
		for(int i = 0; i < paraCompressedArray.length; i++){
			tempCount = 0;
			for(int j = 0; j < paraOriginalArray.length; j++){
				if(paraOriginalArray[j] != paraCompressedArray[i]){
					continue;
				}
				else{
					tempCount++;
				}
			}//end of for j
			totalNumber += tempCount;
			countEachElementNum[i] = tempCount;
		}//end of for i
		
		for(int i = 0; i < paraCompressedArray.length; i++){
			result[i] = (double)countEachElementNum[i]/totalNumber;
		}
		
		return result;
	}// Of compressAndSortIntArray
	
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
	public static double[] compressAndSortDoubleArray(double[] paraIntArray,
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
	 *************************** 
	 * 对数组进行排序，同时记录原始下标. <br>
	 * @param paraIntArray
	 *            The given double array.
	 * @return 排序后的数组，及对应的下标.
	 *************************** 
	 */
	public static double[][] sortDoubleArray(double[] paraIntArray) {
		double[][] resultArray = new double[2][paraIntArray.length];
		boolean[] mark = new boolean[paraIntArray.length];
		for(int i = 0; i < paraIntArray.length; i++){
			mark[i] = false;
		}//end for i
		int realLength = 0;
		double currentLeast = 100000;
		int currentLeastIndex = 0;
		
		for (int i = 0; i < paraIntArray.length; i++) {
			currentLeast = Double.MIN_VALUE;
			//依次扫描，找出当前最小的值及对应的下标
			for (int j = 0; j < paraIntArray.length; j++) {
				if(mark[j] == true)
					continue;
				if (paraIntArray[j] >= currentLeast) {
					currentLeast = paraIntArray[j];
					currentLeastIndex = j;
				}// Of if
			}// Of for j
			//将最小的值及对应的下标保存到二维数组中
			resultArray[0][realLength] = currentLeast;
			resultArray[1][realLength] = currentLeastIndex;
			
			// resultArray[realLength][0] = currentLeast;
			// resultArray[realLength][1] = currentLeastIndex;
			
			//将当前数组元素的状态设置为true
			mark[currentLeastIndex] = true ;
			realLength++;
		}// Of for i
		return resultArray;
	}// Of compressAndSortIntArray
	
	/**
	 *************************** 
	 * 对数组进行排序，同时记录原始下标. <br>
	 * @param paraIntArray
	 *            The given int array.
	 * @return 排序后的数组，及对应的下标.
	 *************************** 
	 */
	public static int[][] sortIntArray(int[] paraIntArray) {
		int[][] resultArray = new int[2][paraIntArray.length];
		boolean[] mark = new boolean[paraIntArray.length];
		for(int i = 0; i < paraIntArray.length; i++){
			mark[i] = false;
		}//end for i
		int realLength = 0;
		int currentLeast = 100000;
		int currentLeastIndex = 0;
		
		for (int i = 0; i < paraIntArray.length; i++) {
			currentLeast = Integer.MAX_VALUE;
			//依次扫描，找出当前最小的值及对应的下标
			for (int j = 0; j < paraIntArray.length; j++) {
				if(mark[j] == true)
					continue;
				if (paraIntArray[j] <= currentLeast) {
					currentLeast = paraIntArray[j];
					currentLeastIndex = j;
				}// Of if
			}// Of for j
			//将最小的值及对应的下标保存到二维数组中
			resultArray[0][realLength] = currentLeast;
			resultArray[1][realLength] = currentLeastIndex;
			
			// resultArray[realLength][0] = currentLeast;
			// resultArray[realLength][1] = currentLeastIndex;
			
			//将当前数组元素的状态设置为true
			mark[currentLeastIndex] = true ;
			realLength++;
		}// Of for i
		return resultArray;
	}// Of compressAndSortIntArray
	
	/**
	 *************************** 
	 * 对有序的数组进行去重. <br>
	 * @param paraDoubleArray
	 *            The given double array.
	 * @return 去重后的数组.
	 *************************** 
	 */
	public static double[] compressDoubleArray(double[] paraDoubleArray) {
		System.out.println("去重前：");
		SimpleTool.printDoubleArray(paraDoubleArray);
		double[] resultArray = new double[paraDoubleArray.length];
		resultArray[0] = paraDoubleArray[0];
		int k=1;
		for(int i = 1; i < paraDoubleArray.length; i++){
			if(paraDoubleArray[i] == resultArray[k-1])
				continue;
			resultArray[k] = paraDoubleArray[i];
			k++;
		}
		
		double[] compressArray = new double[k];
		for (int i = 0; i < k; i++) {
			compressArray[i] = resultArray[i];
		}// Of for i

		return compressArray;
	}//end of compressDoubleArray
	

	
	/**
	 *************************** 
	 * Compress a long array so that no duplicate elements, no redundant elemnts
	 * exist, and it is in an ascendent order. <br>
	 * 
	 * @param paraLongArray
	 *            The given long array.
	 * @param paraLength
	 *            The effecitive length of the given long array.
	 * @return The constructed array.
	 *************************** 
	 */
	public static long[] compressAndSortLongArray(long[] paraLongArray,
			int paraLength) {
		long[] noDuplicateArray = new long[paraLength];
		int realLength = 0;
		long currentLeast = 0;
		int currentLeastIndex = 0;
		for (int i = 0; i < paraLength; i++) {
			if (paraLongArray[i] == Long.MAX_VALUE) {
				continue;
			}

			currentLeast = paraLongArray[i];
			currentLeastIndex = i;

			for (int j = i + 1; j < paraLength; j++) {
				if (paraLongArray[j] < currentLeast) {
					currentLeast = paraLongArray[j];
					currentLeastIndex = j;
				}// Of if
			}// Of for j

			// Swap. The element of [i] should be stored in another place.
			paraLongArray[currentLeastIndex] = paraLongArray[i];

			noDuplicateArray[realLength] = currentLeast;
			realLength++;

			// Don't process this data any more.
			for (int j = i + 1; j < paraLength; j++) {
				if (paraLongArray[j] == currentLeast) {
					paraLongArray[j] = Long.MAX_VALUE;
				}// Of if
			}// Of for j
		}// Of for i

		long[] compressedLongArray = new long[realLength];
		for (int i = 0; i < realLength; i++) {
			compressedLongArray[i] = noDuplicateArray[i];
		}// Of for i

		return compressedLongArray;
	}// Of compressAndSortLongArray

	/**
	 ********************************** 
	 * Subreduct sort according to respective measure.
	 * 
	 * @param paraData
	 *            The data, it may represent a subreduct
	 * @param paraMeasuredValues
	 *            The measured values of the data
	 * @param paraLeft
	 *            The left index.
	 * @param paraRight
	 *            The right index.
	 ********************************** 
	 */
	public static void measureBasedQuickSort(long[] paraData,
			int[] paraMeasuredValues, int paraLeft, int paraRight)
			throws Exception {
		int pivotLoc = 0;
		if (paraLeft < paraRight) {
			pivotLoc = valueBasedLongArrayPartition(paraData,
					paraMeasuredValues, paraLeft, paraRight);
			measureBasedQuickSort(paraData, paraMeasuredValues, paraLeft,
					pivotLoc - 1);// For the left
			measureBasedQuickSort(paraData, paraMeasuredValues, pivotLoc + 1,
					paraRight);// For the right
		}// Of if
	}// Of measureBasedQuickSort

	/**
	 ********************************** 
	 * Invoked only by measureBasedQuickSort.
	 * 
	 * @see #measureBasedQuickSort(long[], double[], int, int)
	 ********************************** 
	 */
	private static int valueBasedLongArrayPartition(long[] paraData,
			int[] paraMeasuredValues, int paraLeft, int paraRight)
			throws Exception {
		double key = paraMeasuredValues[paraLeft];
		int i = paraLeft;
		int j = paraRight + 1;

		while (true) {
			while (paraMeasuredValues[++i] < key && i < paraRight)
				;
			while (paraMeasuredValues[--j] > key)
				;
			if (i >= j)
				break;
			swap(paraMeasuredValues, i, j);
			swap(paraData, i, j);
		}// Of while

		swap(paraMeasuredValues, paraLeft, j);
		swap(paraData, paraLeft, j);
		return j;
	}// Of valueBasedLongArrayPartition

	/**
	 *************************** 
	 * Adjust a long array length.
	 * 
	 * @param paraLongArray
	 *            The given long array.
	 * @param length
	 *            The real length of the given long array.
	 * @return An identical long array.
	 *************************** 
	 */
	public static long[] adjustLongArrayLength(long[] paraLongArray, int length) {
		long[] arrayAim = new long[length];
		for (int i = 0; i < length; i++) {
			arrayAim[i] = paraLongArray[i];
		}// Of for i
		return arrayAim;
	}// Of adjustArrayLength

	/**
	 *************************** 
	 * Copy boolean array
	 * 
	 * @param paramBooleanArray
	 *            The given boolean array.
	 * @return An identical boolean array.
	 *************************** 
	 */
	public static boolean[] copyBooleanArray(boolean[] paramBooleanArray) {
		boolean[] newBooleanArray = new boolean[paramBooleanArray.length];
		for (int i = 0; i < paramBooleanArray.length; i++) {
			newBooleanArray[i] = paramBooleanArray[i];
		}// Of for
		return newBooleanArray;
	}// Of copyBooleanArray

	/**
	 *************************** 
	 * The exponential of an int.
	 * 
	 * @param paraExponential
	 *            the exponential.
	 * @return An identical boolean array.
	 *************************** 
	 */
	public static int exponential(int paraExponential) {
		int results = 1;
		for (int i = 0; i < paraExponential; i++) {
			results *= 2;
		}// Of if
		return results;
	}// Of exponential

	/**
	 *************************** 
	 * The exponential of an int.
	 * 
	 * @param paraExponential
	 *            the exponential.
	 * @return An identical boolean array.
	 *************************** 
	 */
	public static long exponentialLong(int paraExponential) {
		long results = 1;
		for (int i = 0; i < paraExponential; i++) {
			results *= 2;
		}// Of if
		return results;
	}// Of exponentialLong

	/**
	 *************************** 
	 * Who is who's child. Nodes are by integers, and attributes are indicated
	 * by bits.
	 * 
	 * @param paraBits
	 *            How many bits (attributes).
	 * @return An array of children.
	 *************************** 
	 */
	public static boolean[][] children(int paraBits) {
		int sizes = exponential(paraBits);

		boolean[][] children = new boolean[sizes][sizes];
		for (int i = 0; i < sizes; i++) {
			int j = i;
			int m = 1;
			while (j > 0) {
				if (j % 2 == 1) {
					children[i][i - m] = true;
				}// Of if
				j /= 2;
				m *= 2;
			}// Of while
		}// Of for

		return children;
	}// Of children

	/**
	 *************************** 
	 * Convert a long value to a integer array.
	 * 
	 * @param paraLong
	 *            The given long value.
	 * @param paraLength
	 *            The given length of the array.
	 * @return An integer array to indicate which positions (bits) are included.
	 *************************** 
	 */
	public static int[] longToIntArray(long paraLong, int paraLength) {
		int[] result = new int[paraLength];
		/*
		 * if (paraLong < 0) { return null; //No need to throw an exception. }
		 */
		int currentIndex = 0;
		long tempLong = paraLong;
		while (tempLong > 0) {
			if (tempLong % 2 == 1) {
				result[currentIndex] = 1;
			}

			tempLong /= 2;
			currentIndex++;
		}// Of while

		return result;
	}// Of longToIntArray

	/**
	 *************************** 
	 * Convert a long value to a boolean array.
	 * 
	 * @param paraLong
	 *            The given long value.
	 * @param paraLength
	 *            The given length of the array.
	 * @return A boolean array to indicate which positions (bits) are included.
	 *************************** 
	 */
	public static boolean[] longToBooleanArray(long paraLong, int paraLength) {
		long tempLong = paraLong;
		boolean[] returnArray = new boolean[paraLength];
		for (int i = 0; i < paraLength; i++) {
			if (tempLong % 2 == 1) {
				returnArray[i] = true;
			} else {
				returnArray[i] = false;
			}// Of if

			tempLong /= 2;
		}// Of for i

		return returnArray;
	}// Of longToBooleanArray

	/**
	 *************************** 
	 * Convert a boolean array to a long value.
	 * 
	 * @param paraBooleanArray
	 *            The given boolean array.
	 * @return A long to indicate which positions (bits) are included.
	 *************************** 
	 */
	public static long booleanArrayToLong(boolean[] paraBooleanArray)
			throws Exception {
		if (paraBooleanArray.length > 63) {
			throw new Exception(
					"Cannot support the array with length more than 63.");
		}// Of if
		long resultLong = 0;
		long currentPositionValue = 1;
		for (int i = 0; i < paraBooleanArray.length; i++) {
			if (paraBooleanArray[i]) {
				resultLong += currentPositionValue;
			}// Of if

			currentPositionValue *= 2;
		}// Of for i

		return resultLong;
	}// Of booleanArrayToLong

	/**
	 *************************** 
	 * Compute the attribute subset size.
	 * 
	 * @param paraLong
	 *            The given long value.
	 * @param paraLength
	 *            The size of all attributes.
	 * @return the size of the attribute subset.
	 *************************** 
	 */
	public static int attributeSubsetSize(long paraLong, int paraLength) {
		long tempLong = paraLong;
		int returnSize = 0;
		for (int i = 0; i < paraLength; i++) {
			if (tempLong % 2 == 1) {
				returnSize++;
			}// Of if

			tempLong /= 2;
		}// Of for i

		return returnSize;
	}// Of attributeSubsetSize

	/**
	 *************************** 
	 * Compute the attribute subset size.
	 * 
	 * @param paraBooleanArray
	 *            The given boolean array.
	 * @return the size of the attribute subset.
	 *************************** 
	 */
	public static int attributeSubsetSize(boolean[] paraBooleanArray) {
		int returnSize = 0;
		for (int i = 0; i < paraBooleanArray.length; i++) {
			if (paraBooleanArray[i]) {
				returnSize++;
			}// Of if
		}// Of for i

		return returnSize;
	}// Of attributeSubsetSize

	/**
	 *************************** 
	 * Convert an integer value to a boolean array.
	 * 
	 * @param paraInt
	 *            The given long value.
	 * @param paraLength
	 *            The given length of the array.
	 * @return A boolean array to indicate which positions (bits) are included.
	 *************************** 
	 */
	public static boolean[] intToBooleanArray(int paraInt, int paraLength) {
		long tempInt = paraInt;
		boolean[] returnArray = new boolean[paraLength];
		for (int i = 0; i < paraLength; i++) {
			if (tempInt % 2 == 1) {
				returnArray[i] = true;
			} else {
				returnArray[i] = false;
			}// Of if

			tempInt /= 2;
		}// Of for i

		return returnArray;
	}// Of intToBooleanArray

	/**
	 *************************** 
	 * Convert a boolean array to an int value.
	 * 
	 * @param paraBooleanArray
	 *            The given boolean array.
	 * @return An integer to indicate which positions (bits) are included.
	 *************************** 
	 */
	public static int booleanArrayToInt(boolean[] paraBooleanArray)
			throws Exception {
		if (paraBooleanArray.length > 31) {
			throw new Exception(
					"Cannot support the array with length more than 31.");
		}// Of if

		int resultInt = 0;
		int currentPositionValue = 1;
		for (int i = 0; i < paraBooleanArray.length; i++) {
			if (paraBooleanArray[i]) {
				resultInt += currentPositionValue;
			}// Of if

			currentPositionValue *= 2;
		}// Of for i

		return resultInt;
	}// Of booleanArrayToInt

	/**
	 *************************** 
	 * Print an int array, simply for test.
	 * 
	 * @param paraIntArray
	 *            The given int array.
	 *************************** 
	 */
	public static void printIntArray(int[] paraIntArray) {
		if (paraIntArray.length == 0) {
			System.out.println("This is an empty int array.");
			return;
		} else {
			System.out.print("This is an int array: ");
		}
		for (int i = 0; i < paraIntArray.length; i++) {
			System.out.print("" + paraIntArray[i] + "\t");
		}// Of for i
		System.out.println();
	}// Of paraIntArray

	/**
	 ************************* 
	 * printIntMatrix
	 ************************* 
	 */
	public static void printIntMatrix(int[][] paraMatrix) {
		for (int i = 0; i < paraMatrix.length; i++) {
			for (int j = 0; j < paraMatrix[i].length; j++) {
				System.out.print("" + paraMatrix[i][j] + ", ");
			}// Of for j
			System.out.println();
		}// Of for i
	}// Of printIntMatrix
	/**
	 ************************* 
	 * printDoubleMatrix
	 ************************* 
	 */
	public static void printDoubleMatrix(double[][] paraMatrix) {
		for (int i = 0; i < paraMatrix.length; i++) {
			for (int j = 0; j < paraMatrix[i].length; j++) {
				System.out.print("" + paraMatrix[i][j] + ", ");
			}// Of for j
			System.out.println();
		}// Of for i
	}// Of printIntMatrix
	/**
	 ************************* 
	 * copyIntMatrix
	 * 
	 * @author Fan Min 2014/10/15
	 ************************* 
	 */
	public static int[][] copyIntMatrix(int[][] paraMatrix) {
		int[][] returnMatrix = new int[paraMatrix.length][paraMatrix[0].length];
		for (int i = 0; i < paraMatrix.length; i++) {
			for (int j = 0; j < paraMatrix[i].length; j++) {
				returnMatrix[i][j] = paraMatrix[i][j];
			}// Of for j
		}// Of for i

		return returnMatrix;
	}// Of copyIntMatrix

	/**
	 ************************* 
	 * copyDoubleMatrix
	 * 
	 * @author Fan Min 2014/10/15
	 ************************* 
	 */
	public static double[][] copyDoubleMatrix(double[][] paraMatrix) {
		double[][] returnMatrix = new double[paraMatrix.length][paraMatrix[0].length];
		for (int i = 0; i < paraMatrix.length; i++) {
			for (int j = 0; j < paraMatrix[i].length; j++) {
				returnMatrix[i][j] = paraMatrix[i][j];
			}// Of for j
		}// Of for i

		return returnMatrix;
	}// Of copyIntMatrix
	

	/**
	 ************************* 
	 * copyLink
	 * 
	 * @author Fan Min 2014/10/15
	 ************************* 
	 */
	public static List<Integer> copyLink(List<Integer> paraLink) {
		List<Integer> returnList = new LinkedList();
		for (int i = 0; i < paraLink.size(); i++) {
			returnList.add(i,paraLink.get(i));
			
		}// Of for i

		return returnList;
	}// Of copyIntMatrix
	
	
	
	/**
	 ************************* 
	 * copyDoubleArray
	 * 
	 * @author Fan Min 2014/10/15
	 ************************* 
	 */
	public static double[] copyDoubleArray(double[] paraMatrix) {
		double[] returnMatrix = new double[paraMatrix.length];
		for (int i = 0; i < paraMatrix.length; i++) {
			returnMatrix[i] = paraMatrix[i];
			
		}// Of for i

		return returnMatrix;
	}// Of copyIntMatrix
	
	/**
	 ************************* 
	 * copyDoubleArray
	 * 
	 * @author Fan Min 2014/10/15
	 ************************* 
	 */
	public static int[] copyIntArray(int[] paraMatrix) {
		int[] returnMatrix = new int[paraMatrix.length];
		for (int i = 0; i < paraMatrix.length; i++) {
			returnMatrix[i] = paraMatrix[i];
			
		}// Of for i

		return returnMatrix;
	}// Of copyIntMatrix
	/**
	 *************************** 
	 * Print a long array, simply for test.
	 * 
	 * @param paraLongArray
	 *            The given long array.
	 *************************** 
	 */
	public static void printLongArray(long[] paraLongArray) {
		for (int i = 0; i < paraLongArray.length; i++) {
			System.out.print("" + paraLongArray[i] + "\t");
		}// Of for i
	}// Of printLongArray

	/**
	 *************************** 
	 * Print a long array, zero is not printed. Simply for test.
	 * 
	 * @param paraLongArray
	 *            The given long array.
	 *************************** 
	 */
	public static void printLongArrayNoZero(long[] paraLongArray) {
		for (int i = 0; i < paraLongArray.length; i++) {
			if (paraLongArray[i] == 0)
				continue;
			System.out.print("" + paraLongArray[i] + "\t");
		}// Of for i
	}// Of printLongArrayNoZero

	/**
	 ************************* 
	 * Print all reducts.
	 * 
	 * @param paraAllReducts
	 *            The given array to print.
	 ************************* 
	 */
	public static void printAllReducts(boolean[][] paraAllReducts) {
		for (int i = 0; i < paraAllReducts.length; i++) {
			System.out.println();
			for (int j = 0; j < paraAllReducts[0].length; j++) {
				if (paraAllReducts[i][j]) {
					System.out.print("" + 1 + ",");
				} else {
					System.out.print("" + 0 + ",");
				}
			}// Of for j
		}// Of for i
	}// Of printAllReducts

	/**
	 ************************* 
	 * Print a boolean array.
	 ************************* 
	 */
	public static void printBooleanArray(boolean[] paraBooleanArray) {
		for (int i = 0; i < paraBooleanArray.length; i++) {
			if (paraBooleanArray[i]) {
				System.out.print("" + 1 + ",");
			} else {
				System.out.print("" + 0 + ",");
			}

		}// Of for i
		System.out.println();
	}// Of printBooleanArray

	/**
	 ************************* 
	 * Print a boolean matrix.
	 ************************* 
	 */
	public static void printBooleanMatrix(boolean[][] paraBooleanMatrix) {
		for (int i = 0; i < paraBooleanMatrix.length; i++) {
			for (int j = 0; j < paraBooleanMatrix[i].length; j++) {
				if (paraBooleanMatrix[i][j]) {
					System.out.print("1,");
				} else {
					System.out.print("0,");
				}// Of if
			}// Of for j
			System.out.println();
		}// Of for i
		System.out.println();
	}// Of printBooleanMatrix

	/**
	 ************************* 
	 * Print a double array.
	 ************************* 
	 */
	public static void printDoubleArray(double[] paraDoubleArray) {
		for (int i = 0; i < paraDoubleArray.length; i++) {
			System.out.print(paraDoubleArray[i] + " ");
		}// Of for i
		System.out.println();
	}// Of printAllReducts

	/**
	 ********************************** 
	 * Swap two value in a double array.
	 * 
	 * @param paraDoubleArray
	 *            The given double array.
	 * @param src
	 *            The first index of the double array.
	 * @param dest
	 *            The second index of the double array.
	 ********************************** 
	 */
	public static void swap(double[] paraDoubleArray, int src, int dest)
			throws Exception {
		double tempDoubleArray = 0;
		tempDoubleArray = paraDoubleArray[src];
		paraDoubleArray[src] = paraDoubleArray[dest];
		paraDoubleArray[dest] = tempDoubleArray;
	}// Of swap

	/**
	 ********************************** 
	 * Swap two value in a long array.
	 * 
	 * @param paraLongArray
	 *            The given long array.
	 * @param src
	 *            The first index of the long array.
	 * @param dest
	 *            The second index of the long array.
	 ********************************** 
	 */
	public static void swap(long[] paraLongArray, int src, int dest)
			throws Exception {
		long tempIntArray = 0;
		tempIntArray = paraLongArray[src];
		paraLongArray[src] = paraLongArray[dest];
		paraLongArray[dest] = tempIntArray;
	}// Of swap

	/**
	 ********************************** 
	 * Swap two value in a long array.
	 * 
	 * @param paraLongArray
	 *            The given long array.
	 * @param src
	 *            The first index of the long array.
	 * @param dest
	 *            The second index of the long array.
	 ********************************** 
	 */
	public static void swap(int[] paraLongArray, int src, int dest)
			throws Exception {
		int tempIntArray = 0;
		tempIntArray = paraLongArray[src];
		paraLongArray[src] = paraLongArray[dest];
		paraLongArray[dest] = tempIntArray;
	}// Of swap

	/**
	 ********************************** 
	 * Long array to boolean matrix.
	 * 
	 * @param paraLongArray
	 *            The given long array.
	 * @param paraLength
	 *            The length of each long.
	 * @return the boolean matrix
	 ********************************** 
	 */
	public static boolean[][] longArrayToBooleanMatrix(long[] paraLongArray,
			int paraLength) {
		boolean[] availableAttribute = new boolean[paraLength];
		boolean[][] paraAllSubreducts = new boolean[paraLongArray.length][paraLength];

		for (int i = 0; i < paraLongArray.length; i++) {
			availableAttribute = longToBooleanArray(paraLongArray[i],
					paraLength);
			for (int j = 0; j < paraLength; j++) {
				if (availableAttribute[j]) {
					paraAllSubreducts[i][j] = true;
				}// Of if
			}// Of for j
		}// of for i
		return paraAllSubreducts;
	}// Of swap

	/**
	 * Write a message to a new file.
	 * 
	 * @paraFilename The given filename.
	 * @paraMessage The givean message string. public static void
	 *              writeFile(String paraFilename,String paraMessage) throws
	 *              Exception{
	 * 
	 *              File resultFile = new File(paraFilename); if
	 *              (resultFile.exists()) { resultFile.delete(); }
	 *              resultFile.createNewFile(); PrintWriter writer = new
	 *              PrintWriter(new FileOutputStream(resultFile));
	 *              writer.print(paraMessage); writer.flush(); writer.close();
	 *              }//Of writeFile
	 */

	/**
	 ********************************** 
	 * Is the first set a subset of the second one.
	 * 
	 * @param paraFirstSet
	 *            The first set in long.
	 * @param paraSecondSet
	 *            The second set in long.
	 * @param paraAttributes
	 *            Number of attributes.
	 * @return is the first set a subset of the second one?
	 ********************************** 
	 */
	public static boolean isSubset(long paraFirstSet, long paraSecondSet,
			int paraAttributes) throws Exception {
		boolean[] firstSetBooleanArray = longToBooleanArray(paraFirstSet,
				paraAttributes);
		boolean[] secondSetBooleanArray = longToBooleanArray(paraSecondSet,
				paraAttributes);
		return isSubset(firstSetBooleanArray, secondSetBooleanArray);
	}// Of isSubset

	/**
	 ********************************** 
	 * Is the first set a subset of the second one.
	 * 
	 * @param paraFirstSet
	 *            The first set in int array.
	 * @param paraSecondSet
	 *            The second set in int array.
	 * @return is the first set a subset of the second one?
	 ********************************** 
	 */
	public static boolean isSubset(int[] paraFirstSet, int[] paraSecondSet)
			throws Exception {
		Common.computationTime++;
		if ((paraFirstSet.length > paraSecondSet.length)
				|| (paraFirstSet[paraFirstSet.length - 1] > paraSecondSet[paraSecondSet.length - 1])
				|| paraSecondSet[paraSecondSet.length - 1] > 100) {
			return false;
		}// Of if

		int indexInTheFirstSet = 0;
		int indexInTheSecondSet = 0;
		while (indexInTheFirstSet < paraFirstSet.length) {
			Common.computationTime++;
			if (paraFirstSet[indexInTheFirstSet] > paraSecondSet[indexInTheSecondSet]) {
				indexInTheSecondSet++;
			} else if (paraFirstSet[indexInTheFirstSet] < paraSecondSet[indexInTheSecondSet]) {
				return false;
			} else {
				indexInTheFirstSet++;
				indexInTheSecondSet++;
			}// Of if
		}// Of while

		return true;
	}// Of isSubset

	/**
	 ********************************** 
	 * Is the first set a subset of the second one.
	 * 
	 * @param paraFirstSet
	 *            The first set in boolean array.
	 * @param paraSecondSet
	 *            The second set in boolean array.
	 * @return is the first set a subset of the second one?
	 ********************************** 
	 */
	public static boolean isSubset(boolean[] paraFirstSet,
			boolean[] paraSecondSet) throws Exception {
		if (paraFirstSet.length != paraFirstSet.length) {
			throw new Exception(
					"Error occurred in SimpleTool.isSubset(). Boolean arrays should"
							+ " have the same length");
		}// Of if

		for (int i = 0; i < paraFirstSet.length; i++) {
			if (paraFirstSet[i] && !paraSecondSet[i]) {
				return false;
			}// Of if
		}// Of for i
		return true;
	}// Of isSubset

	/**
	 ********************************** 
	 * Convert a long value to a bit string.
	 * 
	 * @param paraLong
	 *            The given long value.
	 * @return the bit string.
	 ********************************** 
	 */
	public static String longToBitString(long paraLong) {
		String returnString = "";
		while (paraLong > 0) {
			if (paraLong % 2 == 1) {
				returnString = "1" + returnString;
			} else {
				returnString = "0" + returnString;
			}// Of if
			paraLong /= 2;
		}// Of while
		return returnString;
	}// Of longToBitString

	/**
	 ********************************** 
	 * Return the size of the subset. It is the true values in the array.
	 * 
	 * @param paraSubset
	 *            A subset with the form of a boolean array.
	 * @return The size.
	 ********************************** 
	 */
	public static int getSubsetSize(boolean[] paraSubset) {
		int tempCounter = 0;
		for (int i = 0; i < paraSubset.length; i++) {
			if (paraSubset[i]) {
				tempCounter++;
			}
		}// Of if
		return tempCounter;
	}// Of getSubsetSize

	/**
	 ********************************** 
	 * Return the size of the subset. It is the true values in the array.
	 * 
	 * @param paraSubset
	 *            A subset with the form of a long.
	 * @return The size.
	 ********************************** 
	 */
	public static int getSubsetSize(long paraSubset) {
		long tempLong = paraSubset;
		int tempCounter = 0;
		while (tempLong > 0) {
			if (tempLong % 2 == 1) {
				tempCounter++;
			}// Of if

			tempLong /= 2;
		}// Of for i

		return tempCounter;
	}// Of getSubsetSize

	/**
	 *************************** 
	 * Check Who is who's subset.
	 * 
	 * @param paraFirstSet
	 *            the first set in long.
	 * @param paraSecondSet
	 *            the second set in long.
	 * @param paraNumberOfConditions
	 *            the number of conditions.
	 * @return '0' means no relationship, '1' means the second is the subset,
	 *         '2' means the first is the subset, '3' means equal.
	 *************************** 
	 */
	public static char subSetCheck(long paraFirstSet, long paraSecondSet,
			int paraNumberOfConditions) {
		boolean supportPositive = true;
		boolean supportNegative = true;

		int[] firstSubset = SimpleTool.longToIntArray(paraFirstSet,
				paraNumberOfConditions);
		int[] secondSubset = SimpleTool.longToIntArray(paraSecondSet,
				paraNumberOfConditions);

		for (int i = 0; i < paraNumberOfConditions; i++) {
			if (firstSubset[i] - secondSubset[i] > 0) {
				supportPositive = false;
			} else if (firstSubset[i] - secondSubset[i] < 0) {
				supportNegative = false;
			}// Of if
		}// Of for i

		if (!supportPositive && !supportNegative) {
			return '0'; // The two sets with not inclusion raletionship
		}// Of if
		if (!supportPositive) {
			return '1'; // The second set is the child of the first set.
		}// Of if
		if (!supportNegative) {
			return '2'; // The first set is the child of the second set.
		}// Of if

		return '3'; // The two sets is equal to each other.
	}// Of subSetCheck

	/**
	 ********************************** 
	 * Convert a boolean matrix to string
	 ********************************** 
	 */
	public static String booleanMatrixToString(boolean[][] tempMatrix) {
		String tempString = "";
		for (int i = 0; i < tempMatrix.length; i++) {
			for (int j = 0; j < tempMatrix[0].length; j++) {
				if (tempMatrix[i][j]) {
					tempString += " 1";
				} else {
					tempString += " 0";
				}// Of if
			}// Of for j
			tempString += "\r\n";
		}// Of for i
		return tempString;
	}// Of booleanMatrixToString

	/**
	 ********************************** 
	 * Generate a boolean array to divide a dataset in two
	 * 
	 * @param paraDatasetSize
	 *            the dataset size
	 * @param paraPercentage
	 *            the percentage of the first subset
	 * @throws Exception
	 ********************************** 
	 */
	public static boolean[] generateBooleanArrayForDivision(
			int paraDatasetSize, double paraPercentage) throws Exception {
		double percentageLowerBound = 1.0 / paraDatasetSize;
		double percentageUpperBound = 1 - percentageLowerBound;
		if ((paraPercentage < percentageLowerBound)
				|| (paraPercentage > percentageUpperBound)) {
			throw new Exception(
					"Error occurred in SimpleTool.generateForPartitionInTwo().\r\n"
							+ "The valid bound of the percentage should be in ["
							+ percentageLowerBound + ", "
							+ percentageUpperBound + "].");
		}// Of if

		boolean[] tempBooleanArray = new boolean[paraDatasetSize];
		int firstSetSize = (int) (paraDatasetSize * paraPercentage);
		int secondSetSize = paraDatasetSize - firstSetSize;

		int firstSetCurrentSize = 0;
		int secondSetCurrentSize = 0;

		int tempInt;
		for (int i = 0; i < paraDatasetSize; i++) {

			tempInt = random.nextInt(paraDatasetSize);
			// System.out.println(tempInt);
			if (tempInt < firstSetSize) {
				tempBooleanArray[i] = true;
				firstSetCurrentSize++;
			} else {
				tempBooleanArray[i] = false;
				secondSetCurrentSize++;
			}// Of if

			// Keep the remaining part to false (the default value)
			if (firstSetCurrentSize >= firstSetSize) {
				break;
			}// Of if

			// Set the remaining part to true
			if (secondSetCurrentSize >= secondSetSize) {
				for (i++; i < paraDatasetSize; i++) {
					tempBooleanArray[i] = true;
				}// Of for i
				break;
			}// Of if
		}// Of for i

		return tempBooleanArray;
	}// Of generateBooleanArrayForDivision

	/**
	 ********************************** 
	 * Revert a boolean array.
	 * 
	 * @param paraArray
	 *            the given array
	 * @return reverted array.
	 ********************************** 
	 */
	public static boolean[] revertBooleanArray(boolean[] paraArray) {
		boolean[] tempArray = new boolean[paraArray.length];
		for (int i = 0; i < paraArray.length; i++) {
			tempArray[i] = !paraArray[i];
		}// Of for i
		return tempArray;
	}// Of revertBooleanArray

	/**
	 ********************************** 
	 * Remove supersets, used to obtain reducts.
	 * 
	 * @param paraSets
	 *            the subsets with each one represented by a long value.
	 * @param paraNumberOfConditions
	 *            the number of condition of the dataset
	 * @return a set of reducts without superflous attributes.
	 ********************************** 
	 */
	public static long[] removeSupersets(long[] paraSets,
			int paraNumberOfConditions) throws Exception {
		char check = 0;

		// Check who is who's child.
		for (int i = 0; i < paraSets.length - 1; i++) {
			if (paraSets[i] == -1) {
				continue;
			}// Of if
			for (int j = i + 1; j < paraSets.length; j++) {
				if (paraSets[j] == -1) {
					continue;
				}// Of if

				check = SimpleTool.subSetCheck(paraSets[i], paraSets[j],
						paraNumberOfConditions);
				if (check == '1') {
					paraSets[i] = -1;
					break;
				}// Of if

				if (check == '2') {
					paraSets[j] = -1;
				}// Of if
			}// Of for j
		}// Of for i

		int numberRemainingSubsets = 0;
		for (int i = 0; i < paraSets.length; i++) {
			if (paraSets[i] > 0) {
				numberRemainingSubsets++;
			}// Of if
		}// Of for i

		long[] returnArray = new long[numberRemainingSubsets];
		int m = 0;
		for (int i = 0; i < paraSets.length; i++) {
			if (paraSets[i] > 0) {
				returnArray[m] = paraSets[i];
				m++;
			}// Of if
		}// Of for i

		return returnArray;
	}// Of removeSupersets

	/**
	 ********************************** 
	 * Set intersection. The sets should be sorted.
	 * 
	 * @param paraFirstSet
	 *            The first set indicated by numbers for indices.
	 * @param paraSecondSet
	 *            The second set indicated by numbers for indices.
	 * @return a set for the intersection.
	 ********************************** 
	 */
	public static int[] setIntersection(int[] paraFirstSet, int[] paraSecondSet)
			throws Exception {
		int[] emptyArray = new int[0];
		if ((paraFirstSet.length == 0) || (paraSecondSet.length == 0)) {
			return emptyArray;
		}

		if (!isAscendingOrder(paraFirstSet)) {
			throw new Exception(
					"The first array is not in an ascending order: "
							+ intArrayToString(paraFirstSet, ','));
		}
		if (!isAscendingOrder(paraSecondSet)) {
			throw new Exception(
					"The second array is not in an ascending order: "
							+ intArrayToString(paraSecondSet, ','));
		}

		int[] tempSet = new int[paraFirstSet.length + paraSecondSet.length];
		int tempLength = 0;
		int firstIndex = 0;
		int secondIndex = 0;
		while ((firstIndex < paraFirstSet.length)
				&& (secondIndex < paraSecondSet.length)) {
			Common.computationTime++;
			if (paraFirstSet[firstIndex] < paraSecondSet[secondIndex]) {
				firstIndex++;
			} else if (paraFirstSet[firstIndex] > paraSecondSet[secondIndex]) {
				secondIndex++;
			} else {
				tempSet[tempLength] = paraFirstSet[firstIndex];
				firstIndex++;
				secondIndex++;
				tempLength++;
			}// Of if
		}// Of while

		int[] resultSet = new int[tempLength];
		for (int i = 0; i < tempLength; i++) {
			resultSet[i] = tempSet[i];
		}// Of for

		return resultSet;
	}// Of setIntersection

	/**
	 ********************************** 
	 * Is the given array in an ascending order?
	 * 
	 * @param paraArray
	 *            The int array.
	 * @return true if it is ascending (equal values permitted).
	 ********************************** 
	 */
	public static boolean isAscendingOrder(int[] paraArray) {
		for (int i = 0; i < paraArray.length - 1; i++) {
			if (paraArray[i] > paraArray[i + 1]) {
				return false;
			}// Of if
		}// Of for
		return true;
	}// Of isAscendingOrder

	/**
	 *************************** 
	 * Judge whether or not the given boolean matrix is upper triangle one.
	 * 
	 * @param paraMatrix
	 *            The given boolean matrix.
	 * @return The constructed String.
	 *************************** 
	 */
	public static boolean isUpperTriangleBooleanMatrix(boolean[][] paraMatrix) {
		for (int i = 0; i < paraMatrix.length; i++) {
			for (int j = 0; j <= i; j++) {
				if (paraMatrix[i][j]) {
					return false;
				}// Of if
			}// Of for j
		}// Of for i

		return true;
	}// Of isUpperTriangleBooleanMatrix

	/**
	 ************************* 
	 * Print a boolean array as IDs.
	 ************************* 
	 */
	public static void printBooleanArrayAsID(boolean[] paraBooleanArray) {
		for (int i = 0; i < paraBooleanArray.length; i++) {
			if (paraBooleanArray[i]) {
				System.out.print(i + ", ");
			}// Of if
		}// Of for i
		System.out.println();
	}// Of printBooleanArrayAsID

	/**
	 ************************* 
	 * Computes the entropy of the array. Should be thrown to SimpleTool.java
	 * 
	 * @param paraArray
	 *            The int array with values starting from 0.
	 * @return the entropy of the data's distribution
	 ************************* 
	 */
	public static double entropy(int[] paraArray) {
		int tempNumValues = 0;

		// 找出决策属性值中最大的那一个
		for (int i = 0; i < paraArray.length; i++) {
			if (tempNumValues < paraArray[i]) {
				tempNumValues = paraArray[i];
			}// Of if
		}// of for i
		tempNumValues++;

		// 对于决策属性值中的每一种值，均统计其个数，存储到valueCounts[]数组中
		double[] valueCounts = new double[tempNumValues];
		for (int i = 0; i < paraArray.length; i++) {
			valueCounts[paraArray[i]]++;
		}// Of for i

		// 计算决策属性值中每一个的信息熵
		double entropy = 0;
		for (int i = 0; i < tempNumValues; i++) {
			if (valueCounts[i] > 0) {
				entropy -= valueCounts[i] * Utils.log2(valueCounts[i]);
			}
		}// Of for i
		entropy /= (double) paraArray.length;
		entropy += Utils.log2(paraArray.length);// important!

		return entropy;
	}// Of entropy

	
	/**
	 ************************* 
	 * Computes the entropy of an array.
	 * Elements in the array should not be smaller than 0.
	 * 
	 * @param paraIntArray
	 ************************* 
	 *//*
	public static double entropy(int[] paraArray) {
		double resultEntropy = 0;
		int tempTotal = 0;
		for (int i = 0; i < paraArray.length; i++) {
			tempTotal += paraArray[i];
			if (paraArray[i] > 0) {
				resultEntropy -= paraArray[i] * Utils.log2(paraArray[i]);
			}//Of if
		}// Of for j
		resultEntropy /= tempTotal;
		resultEntropy += Utils.log2(tempTotal);

		return resultEntropy;
	}// Of classEntropy
*/	
	
	/**
	 ************************* 
	 * The information loss of merging two values.
	 * 
	 * @param paraMatrix
	 *            The int matrix for the internal representation of a
	 *            conditional attribute and the decision.
	 * @param paraFirst
	 *            The first value to be merged.
	 * @param paraSecond
	 *            The second value to be merged.
	 * @return the information loss of merging these two values.
	 * @author Fan Min 2014/10/14
	 ************************* 
	 */
	public static double informationLoss(int[][] paraMatrix, int paraFirst,
			int paraSecond) {
		double tempSeparatedEntropy = conditionalEntropyOneAttribute(paraMatrix);
		// double tempSeparatedEntropy = conditionalEntropy(paraMatrix);
		// System.out.println("Merging " + paraFirst + " and " + paraSecond);
		//System.out.println("Conditional entropy for the initial data is: "
				//+ tempSeparatedEntropy);
		//printIntMatrix(paraMatrix);

		int[][] tempMergedMatrix = new int[paraMatrix.length][2];
		// Copy and revise
		for (int i = 0; i < paraMatrix.length; i++) {
			if (paraMatrix[i][0] == paraSecond) {
				tempMergedMatrix[i][0] = paraFirst;
			} else {
				tempMergedMatrix[i][0] = paraMatrix[i][0];
			}// Of if
			tempMergedMatrix[i][1] = paraMatrix[i][1];
		}// Of for i

		double tempMergedEntropy = conditionalEntropyOneAttribute(tempMergedMatrix);
		// double tempMergedEntropy = conditionalEntropy(tempMergedMatrix);

		// System.out.println("Conditional entropy for the merged data is: "
				// + tempMergedEntropy);
		//printIntMatrix(tempMergedMatrix);
		return (tempMergedEntropy - tempSeparatedEntropy);
	}// Of informationLoss

	/**
	 ************************* 
	 * Computes the entropy of the second column with respect the first one.
	 * Should be thrown to SimpleTool.java
	 * 
	 * @param paraMatrix
	 *            The int matrix with values starting from 0.
	 * @return the entropy of the data's distribution
	 * @author Liuying Wen 2014/11/18
	 ************************* 
	 */
	public static double conditionalEntropy(int[][] paraMatrix) {
		int tempNumConditionalAttributeValues = 0; // 存储条件属性值中的最大值
		int tempNumDecisionAttributeValues = 0;// 存储决策属性值中的最大值
		double tempTotalEntropy = 0;// 存储条件属性值中的最大值

		System.out.println("--Compute conditional entropy for");
		printIntMatrix(paraMatrix);

		// 第一步. 遍历paraMatrix，找到条件属性值和决策属性值的最大值
		for (int i = 0; i < paraMatrix.length; i++) {
			if (tempNumConditionalAttributeValues < paraMatrix[i][0]) {
				tempNumConditionalAttributeValues = paraMatrix[i][0];
			}
			if (tempNumDecisionAttributeValues < paraMatrix[i][1]) {
				tempNumDecisionAttributeValues = paraMatrix[i][1];
			}
		}
		tempNumConditionalAttributeValues++;
		tempNumDecisionAttributeValues++;

		// 第二步. 对于条件属性值的每一种类型的值（Block），计算它的信息熵
		double[] tempEntropyArray = new double[tempNumConditionalAttributeValues];
		double[] tempDecisionAttributeValueCount = new double[tempNumDecisionAttributeValues];
		double tempBlockEntropy = 0;
		int tempBlockSize = 0;

		for (int i = 0; i <= tempNumConditionalAttributeValues; i++) {
			// 初始化
			tempBlockSize = 0;
			for (int j = 0; j < tempNumDecisionAttributeValues; j++) {
				tempDecisionAttributeValueCount[j] = 0;
			}// of for j

			// 遍历paraMatrix矩阵，统计条件属性值中每一块的信息熵
			// 统计当前这个Block中每一个决策属性值的个数
			for (int j = 0; j < paraMatrix.length; j++) {
				if (paraMatrix[j][0] == i) {
					tempBlockSize++;
					tempDecisionAttributeValueCount[paraMatrix[j][1]]++;
				}// of if
			}// of for j

			// 计算当前块的信息熵
			tempBlockEntropy = 0;
			for (int j = 0; j < tempNumDecisionAttributeValues; j++) {
				if (tempDecisionAttributeValueCount[j] > 0) {
					tempBlockEntropy -= tempDecisionAttributeValueCount[j]
							* Utils.log2(tempDecisionAttributeValueCount[j]);
				}// of if
			}// of for j

			if (tempBlockSize == 0) {
				continue;
			}// of if

			// 为何还要加上tempBlockSize * Utils.log2(tempBlockSize)？？？？？
			tempBlockEntropy += tempBlockSize * Utils.log2(tempBlockSize);
			tempEntropyArray[i] = tempBlockEntropy;

			tempTotalEntropy += tempBlockEntropy / paraMatrix.length;
		}// of for i
		return tempTotalEntropy;
	}// of conditionalEntropy

	/**
	 ************************* 
	 * Computes the entropy of the second column with respect the first one.
	 * Should be thrown to SimpleTool.java
	 * 
	 * @param paraMatrix
	 *            The int matrix with values starting from 0.
	 * @return the entropy of the data's distribution
	 * @author Fan Min 2014/10/09
	 ************************* 
	 */
	public static double conditionalEntropyOneAttribute(int[][] paraMatrix) {
		int tempNumConditionalValues = 0;
		int tempNumClassValues = 0;
		double tempTotalEntropy = 0;
		/*
		 * System.out.println("-------------Compute conditional entropy for");
		 * printIntMatrix(paraMatrix);
		 */

		// Step 1. Count the number of conditional/class values.
		// System.out.println(paraMatrix.length);
		for (int i = 0; i < paraMatrix.length; i++) {
			if (tempNumConditionalValues < paraMatrix[i][0]) {
				tempNumConditionalValues = paraMatrix[i][0];
			}// Of if

			if (tempNumClassValues < paraMatrix[i][1]) {
				tempNumClassValues = paraMatrix[i][1];
			}// Of if
		}// of for i
		tempNumConditionalValues++;
		tempNumClassValues++;
		// System.out.println("--------");
		// System.out.println(tempNumConditionalValues+","+tempNumClassValues);
		

		// Step 2. The entropy for each block
		double[] tempEntropyArray = new double[tempNumConditionalValues];
		double[] tempValueCounts = new double[tempNumClassValues];
		double tempEntropy;
		double tempBlockSize = 0;

		for (int i = 0; i < tempNumConditionalValues; i++) {
			// Reinitialize
			tempBlockSize = 0;
			for (int j = 0; j < tempNumClassValues; j++) {
				tempValueCounts[j] = 0;
			}// of for j

			// Count
			for (int j = 0; j < paraMatrix.length; j++) {
				if (paraMatrix[j][0] == i) {
					tempBlockSize++;
					tempValueCounts[paraMatrix[j][1]]++;
				}// Of if
			}// of for j

			// System.out.println("tempBlockSize = " + tempBlockSize);
			// SimpleTool.printDoubleArray(tempValueCounts);
			// Compute entropy for this block
			tempEntropy = 0;
			for (int j = 0; j < tempNumClassValues; j++) {
				if (tempValueCounts[j] > 0) {
					tempEntropy -= tempValueCounts[j]
							* Utils.log2(tempValueCounts[j]);
				}// Of if
			}// Of for j

			if (tempBlockSize == 0) {
				continue;
			}// Of if

			tempEntropy /= (double) tempBlockSize;
			tempEntropy += Utils.log2(tempBlockSize);// important!

			tempEntropyArray[i] = tempEntropy;

			tempTotalEntropy += tempEntropy * tempBlockSize / paraMatrix.length;
			// System.out.println("tempTotalEntropy = " +  tempTotalEntropy);
		}// Of for i

		 System.out.println("*********\r\nThe conditional entropy is: " +
		tempTotalEntropy + "\r\n");
		return tempTotalEntropy;
	}// Of conditionalEntropy

	/**
	 ************************* 
	 * Computes the conditional entropy of the given matrix.
	 * 
	 * @param paraMatrix
	 *            The matrix will be Computes the conditional entropy.
	 * @return the conditional entropy of the given matrix
	 * @author Liuying Wen 2014/12/23
	 ************************* 
	 */
	public static  double conditionalEntropyTwoAttribute(int[][] paraMatrix) {
		int tempNumConditionalValuesOfFirstAttribute = 0;
		int tempNumConditionalValuesOfSecondAttribute = 0;
		int tempNumClassValues = 0;

		double tempTotalEntropy = 0;
		/*
		 * System.out.println("-------------Compute conditional entropy for");
		 * printIntMatrix(paraMatrix);
		 */

		// Step 1. Count the number of conditional/class values.
		for (int i = 0; i < paraMatrix.length; i++) {
			if (tempNumConditionalValuesOfFirstAttribute < paraMatrix[i][0]) {
				tempNumConditionalValuesOfFirstAttribute = paraMatrix[i][0];
			}// Of if
			if (tempNumConditionalValuesOfSecondAttribute < paraMatrix[i][1]) {
				tempNumConditionalValuesOfSecondAttribute = paraMatrix[i][1];
			}// Of if
			if (tempNumClassValues < paraMatrix[i][2]) {
				tempNumClassValues = paraMatrix[i][2];
			}// Of if
		}// of for i
		tempNumConditionalValuesOfFirstAttribute++;
		tempNumConditionalValuesOfSecondAttribute++;
		tempNumClassValues++;

		printIntMatrix(paraMatrix);

		System.out.println("tempNumConditionalValuesOfFirstAttribute=" +
		tempNumConditionalValuesOfFirstAttribute);
		System.out.println("tempNumConditionalValuesOfSecondAttribute=" +
		tempNumConditionalValuesOfSecondAttribute);
		System.out.println("tempNumClassValues=" + tempNumClassValues);

		// Step 2. The entropy for each block
		// double[][] tempEntropyArray = new
		// double[tempNumConditionalValuesOfFirstAttribute][tempNumConditionalValuesOfSecondAttribute];
		double[] tempValueCounts = new double[tempNumClassValues];
		double[] tempValueCountsOfConditional = new double[tempNumClassValues];
		double tempProbabilityOfSecondAttribute = 0;
		double tempEntropy = 0;
		double tempTotalEntropyOfBlock = 0;
		double tempBlockSize = 0;
		double tempBlockSizeOfSecondAttribute = 0;

		// 计算H（X|Y）
		for (int i = 0; i < tempNumConditionalValuesOfSecondAttribute; i++) {

			tempProbabilityOfSecondAttribute = 0;
			for (int j = 0; j < tempNumClassValues; j++) {
				tempValueCountsOfConditional[j] = 0;
			}// of for j
			
			for (int j = 0; j < paraMatrix.length; j++) {
				if (paraMatrix[j][1] == i) {
					tempBlockSizeOfSecondAttribute++;
					tempValueCountsOfConditional[paraMatrix[j][2]]++;
				}// Of if
			}// of for j
			
			for (int j = 0; j < tempNumClassValues; j++) {
				tempProbabilityOfSecondAttribute += tempValueCountsOfConditional[j];
			}// Of for j
			
			tempProbabilityOfSecondAttribute /= paraMatrix.length;

			System.out.println("\r\n p(i="+i +")=" +
		     tempProbabilityOfSecondAttribute);

			// 对于H（X|Y=i）中X的每一块，计算其信息熵
			for (int j = 0; j < tempNumConditionalValuesOfFirstAttribute; j++) {
				// Reinitialize
				tempBlockSize = 0;
				tempBlockSizeOfSecondAttribute = 0;
				for (int k = 0; k < tempNumClassValues; k++) {
					tempValueCounts[k] = 0;
				}// of for j
					// Count
				for (int k = 0; k < paraMatrix.length; k++) {
					if (paraMatrix[k][0] == j && paraMatrix[k][1] == i) {
						tempBlockSize++;
						tempValueCounts[paraMatrix[k][2]]++;
					}// Of if
				}// of for j

				// Compute entropy for this block
				tempEntropy = 0;
				tempTotalEntropyOfBlock = 0;
				for (int k = 0; k < tempNumClassValues; k++) {
					// System.out.println("tempValueCounts["+k+"]="+tempValueCounts[k]);
					if (tempValueCounts[k] > 0) {
						tempEntropy -= tempValueCounts[k]
								* Utils.log2(tempValueCounts[k]);
					}// Of if
				}// Of for k
					// System.out.println("j=" + j + "的信息熵 = " + tempEntropy);
				if (tempBlockSize == 0) {
					continue;
				}// Of if

				tempEntropy /= (double) tempBlockSize;
				tempEntropy += Utils.log2(tempBlockSize);// important!
				tempTotalEntropyOfBlock += tempEntropy * tempBlockSize
						/ paraMatrix.length;

			}// of for j
		    System.out.println("tempTotalEntropyOfBlock= "+tempTotalEntropyOfBlock);
			tempTotalEntropy += tempTotalEntropyOfBlock
					* tempProbabilityOfSecondAttribute;
		}// Of for i

		// System.out.println("*********\r\nThe conditional entropy is: " +
		// tempTotalEntropy + "\r\n");
		return tempTotalEntropy;
	}// Of conditionalEntropyTwoAttribute


	
	/**
	 ************************* 
	 * 随机生成给定范围内的两个整数.
	 * 
	 * @param min
	 * 			 生成的最小的数
	 * @param max
	 * 			生成的最大的数
	 * @return  随机生成的两个数
	 * @author Liuying Wen 2014/12/23
	 ************************* 
	 */
	
	public static int[] randomNumGeneration(int min, int max) {
		int[] intRet = new int[2];

		int intRd = 0; // 存放随机数
		int count = 0; // 记录生成的随机数个数
		int flag = 0; // 是否已经生成过标志

		while (count < 2) {
			Random random = new Random(System.currentTimeMillis());
			intRd = Math.abs(random.nextInt(max)) % (max - min + 1) + min;
			for (int i = 0; i < count; i++) {
				if (intRet[i] == intRd) {
					flag = 1;
					break;
				} else {
					flag = 0;
				}
			}
			if (flag == 0) {
				intRet[count] = intRd;
				count++;
			}
		}
		/*
		 * for(int t=0;t<2;t++){ System.out.println(t+"->"+intRet[t]); }
		 */
		return intRet;

	}// of randomNumGeneration

	/**
	 ************************* 
	 * Convert a boolean array to an index array corresponding to true elements.
	 * 
	 * @param paraBooleanArray
	 *            the given boolean array
	 * @return the index array
	 ************************* 
	 * */
	public static int[] booleanArrayToIndexArray(boolean[] paraBooleanArray) {
		int tempSize = 0;
		// How many elements.
		for (int i = 0; i < paraBooleanArray.length; i++) {
			if (paraBooleanArray[i]) {
				tempSize++;
			}// Of if
		}// Of for i

		if (tempSize == 0) {
			return null;
		}// Of if

		// Determine the values.
		int[] resultArray = new int[tempSize];
		int tempIndex = 0;
		for (int i = 0; i < paraBooleanArray.length; i++) {
			if (paraBooleanArray[i]) {
				resultArray[tempIndex] = i;
				tempIndex++;
			}// Of if
		}// Of for i

		return resultArray;
	}// Of booleanArrayToIndexArray
	
	/**
	 ************************* 
	 * Partition the universe according to the given attributes.
	 * 
	 * @param paraAttributes
	 *            the given box. Which instances are included.
	 ************************* 
	 * */
	public static int[][] intCubicToIntMatrix(int[][][] paraCubic) throws Exception {
		//Step 1. How many valid rows?
		int tempRows = 0;
		for (int i = 0; i < paraCubic.length; i ++) {
			if (paraCubic[i] == null) {
				continue;
			}//Of if
			
			tempRows += paraCubic[i].length;
		}//Of for i
		
		if (tempRows == 0) {
			return null;
		}//Of if
		
		//Step 2. Copy
		int tempCurrentRow = 0;
		int[][] resultMatrix = new int[tempRows][];
		for (int i = 0; i < paraCubic.length; i ++) {
			if (paraCubic[i] == null) {
				continue;
			}//Of if
			
			for (int j = 0; j < paraCubic[i].length; j ++) {
				if (paraCubic[i][j] == null) {
					System.out.println("Cubic " + i + ", " + j + " is empty.");
				}
				resultMatrix[tempCurrentRow] = paraCubic[i][j];
				tempCurrentRow ++;
			}//Of for j
		}//Of for i
		
		return resultMatrix;
	}//Of intCubicToIntMatrix

	public  static double conditionalEntropyForDataset(int[][] paraMatrix) {
		double resultEntropy = 0;
		int tempSubsetSize = 0;
		for (int i = 0; i < paraMatrix.length; i ++) {
			tempSubsetSize = 0;
			if (paraMatrix[i] == null) {
				continue;
			}//Of if
			
			for (int j = 0; j < paraMatrix[i].length; j ++) {
				tempSubsetSize += paraMatrix[i][j];
			}//Of for j
			
			if (tempSubsetSize == 0) {
				continue;
			}//Of if
			
			resultEntropy += tempSubsetSize * entropy(paraMatrix[i]);
		}//Of for i
	
		resultEntropy /= paraMatrix.length;
		//System.out.println(paraMatrix.length);
		return resultEntropy;
	}//Of conditionalEntropyForDataset
	
	
	
	/**
	 * Testing method.
	 * 
	 * @param args
	 */
	public static void main(String args[]) {
		
		
		boolean[] mark = new boolean[5];
		//SimpleTool.printBooleanArray(mark);
		double[] a = {0.8,0.3,1.3,2.3};
		double[][] b = SimpleTool.sortDoubleArray(a);
		SimpleTool.printDoubleMatrix(b);
		/*
		int a[] = {1,1,0,2,3,2,0,4,5,4,6};
		int[] aa = SimpleTool.copyIntArray(a);
		System.out.println("原始数组：");
		SimpleTool.printIntArray(a);
		int b[] = compressAndSortIntArray(aa, aa.length);
		System.out.println("压缩排序后的数组：");

		SimpleTool.printIntArray(b);
		System.out.println("原始数组：");
		SimpleTool.printIntArray(a);
		int result[] = countNumerOfEachElement(a, b);

		SimpleTool.printIntArray(result);
		*/
		/*
		
		
		int[][] firstSet = {{0,1},{1,1},{1,0},{1,1},{1,1},{1,0},{1,0}};
		int[][] secondSet = {{0,1},{0,1},{1,0},{1,1},{1,1},{1,0},{1,0}};
		int[][] thirdSet = {{0,1},{0,1},{0,0},{0,1},{1,1},{1,0},{1,0}};
		int[][] fourthSet = {{0,1},{0,1},{0,0},{0,1},{0,1},{0,0},{1,0}};
	
		
		int[][] firstSet = {{0,1},{1,1},{2,0},{2,1},{2,1},{2,0},{2,0}};
		int[][] secondSet = {{0,1},{0,1},{1,0},{1,1},{2,1},{2,0},{2,0}};
		int[][] thirdSet = {{0,1},{0,1},{1,0},{1,1},{1,1},{1,0},{2,0}};
		
		int[][] firstSet = {
				{0,1},
				{0,1},
				{1,0},
				{1,1},
				{1,0},
				{1,0},
				{1,1}};
		int[][] secondSet = {
				{0,1},
				{0,1},
				{0,0},
				{0,1},
				{0,0},
				{1,0},
				{0,1}};
		int[][] thirdSet = {
				{1,1},
				{0,1},
				{1,0},
				{0,1},
				{1,0},
				{1,0},
				{0,1}};
	
		int[][] fourthSet = {
				{0,1},
				{0,1},
				{1,0},
				{0,1},
				{0,0},
				{1,0},
				{0,1}};
			
		//SimpleTool.printIntMatrix(firstSet);
		SimpleTool.conditionalEntropyOneAttribute(firstSet);
	    //SimpleTool.printIntMatrix(secondSet);
		SimpleTool.conditionalEntropyOneAttribute(secondSet);
		//SimpleTool.printIntMatrix(thirdSet);
		SimpleTool.conditionalEntropyOneAttribute(thirdSet);
		SimpleTool.conditionalEntropyOneAttribute(fourthSet);
		
		int[][] tempMatrix = { 
				 { 0, 1, 1}, 
				 { 0, 0, 1 }, 
				 { 1, 1, 0 }, 
				 { 1, 0, 1},
				 { 1, 1, 0 }, 
				 { 1, 1, 0 }, 
				 { 1, 0, 1}};
		 
			
		 int[][] tempMatrix1 = { 
				 {0, 1, 1}, 
				 {0, 0, 1}, 
				 {0, 1, 0}, 
				 {0, 0, 1},
				 {0, 1, 0}, 
				 {1, 1, 0}, 
				 {0, 0, 1}};
		 int[][] tempMatrix2 = { 
				 {0, 1, 1}, 
				 {0, 0, 1}, 
				 {1, 1, 0}, 
				 {0, 0, 1},
				 {0, 1, 0}, 
				 {1, 1, 0}, 
				 {0, 0, 1}};
		 
		 
		 System.out.println("Conditional entropy = " + 
				 conditionalEntropyTwoAttribute(tempMatrix));
		 */
		 //System.out.println("Conditional entropy = " + 
				// conditionalEntropyTwoAttribute(tempMatrix1));
		 
		// System.out.println("Conditional entropy = " + 
				// conditionalEntropyTwoAttribute(tempMatrix2));
		//boolean[] t = longToBooleanArray(1,127);
		//System.out.println(Integer.MAX_VALUE);
		//System.out.println(Float.MAX_VALUE);
		//9223372036854775807
		//9223372036854775807
	}

}// Of class SimpleTool
