"""
GeneSet MAPR implementation
	Step 03: characterize input set(s)
			ie: rank genes & describe connections

For an input set of nodes, rank all nodes in the
  network by the density of their connections to
  the set. Nodes connected by patterns similar
  to connections within the set are ranked higher.
  Output the rank and details about common
  connection patterns.

author: Greg Linkowski
	for KnowEnG by UIUC & NIH
"""

import argparse
import time
import numpy as np
from sklearn import linear_model as lm
import random
import warnings
import sys
import os
import gzip
import re
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


################################################################
# GLOBAL PARAMETERS

# Level of verbosity (feedback in terminal)
VERBOSITY = 0
# Character used to sepate text in output files
TEXT_DELIM = '\t'
# size the zero-padded matrix file name (match MAPR_networkPrep !)
FNAME_ZPAD = 6
# Data-type for the path matrices (allow room for computation)
MATRIX_DTYPE = np.float32

# end params ##############################




################################################################
# ANCILLARY FUNCTIONS

def readCommandLineFlags():
	parser = argparse.ArgumentParser()

	parser.add_argument('setsRoot', type=str,
	                    help='parent directory containing input sets')
	parser.add_argument('-l', '--length', type=int, default=3,
	                    help='maximum meta-path depth')
	parser.add_argument('-v', '--verbose', type=int, default=0,
	                    help='enable verbose output to terminal: 0=none, 2=all')
	parser.add_argument('-i', '--ignore', type=str, default='NONE',
	                    help='text file containing list of edge types to ignore')
	parser.add_argument('-m', '--numModels', type=int, default=101,
	                    help='number of random null sets to use for comparison')
	parser.add_argument('-p', '--plotAUCs', type=bool, default=False,
	                    help='to save plots of AUC curves, set to "True"')

	flags = parser.parse_args()

	return flags
# end def #################################


def setParamVerbose(newVal):
	"""
	set the global parameters

	:param newVal: bool, the new  value
	:return:
	"""
	global VERBOSITY
	VERBOSITY = min(max(newVal, 0), 2)

	return
# end def #################################


def setParamTextDelim(newVal):
	"""
	set the global parameters

	:param newVal: str, desired text delimiter character
		NOTE: an input value of...
			'-1' keeps the parameter unchanged
			'-2' resets parameter to default
	:return:
	"""
	global TEXT_DELIM

	if str(newVal) == '-1':
		TEXT_DELIM = TEXT_DELIM
	elif str(newVal) == '-2':
		TEXT_DELIM = '\t'
	else:
		TEXT_DELIM = str(newVal)
	# end if

	return
# end def #################################


def getGeneIndexLists(path, gDict):
	# Create index lists for Known, Hidden, Unknown, TrueNeg
	gKnown = readFileAsList(path + 'known.txt')
	giKnown = convertToIndices(gKnown, gDict)
	gHidden = readFileAsList(path + 'concealed.txt')
	giHidden = convertToIndices(gHidden, gDict)
	giUnknown = [g for g in gDict.values() if g not in giKnown]
	giTrueNeg = [g for g in giUnknown if g not in giHidden]

	return giKnown, giUnknown, giHidden, giTrueNeg
# end def #################################


def readFileAsList(fName) :
	"""
	Read in a file as a line-by-line list of items

	:param fName: str, path + name of the the sample files
	:return: fItems, str list: ordered list of items from file
	"""

	# ERROR CHECK: verify file exists
	if not os.path.isfile(fName) :
		print ( "ERROR: Specified file doesn't exist:" +
			" {}".format(fName))
		sys.exit()
	#end if

	# The list of items to return
	fItems = list()

	# Read in from the file
	fn = open(fName, "r")
	for line in fn :
		fItems.append( line.rstrip() )
	#end loop
	fn.close()

	fItems.sort()
	return fItems
# end def #################################


def convertToIndices(names, iDict) :
	"""
	Given a list of names and appropriate dict,
	   convert to corresponding index values

	:param names: str list, names of items to convert
	:param iDict: dict,
		key, str: names of items
		value, int: corresponding index values
	:return: indices, int list: index values of the input items
	"""

	# The item to return
	indices = list()

	for name in names :
		indices.append(iDict[name])
	#end loop

	return indices
# end def #################################


def aggregateRankFromScore(scoreCols, weights):
	# Normalize the columns
	# Center each column about the mean
	mMean = np.mean(scoreCols, axis=0)
	mNormed = np.subtract(scoreCols, mMean)
	# Set the L2 norm = 1
	mAbsMax = np.amax(np.absolute(mNormed), axis=0)
	mAbsMax = np.add(mAbsMax, 0.0001)  # so as not to / 0
	scoreColsNormed = np.divide(mNormed, mAbsMax)

	# numRows = scoreCols.shape[0]
	numRows = scoreColsNormed.shape[0]
	sumRanks = np.zeros(numRows)
	sumScores = np.zeros(numRows)

	# For each column
	#	rank the rows by their score
	#		sort by score, apply rank, sort by index
	#	multiply rank by weight & add to sumRanks
	# Then sort & rank again
	ranker = np.recarray(numRows,
	                     dtype=[('rowIdx', 'i4'), ('score', 'f4'), ('rank', 'f4')])
	for i in range(len(weights)):
		ranker['rowIdx'] = np.arange(numRows)
		ranker['score'] = scoreCols[:, i]
		ranker['rank'] = np.zeros(numRows)
		ranker.sort(order=['score'])
		ranker = ranker[::-1]

		ranker['rank'] = np.arange(numRows)
		ranker.sort(order=['rowIdx'])

		sumRanks = np.add(sumRanks, np.multiply(ranker['rank'], weights[i]))
		sumScores = np.add(sumScores, np.multiply(ranker['score'], weights[i]))
	# end loop

	sNormed = np.subtract(sumScores, np.mean(sumScores))
	sAbsMax = np.add(np.amax(np.absolute(sNormed)), 0.000001)
	finScores = np.divide(sNormed, sAbsMax)

	rankFinal = np.recarray(numRows,
	                        dtype=[('rowIdx', 'i4'), ('score', 'f4'), ('rankSum', 'f4')])
	rankFinal['rowIdx'] = np.arange(numRows)
	rankFinal['score'] = finScores
	rankFinal['rankSum'] = sumRanks
	rankFinal.sort(order=['rankSum'])

	return rankFinal
# end def #################################


def aggRankFromStandardizedScore(scoreCols, weights):
	"""
	Get the aggregate rank of each gene from the weighted mean of their standardized scores
	over each predictive model/vote. Weights come from the model's ability to separate
	the Pos/Neg training data.

	:param scoreCols: 2D array (genes, models/votes), as numpy array
	:param weights: list/array of weights, one per model/vote
	:return:
	"""

	#### Standardize the scores by column
	#	using column mean & column standard deviation
	cMean = np.reshape(np.mean(scoreCols, axis=0), (1, scoreCols.shape[1]))
	cSTD = np.reshape(np.std(scoreCols, axis=0), (1, scoreCols.shape[1]))
	scoresStd = np.divide(np.subtract(scoreCols, cMean), np.add(cSTD, 1e-11))

	# scoreStd = np.divide( scoreCols, np.add(cSTD, 1e-8))

	#### Get the weighted mean
	scoresMean = np.sum(np.multiply(scoresStd, np.reshape(weights, (1, len(weights)))), axis=1)
	scoresMean = np.divide(scoresMean, np.sum(weights))

	#### Return list  of  genes sorted by rank
	ranker = np.recarray(scoreCols.shape[0],
	                     dtype=[('rowIdx', 'i4'), ('score', 'f4'), ('rankSum', 'f4')])
	ranker['score'] = scoresMean
	ranker['rowIdx'] = np.arange(scoreCols.shape[0])
	ranker.sort(order=['score'])
	ranker = ranker[::-1]
	ranker['rankSum'] = np.arange(scoreCols.shape[0])
	ranker.sort(order=['rankSum'])

	return ranker
# end def #################################


def checkIfNameHasDuplicate(pathName):
	"""
	Check if a meta-path name consecutively contains two of the same edge type.

	:param pathName: (str) the MP name, types separated by '-'
	:return: (bool) True if a duplicate occurs, False otherwise
	"""

	retBool = False
	pv = pathName.split('-')
	if len(pv) > 1:
		for i in range(1, len(pv)):
			if pv[i] == pv[i - 1]:
				retBool = True
				break
			# end if
		# end for
	# end if

	return retBool
# end def #################################


def getGeneAndPathDict(path) :
	"""
	Get the specified geneDict & pathDict from parameters.txt

	:param path: str, path to the samples' parameter.txt file
		(parameters.txt tells where the network is stored/named)
	:return: geneDict, pathDict
	"""

	# get the network path & name from the parameters.txt file
	if not path.endswith('/') :
		path = path + '/'
	with open(path + 'parameters.txt', 'r') as fin :
		line = fin.readline()
		del line

		line = fin.readline()
		line = line.rstrip()
		lv = line.split(TEXT_DELIM)
		eName = lv[1]

		line = fin.readline()
		line = line.rstrip()
		lv = line.split(TEXT_DELIM)
		ePath = lv[1]
	#end with

	if VERBOSITY :
		print("Reading gene and path dictionaries for {}".format(eName))
	geneDict = readGenesFile(ePath, eName)
	pathDict = readKeyFile(ePath, eName)

	return geneDict, pathDict
# end def #################################


def readGenesFile(path, name) :
	"""
	Read in the genes.txt file containing the
	gene-name headers to the metapath matrices
	Pre-reqs: readFileAsIndexDict(fname)

	:param path: str, path to the network files
	:param name: name, str, name of the network to use
	:return: gDict, dict
		key, str: name of gene
		value, int: row/col index for that gene
	"""

	fname = name + '_MetaPaths/genes.txt'
	if verifyFile(path, name + fname, True) :
		# The item to return
		gDict = readFileAsIndexDict(fname)
	else :
		fname = path + name + '/genes.txt'
		# The item to return
		gDict = readFileAsIndexDict(fname)
	#end if

	return gDict
# end def #################################


def verifyFile(path, name, quiet) :
	"""
	ERROR CHECK: verify file exists

	:param path: str, path to save the file
	:param name: str, name of the file (w/ extension)
	:param quiet: bool, whether to quietly return T/F
	:return: exists, bool: indicates existence of file
	"""
	exists = True

	# First check the directory
	if not (path == '') :
		exists = verifyDirectory(path, False, quiet)

	# Then look for the file
	if not path.endswith('/') :
		path = path+'/'
	#end if
	if not os.path.isfile(path+name) :
		if quiet:
			exists = False
		else :
			print ( "ERROR: Specified file doesn't exist:" +
				" {}".format(path) )
			sys.exit()
	#end if

	return exists
# end def #################################


def verifyDirectory(path, create, quiet) :
	"""
	ERROR CHECK: verify directory exists

	:param path: str: path to verify
	:param create: bool, whether to create missing dir
	:param quiet: bool, whether to quietly return T/F
	:return: exists, bool: indicates existence of directory
	"""
	exists = True
	if not os.path.isdir(path) :
		if create :
			print("Creating path: {}".format(path))
			os.makedirs(path)
		elif quiet :
			exists = False
		else :
			print ( "ERROR: Specified path doesn't exist:" +
				" {}".format(path) )
			sys.exit()
	#end if
	return exists
# end def #################################


def readFileAsIndexDict(fName) :
	"""
	Read in the gene file. File is an ordered list
	of genes, where the row number (starting at zero)
	corresonds to the index in the matrix/list/etc where
	the gene can be found.

	:param fName: str, path & name to keep file
	:return: iDict, dict:
		key, str: gene name as read from file
		value, int: index to corresponding array
	"""

	# ERROR CHECK: verify file exists
	if not os.path.isfile(fName) :
		print ( "ERROR: Specified file doesn't exist:" +
			" {}".format(fName))
		sys.exit()
	#end if

	# Build the dictionary from the text file
	iDict = dict()
	gf = open(fName, "r")
	index = 0
	for line in gf :
		gene = line.rstrip()    # remove "\n"
		iDict[gene] = int(index)
		index += 1
	#end loop

	return iDict
# end def #################################


def readKeyFile(path, name) :
	"""
	Read in the key.txt file regarding the
	metapath matrices

	:param path: str, path to the network files
	:param name: str, name of the network to use
	:return: keyDict, dict
		key, str: name of metapath
		value, tuple: int is matrix/file ID number
			bool where True means use matrix transpose
	"""

	fName = path + name + "_MetaPaths/key.txt"

	# ERROR CHECK: verify file exists
	if not os.path.isfile(fName) :
		print ( "ERROR: Specified file doesn't exist:" +
			" {}".format(fName) )
		sys.exit()
	#end if

	# The item to return
	keyDict = dict()

	# Read in the file
	fk = open(fName, "r")
	firstLine = True
	for line in fk :

		# skip the first line
		if firstLine :
			firstLine = False
			continue
		#end if

		# separate the values
		line = line.rstrip()
		lk = line.split('\t')
		lv = lk[0].split(',')

		transpose = False
		if lv[1] == "t" :
			transpose = True
		#end if

		# add to the dict
		keyDict[lk[1]] = [int(lv[0]), transpose]
	#end loop
	fk.close()

	return keyDict
# end def #################################


def removeInvertedPaths(mpDict) :
	"""
	Find the number of paths of this type joining
	the nodes in the sample

	:param mpDict: {str: [int, bool]} dict,
		key, str - name of the metapath
		value, [int, bool] - which matrix file to use, and
			whether to use the transpose (inverse path)
	:return: mpList, str list: ordered names of paths available,
		less paths that are mirror-images of another
	"""

	# The item to return
	mpList = list()

	# Check the keys in the dict
	for key in mpDict.keys() :
		# If the boolean is True, then the path is an
		#   inverse of another; only append if false
		if not mpDict[key][1] :
			mpList.append(key)
	#end loop

	mpList.sort()
	return mpList
# end def #################################


def getFeaturesNeighborhood(path, suffix):
	"""
	Get the neighborhood features from parameters.txt

	:param path: str, path to the samples' parameter.txt file
		(parameters.txt tells where the network is stored/named)
	:param suffix: str, filename suffix (which version of feature to load)
	:return: featVals, featNames
	"""
	# get the network path & name from the parameters.txt file
	if not path.endswith('/'):
		path = path + '/'
	with open(path + 'parameters.txt', 'r') as fin:
		line = fin.readline()
		del line

		line = fin.readline()
		line = line.rstrip()
		lv = line.split(TEXT_DELIM)
		eName = lv[1]

		line = fin.readline()
		line = line.rstrip()
		lv = line.split(TEXT_DELIM)
		ePath = lv[1]
	# end with

	if VERBOSITY:
		print("Reading neighborhood features file for {}".format(eName))

	if not ePath.endswith('/'):
		ePath = ePath + '/'
	if not eName.endswith('/'):
		eName = eName + '/'

	featVals = np.loadtxt(ePath + eName + 'featNeighbor_' + suffix + '.gz')
	featNames = readFileAsList(ePath + eName + 'featNeighbor_Names.txt')
	featNames = np.reshape(featNames, (1, len(featNames)))

	return featVals, featNames
# end def #################################


def getSubDirectoryList(root) :
	"""
	Return list of paths to folders in the
	given root directory

	:param root: str, path where the folders reside
	:return: subDirs, str list: sorted list of subdirectories
		contains full path: root+subdir
	"""

	verifyDirectory(root, False, False)

	if not root.endswith('/') :
		root = root+'/'
	subDirs = [(root+d+'/') for d in os.listdir(root) if os.path.isdir(root+d)]
	subDirs.sort()

	return subDirs
# end def #################################


def normalizeFeatureColumns(featMatrix) :
	"""
	Normalize each column of the feature matrix

	:param featMatrix: matrix containing feature weights
		row: gene feature vector
		col: each individual feature
	:return: featNormed: the normalized copy of the original matrix
	"""

	# Center each column about the mean
	featMean = np.mean(featMatrix, axis=0)
	featNormed = np.subtract(featMatrix, featMean)

	# Set the L2 norm = 1
	featAbsMax = np.minimum(featMean, np.amax(featNormed, axis=0))
	featAbsMax = np.add(featAbsMax, 1)	# hack so as not to / by 0
	featNormed = np.divide(featNormed, featAbsMax)

	return featNormed
# end def #################################


def getFeaturesTermsV2(path) :
	"""
	Get the term weight features from parameters.txt

	:param path: str, path to the samples' parameter.txt file
		(parameters.txt tells where the network is stored/named)
	:return: featVals, featNames
	"""

	# get the network path & name from the parameters.txt file
	if not path.endswith('/') :
		path = path + '/'
	with open(path + 'parameters.txt', 'r') as fin :
		line = fin.readline()
		del line

		line = fin.readline()
		line = line.rstrip()
		lv = line.split('\t')
		eName = lv[1]

		line = fin.readline()
		line = line.rstrip()
		lv = line.split('\t')
		ePath = lv[1]
	#end with

	if not ePath.endswith('/') :
		ePath = ePath + '/'
	if eName.endswith('/') :
		eName = eName.rstrip('/')


	########
	# Load the non-gene term raw data matrices

	# Read in the key file for terms: key_termNonGene.txt
	termKey = readKeyFileSimple(ePath, eName, 'key_termNonGene.txt')

	# Count how many genes there are (number rows)
	gList = readFileAsList(ePath + eName + '/genes.txt')
	gCount = len(gList)
	del gList

	# Initialize the items to return
	featVals = np.zeros( (gCount,0), dtype=np.float32)
	featNames = list()

	# Append the items from each term dictionary
	for e in list(termKey.keys()) :
		ftlName = ePath + eName + '_MetaPaths/' + termKey[e] + 'tl.txt'
		tList = readFileAsList(ftlName)
		tMatrix = getTermMatrix(termKey[e], ePath, eName, gCount, len(tList))

		featVals = np.hstack((featVals, tMatrix))
		featNames = np.append(featNames, tList)
	#end loop

	########
	# Load the gene-gene network matrices

	# Read in the key file for all primary networks: key_primaries.txt
	primKey = readKeyFileSimple(ePath, eName, 'key_primaries.txt')
	geneKey = dict()
	for k in primKey.keys() :
		if k in termKey.keys() :
			continue
		geneKey[k] = (primKey[k], False)
	#end for
	del termKey, primKey

	# Load the gene names
	gNames = readFileAsList(ePath + eName + '/genes.txt')

	# Append the items from each gene-gene matrix
	sortedKeys = list(geneKey.keys())
	sortedKeys.sort()
	sumMatrix = np.zeros( (len(gNames), len(gNames)), dtype=np.float32)
	for e in list(sortedKeys) :
		# Normalize the weights for this set of terms
		fMatrix = loadPathMatrix(geneKey[e], ePath, eName, len(gNames))
		fMatrix = np.divide(fMatrix, np.sum(fMatrix))
		sumMatrix =  np.add(sumMatrix, fMatrix)

		del fMatrix
	#end loop

	featVals = np.hstack((featVals, sumMatrix))
	featNames = np.append(featNames, gNames)

	if VERBOSITY == 2 :
		sizeGB = featVals.nbytes / 1.0e9
		print("  ... total feature vector size, {:.3f} Gb".format(sizeGB))
	#end if

	return featVals, featNames
# end def #################################


def readKeyFileSimple(path, name, kName):
	"""
	Read in the key.txt file regarding the metapath matrices

	:param path: (str) path to the network files
	:param name: (str) name of the network to use
	:param kName: (str) name of the keyfile to read
	:return: keyDict (dict)
      key, str: name of metapath
      value, tuple: int is matrix/file ID number
          bool where True means use matrix transpose
	"""

	fName = path + name + "_MetaPaths/" + kName

	# ERROR CHECK: verify file exists
	if not os.path.isfile(fName):
		fName = path + name + '/' + kName
		if not os.path.isfile(fName):
			print("ERROR: Specified file doesn't exist:" +
			      " {}".format(fName))
			sys.exit()
	# end if

	# The item to return
	keyDict = dict()

	# Read in the file
	fk = open(fName, "r")
	for line in fk:

		# separate the values
		line = line.rstrip()
		if line.startswith('NOTE:'):
			continue
		lv = line.split('\t')

		# add to the dict
		keyDict[lv[1]] = lv[0]
	# end loop
	fk.close()

	return keyDict
# end def #################################


def getTermMatrix(mpKeyNum, path, name, nRows, nCols) :
	"""
	Load the matrix containing the number of paths
	of this type which join the nodes in the network

	:param mpKeyNum: (int, bool): indicates which matrix file to use
	:param path: str, path to the network files
	:param name: str, name of the network to use
	:param nRows: int, number of rows in new matrix
	:param nCols: int, number of columns in new matrix
	:return: matrix, int array: num paths between node pairs
	"""

	# Define the file name for the matrix
	preName = (path + name + "_MetaPaths/" +
		"{}tm".format(str(mpKeyNum).zfill(FNAME_ZPAD)) )
	if os.path.isfile(preName + '.gz') :
		fName = (path + name + "_MetaPaths/" +
		"{}tm.gz".format(str(mpKeyNum).zfill(FNAME_ZPAD)) )
	elif os.path.isfile(preName + '.txt') :
		fName = (path + name + "_MetaPaths/" +
		"{}tm.txt".format(str(mpKeyNum).zfill(FNAME_ZPAD)) )
	else :
		# ERROR CHECK: verify file exists
		print ( "ERROR: Specified file doesn't exist:" +
			"{}".format(preName) )
		sys.exit()
	#end if

	# Allocate the matrix
	matrix = np.zeros([nRows, nCols], dtype=MATRIX_DTYPE)

	# Read in the file, placing values into matrix
	row = 0
	with gzip.open(fName, 'rb') as fin :
		for line in fin :
			line = line.rstrip()
			ml = line.split()
			# print(line)
			# print(ml)
			matrix[row,:] = ml[:]
			row += 1
	#end with

	return matrix
# end def #################################


def loadPathMatrix(mpTuple, path, name, sizeOf) :
	"""
	Load the matrix containing the number of paths
	of this type which join the nodes in the network

	:param mpTuple: [int, bool]: indicates which matrix file to use
	:param path: str, path to the network files
	:param name: str, name of the network to use
	:param sizeOf: int, number of rows (& cols) in new matrix
	:return: int array: num paths between node pairs
	"""

	preName = (path + name + "_MetaPaths/" +
		"{}".format(str(mpTuple[0]).zfill(FNAME_ZPAD)) )
	if os.path.isfile(preName + '.gz') :
		fName = (preName + '.gz')
	elif os.path.isfile(preName + '.txt') :
		fName = (preName + '.txt')
	else :
		# ERROR CHECK: verify file exists
		print ( "ERROR: Specified file doesn't exist:" +
			" {} .gz/.txt".format(preName) )
		sys.exit()
	#end if

	# Declare the matrix
	matrix = np.zeros([sizeOf, sizeOf], dtype=MATRIX_DTYPE)

	# Read in the file, placing values into matrix
	row = 0
	with gzip.open(fName, 'rb') as fin :
		for line in fin :
			line = line.rstrip()
			ml = line.split()
			matrix[row,:] = ml[:]
			row += 1
	#end with

	# Convert to transpose if flag==True
	if mpTuple[1] :
		return np.transpose(matrix)
	else :
		return matrix
# end def #################################


def readFileColumnAsString(fName, iCol, nSkip):
	"""
	Read in the desired column from a csv/tsv file
		(typically a list of genes), skip N header rows

	:param fName: str, path & filename of input
	:param iCol: int, index of column to read (! INDEXED AT 0 !)
	:param nSkip: int, number of rows to skip at top (ie: header rows)
	:return:
	"""
	# ERROR CHECK: verify file exists
	if not os.path.isfile(fName):
		print("ERROR: Specified file doesn't exist:" +
		      " {}".format(fName))
		sys.exit()
	# end if

	# the item to return
	theList = list()

	# Read in from the file
	theFile = open(fName, 'r')
	count = 0
	firstWarn = True
	lvLen = -1
	for line in theFile:
		# Skip first nSkip number of rows
		if count < nSkip:
			continue
		# end if

		line = line.rstrip()
		lv = line.split(TEXT_DELIM)

		# ERROR CHECK: verify iCol is within range
		if (firstWarn == True) and (iCol >= len(lv)):
			print("WARNING: File contains {} columns,".format(len(lv)) +
			      " but col {} was requested.".format(iCol))
			print("    Note: Indexing starts at 0.")
			print("    Returning last column. ")
			iCol = len(lv) - 1
			firstWarn = False
		# end if

		# ERROR CHECK: Warn if column length changes
		if (lvLen != len(lv)) and (lvLen >= 0):
			print("WARNING: Inconsistent number of columns in file.")
			print("    Previous width: {}; Current: {}".format(
				lvLen, len(lv)))
		# end if
		lvLen = len(lv)

		theList.append(str(lv[iCol]))
		count += 1
	# end loop
	theFile.close()

	return theList
# end def #################################


def countLinesInFile(fName):
	"""
	Count the number of lines in a file
		used to set size of matrices & arrays

	:param fName: str, path & filename of input
	:return:
		cRows, int: total # of lines in the file
		cColMin, int: minimum # of columns in the file
		cColMax, int: maximum # of columns in the file
	"""
	# ERROR CHECK: verify file exists
	if not os.path.isfile(fName):
		print("ERROR: Specified file doesn't exist:" +
		      " {}".format(fName))
		sys.exit()
	#end if

	cRows = 0
	cColMin = -1
	cColMax = 0
	if fName[-3:0] == '.gz':
		with gzip.open(fName, 'r') as fin:
			for line in fin:
				lv = line.split(TEXT_DELIM)
				lineLength = len(lv)

				cRows += 1

				if cColMax < lineLength:
					cColMax = lineLength

				if cColMin == -1:
					cColMin = lineLength
				elif cColMin > lineLength:
					cColMin = lineLength
	#end if
	else:
		fin = open(fName, 'r')
		for line in fin:
			lv = line.split(TEXT_DELIM)
			lineLength = len(lv)

			cRows += 1

			if cColMax < lineLength:
				cColMax = lineLength

			if cColMin == -1:
				cColMin = lineLength
			elif cColMin > lineLength:
				cColMin = lineLength
		#end for
		fin.close()
	#end if

	return cRows, cColMin, cColMax
# end def #################################


def getAUCStats(path, name):
	"""
	calculate Recall (TPR), FPR, & Precision

	:param path: str, path to input file
	:param name: str, name of input file
	:return:
		recall, float:
		FPR, float:
		precision, float:
		nHidden, float:
	"""

	# Read in the ranked genes
	fnConcealed = 'concealed.txt'
	if not os.path.isfile(path + fnConcealed):
		fnConcealed = 'hidden.txt'
	gHidden = readFileColumnAsString(path + fnConcealed, 0, 0)
	gHidSet = set(gHidden)
	nHidden = len(gHidden)

	# In case concealed.txt is empty
	if nHidden == 0:
		if VERBOSITY > 1 :
			print("There are no Concealed Positives to predict in {}".format(path))
		return [-1], [-1], [-1], [-1]
	# end if

	# Declare the confusion matrix
	rows, colMin, colMax = countLinesInFile(path + name)
	posActual = len(gHidden)
	negActual = rows - posActual
	confusion = np.zeros([2, rows])  # TP, FP

	# Create the matrix by reading in the file
	#	matrix is running count of: True Positives, False Positives
	fin = open(path + name)
	col = 0
	TP = 0
	FP = 0
	for line in fin:
		line = line.rstrip()
		lv = line.split(TEXT_DELIM)

		if lv[1] in gHidSet:
			TP += 1
		else:
			FP += 1
		# end if
		confusion[:, col] = [TP, FP]

		col += 1
	# end loop
	fin.close()

	# Convert into FPR, Recall (TPR), & Precision
	recall = confusion[0, :] / posActual  # aka TPR
	FPR = confusion[1, :] / negActual
	precision = confusion[0, :] / (confusion[0, :] + confusion[1, :])

	return FPR, recall, precision, nHidden
# end def #################################




################################################################
# PRIMARY FUNCTION(S)
def predictIterative(ignoreList, limitMPLen, sDir, numVotes, flagVerbose):

	setParamVerbose(flagVerbose)

	#### PARAMETERS

	## Control the iterations & error
	# how many iterations to perform
	numIterations = 1
	# minimum number of Known genes before iterations stop
	numExitKnown = 29
	# min number of Unknown genes before iterations halt
	numExitUnknown = 399

	## Recognize a failed model and retry
	# whether to allow coeffs = 0 in results
	retryOnZeroCoeffs = True  # True
	# how many of Known to keep in new sub-sample
	retrySubPortion = 0.75
	# minimum Known genes to use for PosTrain
	retryMinValid = 9
	# total maximum number of retries to attempt for a gene set
	maxRetries = (numVotes * 5)  # (numVotes * 5)

	# select only meta-paths of specific length(s)
	if not limitMPLen:
		limitMPLen = [1, 2, 3]
	elif type(limitMPLen) == int :
		limitMPLen = np.arange(1, limitMPLen + 1)
	# True/False: remove paths where same edge type occurs consecutively
	limitDuplicatePaths = False  # False
	# File name containing path z-score vectors
	fZScoreSim = 'features_ZScoreSim.gz'

	## Classifier parameters
	# True/False: limit to only Positive coefficient values
	usePos = False
	# calculate new alpha for every round
	alwaysNewAlpha = True  # True
	# array of alpha values; 'None' means to auto-search for alphas
	useGivenRange = np.logspace(-6, -2.301, num=13, base=10.0, endpoint=True)
	lMaxIter = 200
	lNorm = True
	lFitIcpt = True

	# Label for pos & neg labels
	pLabel = 1
	nLabel = 0  # 0
	negMultiplier = 1

	# end of parameters
	############################

	if VERBOSITY:
		print("\nPerforming regression(s) on {}".format(sDir))

	setParamVerbose(VERBOSITY)


	# 0) Create the useLabel variable
	useLabel = 'Iter{}V{}_c'.format(numIterations, numVotes)
	useLabel = useLabel + 'Las'
	if usePos:
		useLabel = useLabel + 'Pos'
	if alwaysNewAlpha:
		useLabel = useLabel + '_aA'
	if retryOnZeroCoeffs:
		useLabel = useLabel + '_wRS2'  # indicating re-sample on failed classifier
	useLabel = useLabel + '_m{}'.format(negMultiplier)
	useLabel = useLabel + '_fZ'
	for item in limitMPLen:
		useLabel = useLabel + '{}'.format(item)
	if limitDuplicatePaths:
		useLabel = useLabel + 'noDup'
	if ignoreList:  # List ignored edge types
		useLabel = useLabel + '_i'
		for it in ignoreList:
			useLabel = useLabel + it[1]
	# end if

	if VERBOSITY:
		print("Using label: {}".format(useLabel))


	# 1) Load the gene-index dictionary & path names
	geneDict, pathDict = getGeneAndPathDict(sDir)
	geneNames = list(geneDict.keys())
	geneNames.sort()
	pathNames = removeInvertedPaths(pathDict)
	del pathDict

	# Keep only edges that match desired length & ignore requirement
	keepPathIdx = list()
	idx = -1
	for name in pathNames:
		idx += 1

		# Skip if path length not in limitMPLen
		pLen = name.count('-') + 1
		if pLen not in limitMPLen:
			continue

		# Skip if path contains duplicate (if selected)
		if limitDuplicatePaths:
			if checkIfNameHasDuplicate(name):
				continue
		# end if

		# Keep only if ignored item NOT in path name
		if ignoreList:
			match = False
			for it in ignoreList:
				edge = it[0]
				if edge in name:
					match = True
			if not match : # match == False
				keepPathIdx.append(int(idx))
		else:
			keepPathIdx.append(int(idx))
		# end if
	# end loop

	if len(keepPathIdx) < len(pathNames):
		newPathNames = list()
		for idx in keepPathIdx:
			newPathNames.append(pathNames[idx])
		pathNames = newPathNames
	else :
		keepPathIdx = np.arange(len(pathNames))
	# end if
	if VERBOSITY:
		print("After removing ignored edges and lengths, using " +
		      "{} metapaths".format(len(pathNames)))

	# ERROR CHECK: exit program if no features being used
	if len(pathNames) == 0 :
		print("WARNING: No metapaths selected, and no term weights, exiting function ...")
		print("    MP lengths: {}".format(limitMPLen))
		print("    ignored edges: {}".format(ignoreList))
		return
	# end if


	# 2) Loop over the list of the sample subdirectories
	dSubDirs = getSubDirectoryList(sDir)

	usedAlphasAll = list()
	usedIterConvergenceAll = list()
	usedNumFeatures = list()
	usedTimePerSample = list()
	thisRound = 0

	for si in dSubDirs:
		thisRound += 1
		tStartSample = time.time()
		usedAlphas = list()
		usedIterConvergence = list()

		# Display directory to examine
		sv = si.split('/')
		if VERBOSITY:
			print("\n{}/{}/".format(sv[-3], sv[-2]))

		# Create index lists for Known, Hidden, Unknown, TrueNeg from files
		giKnown, giUnknown, giHidden, giTrueNeg = getGeneIndexLists(si, geneDict)


		# 3) Extract features for each set from each sub-dir

		# z-score of path counts features
		featZSVals = np.loadtxt(si + fZScoreSim)
		features = featZSVals[:, keepPathIdx]
		featNames = pathNames
		numFeatures = len(featNames)

		# verify some features have been loaded
		if features.shape[1] == 0:
			print("ERROR: No features were specified for classification.")
			sys.exit(0)
		if numFeatures != features.shape[1]:
			print("ERROR: Mismatch b/t # feature columns & # feature names.")
			sys.exit(0)
		# end if

		usedNumFeatures.append(numFeatures)
		if VERBOSITY:
			print("    ... using {} features".format(numFeatures))

		# Normalize the feature values
		features = normalizeFeatureColumns(features)


		# 4) Create storage; prepare train/test vectors & labels

		# Create the structure to rank the Unknown genes & paths
		geneScores = np.zeros((len(geneDict), numIterations), dtype=np.float32)

		# set the gene indices for the first iteration
		iterKnown = giKnown
		iterUnknown = giUnknown
		iterAll = list()
		iterAll.extend(iterKnown)
		iterAll.extend(iterUnknown)
		iterAll.sort()

		for itr in range(numIterations):
			# store the results for each random sample
			iterNumGenes = len(iterAll)
			voteScores = np.zeros((iterNumGenes, numVotes), dtype=np.float32)

			if VERBOSITY >= 2:
				print("  iteration {} of {}; {} votes".format((itr + 1),
				                                              numIterations, numVotes))
				print("  known: {}, total: {}, trainSet: {}".format(len(iterKnown),
				                                                    iterNumGenes,
				                                                    (len(iterKnown) * (1 + negMultiplier))))
			# end if

			retryNewAlpha = True
			retrySubSample = False
			retries = 0
			vote = 0
			vWeights = list()

			# NOTE: featScores only valid if numIterations == 1
			featScores = np.zeros((numFeatures, numVotes), dtype=np.float64)

			while vote < numVotes:
				if retrySubSample:
					retrySubSample = False

					numSubSample = int(numSubSample * retrySubPortion) + 1
					retryIterKnown = random.sample(iterKnown, numSubSample)
					if len(retryIterKnown) < retryMinValid:
						retryIterKnown = random.sample(iterKnown, retryMinValid)

					posTrain = features[retryIterKnown, :]
					posTrainLabel = np.ones((len(retryIterKnown), 1)) * pLabel
					nExamples = min(negMultiplier * len(retryIterKnown), (iterNumGenes - len(retryIterKnown)))
				else:
					numSubSample = len(iterKnown)

					posTrain = features[iterKnown, :]
					posTrainLabel = np.ones((len(iterKnown), 1)) * pLabel
					nExamples = min(negMultiplier * len(iterKnown), (iterNumGenes - len(iterKnown)))
				# end if

				# Extract the vectors for neg sets
				# as one-class: train with rand samp from Unknown
				#		test with all Unknown (TrueNeg + Hidden/TruePos)
				giTrainNeg = random.sample(iterUnknown, nExamples)
				negTrain = features[giTrainNeg, :]
				negTrainLabel = np.ones((len(giTrainNeg), 1)) * nLabel

				# Combine to create the full train & test data sets
				# as one-class:
				trainSet = np.vstack((posTrain, negTrain))
				trainLabel = np.vstack((posTrainLabel, negTrainLabel))
				testSet = features[iterAll, :]

				# reshape the labels for classifier
				trainLabel = np.reshape(trainLabel, [trainLabel.shape[0], ])


				# 5) Train classifier, predict on test, collect scores

				if alwaysNewAlpha:
					retryNewAlpha = True
				if retryNewAlpha:
					with warnings.catch_warnings():
						warnings.simplefilter('ignore')
						cfier = lm.LassoCV(alphas=useGivenRange, positive=usePos,
						                   max_iter=lMaxIter, normalize=lNorm, fit_intercept=lFitIcpt)
						cfier.fit(trainSet, trainLabel)
					# end with (warnings)
					foundAlpha = cfier.alpha_
					usedAlphas.append(foundAlpha)
					usedIterConvergence.append(cfier.n_iter_)
					usedAlphasAll.append(foundAlpha)
					usedIterConvergenceAll.append(cfier.n_iter_)
					retryNewAlpha = False
				else:
					with warnings.catch_warnings():
						warnings.simplefilter('ignore')
						cfier = lm.Lasso(alpha=foundAlpha, max_iter=lMaxIter, normalize=lNorm,
						                 positive=usePos, fit_intercept=lFitIcpt)
						cfier.fit(trainSet, trainLabel)
					# end with (warnings)
				# end if
				vWeights.append(cfier.score(trainSet, trainLabel))

				# view quick statistics from this training session
				if VERBOSITY >= 2:
					print("    Vote {}-{}; iters {:3d}, alpha {:.5f}, score {:.3f}; coeffs {}; sample {}".format(
						(itr + 1), (vote + 1), cfier.n_iter_, foundAlpha, cfier.score(trainSet, trainLabel),
						len(np.nonzero(cfier.coef_)[0]), len(posTrainLabel)))

				cfPredLabel = cfier.predict(testSet)
				cfPredLabel = np.ravel(cfPredLabel)

				# If no coefficients (train score == 0) try again
				if retryOnZeroCoeffs:
					if len(np.nonzero(cfier.coef_)[0]) <= 0:
						if retries < maxRetries:
							retryNewAlpha = True
							retrySubSample = True
							vote = vote - 1
							vWeights.pop()
							retries += 1
					else:
						numSubSample = len(iterKnown)

						# NOTE: featScores only valid if numIterations == 1
						featWeights = cfier.coef_
						# Norm scores to (0,1) and preserve each round
						normMaxScore = np.amax(np.abs(featWeights)) + 1e-6
						featScores[:, vote] = np.divide(featWeights, normMaxScore)
				# end if
				voteScores[:, vote] = cfPredLabel

				vote += 1
			# end loop (vote)


			# 6) Place the scores into the array and store across iterations

			# first, average across the normalized random negative samples (votes)
			voteScores = normalizeFeatureColumns(voteScores)
			voteAvgScore = np.mean(voteScores, axis=1)

			# then, place into full gene score array
			#	NOTE: carry the scores forward from each iteration
			for g in range(len(iterAll)):
				geneScores[iterAll[g], itr] = voteAvgScore[g]
			for i in range(itr + 1, numIterations):
				geneScores[:, i] = geneScores[:, itr]
			# end loop


			# 7) Select Known & Unknown for the next round/iteration
			#	for now, just take a percentage of least-confident scores

			# find the cutoff value for scores to keep
			#			idxKeep = len(iterAll) - int(len(iterAll) / float(numIterations))
			cutoffIdx = iterNumGenes - int(iterNumGenes / float(numIterations))
			absScore = np.absolute(voteAvgScore)
			absScore.sort()
			cutoffVal = absScore[cutoffIdx]

			# extract indices for any genes scoring less than cutoff
			iterKeep = list()
			for x in range(len(iterAll)):
				if abs(voteAvgScore[x]) < cutoffVal:
					iterKeep.append(iterAll[x])
			# end loop

			# find intersections of Keep w/ previous Known & Unknown
			setKeep = set(iterKeep)
			newKnown = [gi for gi in iterKnown if gi in setKeep]
			newUnknown = [gi for gi in iterUnknown if gi in setKeep]

			# set the gene indices for the next iteration
			iterKnown = newKnown
			iterUnknown = newUnknown
			iterAll = list()
			iterAll.extend(iterKnown)
			iterAll.extend(iterUnknown)
			iterAll.sort()

			numKnown = len(iterKnown)
			numUnknown = len(iterUnknown)
			if (numKnown <= numExitKnown) or (numUnknown <= numExitUnknown):
				if VERBOSITY:
					print("known: {}, unknown: {}; exiting loop".format(numKnown, numUnknown))
				break
		# end loop (itr)


		# 8) Rank the genes across the iterations
		ranker = aggregateRankFromScore(voteScores[giUnknown, :], vWeights)
		ranker.sort(order=['rowIdx'])
		ranker['rowIdx'] = giUnknown
		ranker.sort(order=['rankSum'])


		# 9) Output the ranked genes to file

		# write the file
		fName = 'ranked_genes-' + useLabel + '_Avg.txt'
		if VERBOSITY:
			print("  Saving ranked genes to file {}".format(fName))
		with open(si + fName, 'w') as fout:
			firstRow = True
			for row in range(len(ranker)):
				if not firstRow:
					fout.write('\n')
				fout.write('{}\t{}\t{}\t{}'.format(
					ranker['score'][row],
					geneNames[ranker['rowIdx'][row]],
					(row + 1),
					ranker['rankSum'][row]))
				firstRow = False
		# end with

		# This time, include genes in the Known set
		ranker = aggRankFromStandardizedScore(voteScores, vWeights)

		#	Output the ranked genes to file
		fName = 'ranked_all_genes-' + useLabel + '_Avg.txt'
		if VERBOSITY:
			print("  Saving ranked genes to file {}".format(fName))
		with open(si + fName, 'w') as fout:
			firstRow = True
			for row in range(len(ranker)):
				if not firstRow:
					fout.write('\n')
				fout.write('{}\t{}\t{}\t{}'.format(
					ranker['score'][row],
					geneNames[ranker['rowIdx'][row]],
					(row + 1),
					ranker['rankSum'][row]))
				firstRow = False
		# end with


		if numIterations > 1:
			# 10-b) Rank the genes from the LAST (final) iteration
			useScore = geneScores[giUnknown, itr]

			ranker = np.recarray(len(giUnknown),
			                     dtype=[('inverse', 'f4'), ('score', 'f4'), ('geneIdx', 'i4')])
			ranker['score'] = useScore
			ranker['inverse'] = np.multiply(useScore, -1)
			ranker['geneIdx'] = giUnknown
			ranker.sort(order=['inverse', 'geneIdx'])

			# 11-b) Output the ranked genes to file
			# write the file
			fName = 'ranked_genes-' + useLabel + '_Last.txt'
			if VERBOSITY:
				print("  Saving ranked genes to file {}".format(fName))
			with open(si + fName, 'w') as fout:
				firstRow = True
				for row in range(len(ranker)):
					if not firstRow:
						fout.write('\n')
					fout.write('{:3.3f}{}{}'.format(ranker['score'][row],
					                                TEXT_DELIM, geneNames[ranker['geneIdx'][row]]))
					firstRow = False
			# end with

			# 10-c) Rank the genes from the FIRST iteration
			useScore = geneScores[giUnknown, 0]
			ranker = np.recarray(len(giUnknown),
			                     dtype=[('inverse', 'f4'), ('score', 'f4'), ('geneIdx', 'i4')])
			ranker['score'] = useScore
			ranker['inverse'] = np.multiply(useScore, -1)
			ranker['geneIdx'] = giUnknown
			ranker.sort(order=['inverse', 'geneIdx'])
			# 11-b) Output the ranked genes to file
			# write the file
			fName = 'ranked_genes-' + useLabel + '_First.txt'
			if VERBOSITY:
				print("  Saving ranked genes to file {}".format(fName))
			with open(si + fName, 'w') as fout:
				firstRow = True
				for row in range(len(ranker)):
					if not firstRow:
						fout.write('\n')
					fout.write('{}{}{}'.format(ranker['score'][row],
					                           TEXT_DELIM, geneNames[ranker['geneIdx'][row]]))
					firstRow = False
			# end with
		# end if (numIterations)


		# 10) Output the selected feature info to file
		featAggScr = np.sum(np.multiply(featScores, np.reshape(vWeights, (1, len(vWeights)))), axis=1)
		featAggScr = np.divide(featAggScr, np.sum(vWeights))

		# Output the normed feature coeff scores to a file
		fName = 'scored_features-' + useLabel + '.txt'
		if VERBOSITY:
			print("  Saving feature scores to file {}".format(fName))
		with open(si + fName, 'w') as fout:
			# First line: the weights for each model
			fout.write(" \t \tweights:")
			for col in range(numVotes):
				fout.write("\t{}".format(vWeights[col]))
			fout.write("\n")

			# each row: agg score, path length, path name, individual scores
			fout.write("{}".format(featAggScr[0]))
			fout.write("\t{}".format(featNames[0].count('-') + 1))
			fout.write("\t{}".format(featNames[0]))
			for v in range(numVotes):
				fout.write("\t{}".format(featScores[0, v]))
			# end for
			for row in range(1, len(featNames)):
				fout.write("\n{}".format(featAggScr[row]))
				fout.write("\t{}".format(featNames[row].count('-') + 1))
				fout.write("\t{}".format(featNames[row]))
				for v in range(numVotes):
					fout.write("\t{}".format(featScores[row, v]))
		# end with


		# 11) Output the parameters to file
		fName = 'parameters-' + useLabel + '.txt'
		with open(si + fName, 'w') as fout:
			fout.write('\n')
			fout.write('Sampling Method for Neg examples\n')
			fout.write('  as One-Class w/ iterations on the weaker predictions\n')
			fout.write('\n')

			fout.write('Classifier Parameters\n')
			fout.write('method:{}Lasso\n'.format(TEXT_DELIM))
			fout.write('positive:{}{}\n'.format(TEXT_DELIM, usePos))
			fout.write('alpha mean: {}{}\n'.format(TEXT_DELIM, np.mean(usedAlphas)))
			fout.write('alpha range:{}{}\n'.format(TEXT_DELIM, useGivenRange))
			fout.write('alpha chosen:{}{}\n'.format(TEXT_DELIM, usedAlphas))
			fout.write('iters to convg:{}{}\n'.format(TEXT_DELIM, usedIterConvergence))
			fout.write('max_iter:{}{}\n'.format(TEXT_DELIM, lMaxIter))
			fout.write('normalize:{}{}\n'.format(TEXT_DELIM, lNorm))
			fout.write('fit_intercept:{}{}\n'.format(TEXT_DELIM, lFitIcpt))
			fout.write('\n')

			fout.write('model r2 score:{}{}\n'.format(TEXT_DELIM, vWeights))
		# end with

		tSample = (time.time() - tStartSample) / 3600.0
		usedTimePerSample.append(tSample)
		if VERBOSITY:
			print("--{} of {}".format(thisRound, len(dSubDirs)))
			print("--elapsed time: {:.2f} (h)".format(tSample))
	# end loop (si in dSubDirs)

	fName = 'classifier-' + useLabel + '.txt'
	if VERBOSITY:
		print("\nSaving classifier stats to:")
		print("  {}...  {}".format(sDir, fName))
	with open(sDir + '/' + fName, 'w') as fout:
		fout.write('\n')
		fout.write('Sampling Method for Neg examples\n')
		fout.write('  as One-Class w/ iterations on the weaker predictions\n')
		fout.write('\n')

		fout.write('Mean Used :{}{}\n'.format(TEXT_DELIM, np.mean(usedNumFeatures)))
		fout.write('Median Used:{}{}\n'.format(TEXT_DELIM, np.median(usedNumFeatures)))
		fout.write('Standard Dev:{}{}\n'.format(TEXT_DELIM, np.std(usedNumFeatures)))
		fout.write('Min Used :{}{}\n'.format(TEXT_DELIM, np.amin(usedNumFeatures)))
		fout.write('Max Used :{}{}\n'.format(TEXT_DELIM, np.amax(usedNumFeatures)))
		fout.write('\n')

		fout.write('Time to classify (hours per sample)\n')
		fout.write('Mean Used :{}{}\n'.format(TEXT_DELIM, np.mean(usedTimePerSample)))
		fout.write('Median Used:{}{}\n'.format(TEXT_DELIM, np.median(usedTimePerSample)))
		fout.write('Standard Dev:{}{}\n'.format(TEXT_DELIM, np.std(usedNumFeatures)))
		fout.write('Min Used :{}{}\n'.format(TEXT_DELIM, np.amin(usedTimePerSample)))
		fout.write('Max Used :{}{}\n'.format(TEXT_DELIM, np.amax(usedTimePerSample)))

		fout.write('Classifier Parameters\n')
		fout.write('method:{}Lasso\n'.format(TEXT_DELIM))
		fout.write('positive:{}{}\n'.format(TEXT_DELIM, usePos))
		fout.write('max_iter:{}{}\n'.format(TEXT_DELIM, lMaxIter))
		fout.write('normalize:{}{}\n'.format(TEXT_DELIM, lNorm))
		fout.write('fit_intercept:{}{}\n'.format(TEXT_DELIM, lFitIcpt))
		fout.write('\n')

		fout.write('mean alpha:{}{}\n'.format(TEXT_DELIM, np.mean(usedAlphasAll)))
		fout.write('median alpha:{}{}\n'.format(TEXT_DELIM, np.median(usedAlphasAll)))
		fout.write('mean iters:{}{}\n'.format(TEXT_DELIM, np.mean(usedIterConvergenceAll)))
		fout.write('median iters:{}{}\n'.format(TEXT_DELIM, np.median(usedIterConvergenceAll)))
		fout.write('alpha range:{}{}\n'.format(TEXT_DELIM, useGivenRange))
		fout.write('alpha chosen:{}{}\n'.format(TEXT_DELIM, usedAlphasAll))
		fout.write('iters to convg:{}{}\n'.format(TEXT_DELIM, usedIterConvergenceAll))
		fout.write('\n')
	# end with

	rankedListName = 'ranked_genes-' + useLabel + '.txt'

	return rankedListName
#end def #################################


def calcAndDrawAUCs(pathData, doDrawAUCs, flagVerbose):
	"""
	Calculate AUC values for all folds
		Find average AUC across all folds for each set

	:param pathData: str, path to output files (eg: ./output/batch-009
	:param doDrawAUCs: bool, whether to save AUC plots
	:param flagVerbose: int, level of verbose output to terminal
	:return:
	"""
	setParamVerbose(flagVerbose)


	# 1) Load the path names

	# Read the network location from parameters file
	pathData = pathData.rstrip('/') + '/'
	with open(pathData + 'parameters.txt', 'r') as fin:
		line = fin.readline()

		line = fin.readline()
		line = line.rstrip()
		lv = line.split(TEXT_DELIM)
		eName = lv[1]

		line = fin.readline()
		line = line.rstrip()
		lv = line.split(TEXT_DELIM)
		ePath = lv[1]
	#end with

	if VERBOSITY > 1 :
		print("Reading in the path names.")
	pathDict = readKeyFile(ePath, eName)
	pathList = removeInvertedPaths(pathDict)
	del pathDict

	# Make new pathDict to give index of pathList
	pathDict = dict()
	idx = -1
	for item in pathList:
		idx += 1
		pathDict[item] = idx
	#end loop


	# 2) Get the list of unique samples
	dSubDirs = getSubDirectoryList(pathData)
	partSubDirs = set()
	sampSet = set()
	prevSN = ''
	numFolds = 0
	prevNumFolds = 0
	for si in dSubDirs:

		# extract sample name from folder name
		sv = si.split('/')
		sDir = sv[-2]
		sdv = sDir.split('-')
		if re.search('part-', si):
			partSubDirs.add(si)
		if sDir.startswith('part-') or sDir.startswith('full-'):
			sn = sdv[1]
		else:
			sn = sdv[0]
		sampSet.add(sn)

		# count how many folds were created
		if sn != prevSN:
			prevNumFolds = numFolds
			numFolds = 1
		else:
			numFolds += 1
		#end if
		prevSN = sn
	#end loop
	sampList = list(sampSet)
	sampList.sort()
	partSubDirsList = list(partSubDirs)
	partSubDirsList.sort

	if VERBOSITY:
		print("Number of cross-validation folds found: {}".format(numFolds))

	if VERBOSITY:
		print("There are {} subdirectories ...".format(len(dSubDirs)))




	# 3) Create the matrices to hold the results

	# Get list of files in the subdirectory
	fileList = os.listdir(dSubDirs[0])
	fileList.sort()

	# ERROR CHECK: ensure ranked_genes & ranked_paths files exist
	rg = 0
	sf = 0
	for item in fileList:
		iv = item.split('/')
		if iv[-1].startswith('ranked_genes'):
			rg += 1
		elif iv[-1].startswith('scored_features'):
			sf += 1
	# end loop
	if rg < 1:
		print("ERROR: No ranked_genes... files exist in {}".format(dSubDirs[0]))
		sys.exit
	elif sf < 1:
		print("WARNING: No scored_features... files exist in {}".format(dSubDirs[0]))
	# end if

	# Get list of the methods used from the file names
	methodList = list()
	for item in fileList:
		iv = item.split('/')
		fn = iv[-1]
		if fn.startswith('ranked_genes'):
			fn = fn[0:-4]
			fv = fn.split('-')
			methodList.append(fv[1])
	#end loop
	methodList.sort()

	if VERBOSITY:
		print("  ... each containing {} different experiments.".format(len(methodList)))

	resultsROC = np.zeros((len(methodList), len(partSubDirs)), dtype=np.float64)
	resultsAvgROC = np.zeros((len(methodList), len(sampList)), dtype=np.float64)


	# 4) Create the Area Under the Curve tables
	if VERBOSITY :
		print("Finding AUCs for each sample ...")

	col = -1
	for si in partSubDirsList:
		col += 1

		if not (col % 20) and VERBOSITY:
			if VERBOSITY > 1 :
				print("beginning subdirectory {}".format(col))

		# Get data relating to each method
		row = -1
		for m in methodList:
			row += 1

			fn = 'ranked_genes-' + m + '.txt'
			FPR, recall, precision, numHid = getAUCStats(si, fn)

			# Calculate (approximate) are under the ROC curve
			areaROC = 0
			for r in recall:
				areaROC += (r / float(len(recall)))
			#end loop

			# save data into the matrix
			resultsROC[row, col] = areaROC

			if doDrawAUCs:
				# Save the AUC figure(s)
				outName2 = si + 'AUC-' + m + '.png'
				fig = plt.figure()

				# Plot the ROC curve
				plt.plot(FPR, recall)
				plt.plot([0, 1], [0, 1], 'lightgrey')
				plt.xlabel('False Positive Rate')
				plt.ylabel('True Positive Rate')
				plt.axis([0, 1, 0, 1])

				# Final touches
				sv = si.split('/')
				sdir = sv[-2]
				sdv = sdir.split('-')
				plt.suptitle(sdv[1] + '\nconcealed = {}'.format(numHid) +
				             ', ROC area = {:.3}'.format(float(areaROC)))
				plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.4, hspace=None)

				# Save the figure
				plt.savefig(outName2)
				plt.close()
			#end if
		#end loop (methodList)
	#end loop (dSubDirs)

	if VERBOSITY:
		print("Finished collecting results.")


	# Get average AUC for each sample, across all folds
	for i in range(len(sampList)):
		left = i * numFolds
		right = left + numFolds

		resultsAvgROC[:, i] = np.mean(resultsROC[:, left:right], axis=1)
	#end loop


	# 5) Output to file(s)
	if VERBOSITY:
		print("Writing the AUC tables ...")

	# Write the AUC tables to file(s) for the COMPLETE results (every fold)
	fileNames = 'results-AUCs_every_fold.txt'
	fileHeaders = 'Area Under ROC Curve, all folds'
	fileData = resultsROC
	with open(pathData + fileNames, 'w') as fOut:
		fOut.write(fileHeaders)
		fOut.write('\nnetwork:{}{}'.format(TEXT_DELIM, eName))
		fOut.write('\nfolds:{}{}'.format(TEXT_DELIM, numFolds))
		fOut.write('\n\n')

		for j in range(len(dSubDirs)):
			sv = dSubDirs[j].split('/')
			fOut.write('{}{}'.format(sv[-2], TEXT_DELIM))
		# fOut.write('\n')

		for i in range(len(methodList)):
			fOut.write('\n')
			for j in range(len(dSubDirs)):
				fOut.write('{}{}'.format(fileData[i, j], TEXT_DELIM))
			fOut.write('{}'.format(methodList[i]))
	#end with

	# Write the AUC tables to file(s) for the per-sample results
	fileNames = 'results-AUC_mean_by_set.txt'
	fileHeaders = 'Average -- Area Under ROC Curve, per-sample'
	fileData = resultsAvgROC
	with open(pathData + fileNames, 'w') as fOut:
		fOut.write(fileHeaders)
		fOut.write('\nnetwork:{}{}'.format(TEXT_DELIM, eName))
		fOut.write('\nfolds:{}{}'.format(TEXT_DELIM, numFolds))
		fOut.write('\n\n')

		for j in range(len(sampList)):
			fOut.write('{}{}'.format(sampList[j], TEXT_DELIM))
		# fOut.write('\n')

		for i in range(len(methodList)):
			fOut.write('\n')
			for j in range(len(sampList)):
				fOut.write('{}{}'.format(fileData[i, j], TEXT_DELIM))
			fOut.write('{}'.format(methodList[i]))
	#end with

	return fileNames
# end def #################################


################################################################
# MAIN FUNCTION & CALL
def main(params, passedDir = '') :

	##########################################
	# set parameter values & IO file locations
	#   read in the command-line arguments
	#   convert to file names and paths as appropriate

	# # Collect parameters from the command line
	# params = readCommandLineFlags()

	ignoreList = list()
	# if params.ignore is not "NONE" :
	# 	#TODO: read in ignoreList from file
	# #end if

	numVotes = params.numModels
	if numVotes < 1 : #TODO: warn user
		numVotes = 11
	#end if


	if passedDir == '' :
		setsPath = params.setsRoot
	else :
		setsPath = passedDir.rstrip('/') + '/'
	#end if

	##########################################
	# characterize the input set(s) using the pre-processed network
	fnRanking = predictIterative(ignoreList, params.length, setsPath, numVotes, params.verbose)

	if params.verbose :
		print("\nRanked list of non-input genes saved to: ")
		print("    {}full-.../{}".format(setsPath, fnRanking))
	#end if

	##########################################
	# calculate AUC scores for each fold & find average per set
	fnAUCs = calcAndDrawAUCs(setsPath, params.plotAUCs, params.verbose)

	if params.verbose :
		print("\nAUC values for each set saved to: ")
		print("    {}full-.../{}".format(setsPath, fnAUCs))
	#end if


	return fnRanking
# end def #################################


if __name__ == "__main__":
	print("\nRunning GeneSet MAPR set characterization ...")

	# Collect parameters from the command line
	params = readCommandLineFlags()

	main(params)

	print("\nDone.\n")
# end if
