"""
GeneSet MAPR implementation
	Step 02: build features from the network
	
For an input set (or batch of sets), condense all
  of the node-node meta-paths in the network into
  a single feature matrix per input set.

author: Greg Linkowski
consolidated by Aamir Hasan
	for KnowEnG by UIUC & NIH
"""

import time
import numpy as np
import random
import os
from os import listdir
import sys
import gzip
import argparse



################################################################
# GLOBAL PARAMETERS

# Default path to output files
OUTPUT_ROOT = './output'
# Default path to network files
NETWORK_ROOT = './networks'
# size the zero-padded matrix file name (match MAPR_networkPrep !)
FNAME_ZPAD = 6
# Data-type for the path matrices (allow room for computation)
MATRIX_DTYPE = np.float32
# File extension to use when saving the matrix
MATRIX_EXTENSION = '.gz'	# '.txt' or '.gz' (gz is compressed)

# end params ##############################



################################################################
# ANCILLARY FUNCTIONS

def readCommandLineFlags():
	parser = argparse.ArgumentParser()

	parser.add_argument('networkName', type=str,
						help='name of the input network') # same as input network directory
	parser.add_argument('-v', '--verbose', type=int, default=0,
						help='enable verbose output to terminal: 0=none, 1=all')
	parser.add_argument('-f', '--folds', type=int, default=4,
						help='number of cross-validation folds')
	parser.add_argument('-o', '--output', type=str, default=OUTPUT_ROOT,
						help='output directory to store processed network')
	parser.add_argument('-s', '--sample', type=str, default='samples/',
						help='samples directory')
	parser.add_argument('-n', '--networkPath', type=str, default=NETWORK_ROOT,
						help='alternative processed network directory')

	flags = parser.parse_args()

	return flags
# end def #################################


def verifyDirectory(path, create, quiet):
	"""
	ERROR CHECK: verify directory exists
	
	:param path: str, path to save the file
	:param create: bool, whether to create missing dir
	:param quiet: bool, whether to quietly return T/F
	:return: exists, bool, indicates existence of directory
	"""
	exists = True
	if not os.path.isdir(path):
		exists = False
		if create:
			os.makedirs(path)
			if not quiet:
				print("Creating path: {}".format(path))
		elif not quiet:
			print("ERROR: Specified path doesn't exist:" +
				  " {}".format(path))
			sys.exit()
	# end if

	return exists
# end def #################################


def verifyFile(path, name, quiet):
	"""
	ERROR CHECK: verify file exists
	
	:param path: str, path to save the file
	:param name: str, name of the file (w/ extension)
	:param quiet: bool, whether to quietly return T/F
	:return: exists, bool, indicates existence of file
	"""
	exists = True

	# First check the directory
	if path != '':
		exists = verifyDirectory(path, False, quiet)
		# Then concatenate path & name
		if not path.endswith('/'):
			path = path + '/'
		fName = path + name
	else:
		fName = name
	# end if

	# Then look for the file
	if not os.path.isfile(fName):
		if quiet:
			exists = False
		else:
			print("ERROR: Specified file doesn't exist:" +
				  " {}".format(fName))
			sys.exit()
	# end if

	return exists
# end def #################################


def concatenatePaths(root, subDir):
	"""
	Combine a root directory with a sub-directory
	
	:param root: str, the root directory
	:param subDir: str, the sub-directory
	:return: path, str, the full combined path
	"""
	if (not root.endswith('/')) and (root != ''):
		root = root + '/'
	if (not subDir.endswith('/')) and (subDir != ''):
		subDir = subDir + '/'
	# end if
	path = root + subDir

	return path
# end def #################################


def readFileAsIndexDict(fName):
	"""
	Read in the gene file. File is an ordered list
	  of genes, where the row number (starting at zero)
	  corresonds to the index in the matrix/list/etc where
	  the gene can be found.
	
	:param fName: str, path & name to gene file
	:return: iDict, dict,
      key, str: gene name as read from file
      value, int: index to corresponding array
	"""
	verifyFile('', fName, False)

	# Build the dictionary from the text file
	iDict = dict()
	gf = open(fName, "r")
	index = 0
	for line in gf:
		gene = line.rstrip()  # remove "\n"
		iDict[gene] = int(index)
		index += 1
	# end loop

	return iDict
# end def #################################


def nameOutputPath(path, dirPre):
	"""
	Choose an unused name for the output path
	
	:param path: str, path to where output should be saved
	:param dirPre: str, prefix of the folder name to return
	:return: dirFull, str, name of output file (without path)
	"""
	# ERROR CHECK: verify directory exists
	verifyDirectory(path, False, False)

	zpad = 3

	# Set of all sub-folders in the path
	dirSet = [name for name in os.listdir(path)
			  if os.path.isdir(path + name)]

	# increment folder name until an unused one is found
	num = int(0)
	dirFull = dirPre + "-{}".format(str(num).zfill(zpad))
	while dirFull in dirSet:
		num += 1
		dirFull = dirPre + "-{}".format(str(num).zfill(zpad))
	# end loop
	dirFull = dirFull + '/'

	return dirFull
# end def #################################


def writeGenericLists(path, fName, columnList):
	"""
	Write a text file where the columns are given as lists
	Creates ranked_paths.txt, original version of the output file
	
	:param path: str, directory to write output file
	:param fName: str, name of the file to write
	:param columnList: list of str lists,
		each entry in columnList represents a column
		where each entry is a string to write to the file
	:return:
	"""
	verifyDirectory(path, True, False)

	# ASSUME: the contained lists are of equal length

	fOut = open(path + fName, 'w')

	for i in range(len(columnList[0])):
		fOut.write("{}".format(columnList[0][i]))

		for j in range(1, len(columnList)):
			fOut.write("{}{}".format('\t', columnList[j][i]))
		# end loop

		fOut.write("\n")
	# end if

	fOut.close()

	return
# end def #################################


def getGeneDictionary(path, name):
	"""
	Read in the genes.txt file containing the
		gene-name headers to the meta-path matrices
	Pre-reqs: readFileAsIndexDict(fName)
	
	:param path: str, path to the network files
	:param name: str, name of the network to use
	:return: gDict, dict,
      key, str: name of gene
      value, int: row/col index for that gene
	"""
	fname = concatenatePaths(path, name)
	fname = fname + "genes.txt"

	# The item to return
	gDict = readFileAsIndexDict(fname)

	return gDict
# end def #################################


def getPathDictionary(path, name):
	"""
	Read in the key.txt file regarding the
        metapath matrices
	
	:param path: str, path to the network files
	:param name: str, name of the network to use
	:return: keyDict, dict,
      key, str: name of metapath
      value, tuple: int is matrix/file ID number
          bool where True means use matrix transpose
	"""
	fname = path + name + "_MetaPaths/key.txt"

	# ERROR CHECK: verify file exists
	if not os.path.isfile(fname):
		print("ERROR: Specified file doesn't exist:" +
			  " {}".format(fname))
		sys.exit()
	# end if

	# The item to return
	keyDict = dict()

	# Read in the file
	fk = open(fname, "r")
	firstline = True
	for line in fk:

		# skip the first line
		if firstline:
			firstline = False
			continue
		# end if

		# separate the values
		line = line.rstrip()
		#		print(line)
		lk = line.split('\t')
		lv = lk[0].split(',')

		transpose = False
		if lv[1] == "t":
			transpose = True
		# end if

		# add to the dict
		keyDict[lk[1]] = [int(lv[0]), transpose]
	# end loop
	fk.close()

	return keyDict
# end def #################################


def removeInvertedPaths(mpDict):
	"""
	Find the number of paths of this type joining
        the nodes in the sample
	
	:param mpDict: dict {str: [int, bool]},
      key, str - name of the metapath
      value, [int, bool] - which matrix file to use, and
          whether to use the transpose (inverse path)
	:return: mpList, str list, ordered names of paths available,
      less paths that are mirror-images of another
	"""
	# The item to return
	mpList = list()

	# Check the keys in the dict
	for key in mpDict.keys():
		# If the boolean is True, then the path is an
		#   inverse of another; only append if false
		if mpDict[key][1] == False:
			mpList.append(key)
	# end loop

	mpList.sort()
	return mpList
# end def #################################


def getPathMatrixSize(ePath, eName):
	"""
	Count number of rows in a metapath matrix
		(num rows = num cols)
	
	:param ePath: str, path to network
	:param eName: str, folder containing processed network files
	:return: mxSize, int, number of rows/columns in path matrix
	"""
	# the item to return
	mxSize = 0

	# open and read through the file
	fname = (ePath + eName + "_MetaPaths/" +
			 "{}.gz".format(str(0).zfill(FNAME_ZPAD)))
	with gzip.open(fname, 'rb') as fin:
		for line in fin:
			mxSize += 1
	# end with

	return mxSize
# end def #################################


def getSampleNamesFromFolder(path):
	"""
	Create a list of samples contained in folder
	
	:param path: str, path where samples stored
	:return: sNames, str list, sorted list of sample names
	"""
	verifyDirectory(path, False, False)
	
	# Get list of all text files in folder
	fNames = [f for f in listdir(path) if f.endswith('.txt')]
	
	#TODO: throw an error if no samples found

	# Identify & create list of sample names in folder
	sNames = list()
	for item in fNames:
		# Strip the extension and any "_UP" or "_DN"
		newItem = item[:-4]
		if newItem.endswith('_UP') or newItem.endswith('_DN'):
			newItem = newItem[:-3]
		# end if
		sNames.append(newItem)
	# end loop

	sNames = np.unique(sNames)  # also sorts list
	return sNames
# end def #################################


######## ######## ######## ########
# Function: Read in a file as a line-by-line list of items
# Input ----
#   fname, str: path + name of the the sample files
# Returns ----
#   fItems, str list: ordered list of items from file
def readFileAsList(fname):
	"""
	Read in a file as a line-by-line list of items
	
	:param fname: str, path + name of the the sample files
	:return: fItems, str list, ordered list of items from file
	"""
	# ERROR CHECK: verify file exists
	verifyFile('', fname, False)

	# The list of items to return
	fItems = list()

	# Read in from the file
	fn = open(fname, "r")
	for line in fn:
		fItems.append(line.rstrip())
	# end loop
	fn.close()

	fItems.sort()
	return fItems
# end def #################################


def readSampleFiles(sFile, up, down):
	"""
	Read in the dataset from a samplename
    Check for variants: ".txt", "_UP.txt", "_DN.txt"
	Pre-reqs: readFileAsList(fname)
	
	:param sFile: str, path + name of the the sample files
	:param up: bool, only read the _UP file if true
	:param down: bool, only read the _DN file if true
	:return: sNodes, str list, ordered list of names from file(s)
	"""
	# The list of items to return
	sNodes = list()

	# Flag indicates a file existed and was read
	exists = False

	# First look for the file as named (no _UP or _DN)
	if os.path.isfile(sFile + ".txt"):
		temp = readFileAsList(sFile + ".txt")
		sNodes.extend(temp)
		exists = True
	# end if

	# Look for the _DN file
	if down and os.path.isfile(sFile + "_DN.txt"):
		temp = readFileAsList(sFile + "_DN.txt")
		sNodes.extend(temp)
		exists = True
	# end if

	if up and os.path.isfile(sFile + "_UP.txt"):
		temp = readFileAsList(sFile + "_UP.txt")
		sNodes.extend(temp)
		exists = True
	# end if

	# Alert user if nothing was read in
	if not exists:
		print("WARNING: no file found: {}".format(sFile))

	# Do NOT return duplicates
	uNodes = np.unique(sNodes)  # sorted list of unique items
	return uNodes
# end def #################################


def checkListAgainstDictKeys(theList, theDict):
	"""
	Given a list of items, remove any items not in specified dict
	
	:param theList: list, list of items that may not be in theDict
	:param theDict: dict, dictionary against which to check (the keys)
	:return:
      inlist, list, items from list found in dict keys
      outlist, list, items from list NOT found in dict keys
	"""
	# The items to return
	inList = list()
	outList = list()

	# extract the keys as a set
	keySet = set(theDict.keys())

	# Sift through the sample
	for item in theList:
		if item in keySet:
			inList.append(item)
		else:
			outList.append(item)
	# end if
	# end loop

	inList.sort()
	outList.sort()
	return inList, outList
# end def #################################


def saveListToText(path, name, theList):
	"""
	save a list to a text file
	Creates a file containing ordered list of items
	
	:param path: str, path to save the file
	:param name: str, name of file to save
	:param theList: list of str, list of items to save
      ASSUMPTION: list is already properly ordered
	:return:
	"""
	# If folder doesn't exist, create it
	if not os.path.exists(path):
		os.makedirs(path)
	# end if

	theFile = open(path + name, 'w')
	firstLine = True
	for item in theList:
		if firstLine:
			firstLine = False
		else:
			theFile.write("\n")
		# end if
		theFile.write("{}".format(item))
	# end if
	theFile.close()

	return
# end def #################################



def getPathMatrix(mpTuple, path, name, sizeOf):
	"""
	Load the matrix containing the number of paths
	  of this type which join the nodes in the network

	:param mpTuple:[int, bool]: indicates which matrix file to use
	:param path: str, path to the network files
	:param name: str, name of the network to use
	:param sizeOf: int, dimensions of (square) array
	:return: matrix, int array, num paths between node pairs
	"""

	prename = (path + name + "_MetaPaths/" +
			   "{}".format(str(mpTuple[0]).zfill(FNAME_ZPAD)))
	if os.path.isfile(prename + '.gz'):
		fname = (prename + '.gz')
	elif os.path.isfile(prename + '.txt'):
		fname = (prename + '.txt')
	else:
		# ERROR CHECK: verify file exists
		print("ERROR: Specified file doesn't exist:" +
			  " {} .gz/.txt".format(prename))
		sys.exit()
	# end if

	# Declare the matrix
	matrix = np.zeros([sizeOf, sizeOf], dtype=MATRIX_DTYPE)

	# Read in the file, placing values into matrix
	row = 0
	with gzip.open(fname, 'rb') as fin:
		for line in fin:
			line = line.rstrip()
			ml = line.split()
			matrix[row, :] = ml[:]
			row += 1
	# end with

	# Convert to transpose if flag==True
	if mpTuple[1]:
		return np.transpose(matrix)
	else:
		return matrix
# end def #################################


def isPathSymmetric(mpName):
	"""
	Return T/F if given meta-path is symmetric
	
	:param mpName: str, name of meta-path
	:return: symmetric, bool, result of test
	"""
	mpNList = mpName.split('-')
	mpNLen = len(mpNList)

	symmetric = True
	if mpNLen > 1:

		a = 0
		b = mpNLen - 1
		while b > a:
			# print(mpNList[a], mpNList[b])
			if not (mpNList[a] == mpNList[b]):
				symmetric = False
				break
			a = a + 1
			b = b - 1
		# end loop
	# end if

	return symmetric
# end def #################################


def saveMatrixNumpy(matrix, mxName, mxPath, flagAsInt):
	"""
	Save given matrix as a .npy file
	
	:param matrix: (NxN) list: the values to save
	:param mxName: str: name of the file to save
	:param mxPath: str: path to the folder to save the file
	:param flagAsInt: bool: True means save values as int()
	:return:
	"""
	# If folder doesn't exist, create it
	if not os.path.exists(mxPath):
		os.makedirs(mxPath)
	# end if

	# Write to the file
	if flagAsInt:
		np.savetxt(mxPath + mxName + MATRIX_EXTENSION, matrix, fmt='%u')
	else:
		np.savetxt(mxPath + mxName + MATRIX_EXTENSION, matrix, fmt='%f')
	# end if

	return
# end def #################################


def processFolderName(parentDirectory, outputDirectory):
	"""
	Creates new output directory
	Appends number to dir name to make it unique
	
	:param parentDirectory: str, base path to where dir will exist
	:param outputDirectory: str, base name of the subdir to create
	:return:
	"""
	
	parentDirectory = parentDirectory.rstrip('/') + '/'
	outputDirectory = outputDirectory.rstrip('/')
	
	output = parentDirectory + outputDirectory + '_'

	# getting all output directories with same name as requested output directory
	directories = [int(x.replace(outputDirectory + '_', '')) for x in os.listdir(parentDirectory) if x.startswith(outputDirectory)]

	# setting the output directory number
	output += str(len(directories)).zfill(4) + '/'

	os.mkdir(output)

	return output
# end def #################################



################################################################
# PRIMARY FUNCTION(S)

def createFeatureZScore(eName, ePath, sDir, oRoot, numFolds, verbosity):
	"""
	Partition the samples and create z-score features
	Creates dir in oDir containing folder for each sample
	  (whole + 4 folds), containing gene partitions, and
	  a matrix of z-score feature vectors for that sample
	
	:param eName: str, folder containing network files
	:param ePath: str, path to the network folder
	:param sDir: str, directory containing samples
	:param oRoot: str, output directory
	:param verbosity: bool, whether to enable terminal output
	:return:
	"""
	ePath = ePath.rstrip('/') + '/'
	oRoot = oRoot.rstrip('/') + '/'
	
	# 1) Name & create a folder to store output files
	oSubDir = nameOutputPath(oRoot, 'batch')
	if verbosity:
		print("Files will be saved to {}".format(oSubDir))
	oDir = oRoot + oSubDir

	# Save experiment parameters to file
	fOutput = list()
	fOutput.append(['date', 'network', 'ntwk path', 'cross-val folds', 'samples'])
	fOutput.append([time.strftime('%d/%m/%Y'), eName, ePath, numFolds, sDir])
	fOutputName = 'parameters.txt'
	writeGenericLists(oDir, fOutputName, fOutput)
	
	# 2a) Load the gene-index dict
	if verbosity:
		print("Creating the gene-index dictionary.")
	geneDict = getGeneDictionary(ePath, eName)

	# 2b) Get the list of available paths
	if verbosity:
		print("Checking what paths are available ...")
	pathDict = getPathDictionary(ePath, eName)
	pathList = removeInvertedPaths(pathDict)
	
	#TODO: save path list (column names) just like genes.txt

	# 2c) Get expected matrix size
	if verbosity:
		print("Finding the matrix dimensions ...")
	mxRows = getPathMatrixSize(ePath, eName)

	# 3) Read & partition samples, save to output dir
	sNames = getSampleNamesFromFolder(sDir)

	oSampLists = list()
	oSubDirList = list()

	# Read samples & create cross-validation folds
	sCount = 0
	for s in sNames:
		sCount += 1
		if verbosity:
			print("Collecting sample: {}, {}".format(sCount, s))

		# Read in genes from the full sample
		# Remove any genes not in the network
		# Convert sample names to indices
		gAll = readSampleFiles(sDir + s, True, True)
		gAllValid, gIgnored = checkListAgainstDictKeys(
			gAll, geneDict)
		giAll = [geneDict[g] for g in gAllValid]

		# Append the full sample to the lists
		oSampLists.append(giAll)
		oSubDir = '{}full-{}/'.format(oDir, s)
		oSubDirList.append(oSubDir)

		# Write the genes to file
		saveListToText(oSubDir, 'known.txt', gAllValid)
		saveListToText(oSubDir, 'concealed.txt', list())
		saveListToText(oSubDir, 'ignored.txt', gIgnored)

		# Create the cross-validation folds
		if numFolds >= 2 :
			random.seed()
			random.shuffle(gAllValid)
			percHide = 1.0 / float(numFolds)
			sizeFold = int(len(gAllValid) * percHide)
			for i in range(numFolds):
				start = i * sizeFold
				stop = (i * sizeFold) + sizeFold
				gHidden = gAllValid[start:stop]
				gHidden.sort()
				gKnown = gAllValid[0:start]
				gKnown.extend(gAllValid[stop:len(gAllValid)])
				gKnown.sort()
	
				# Append this fold to the lists
				giKnown = [geneDict[g] for g in gKnown]
				oSampLists.append(giKnown)
				oSubDir = '{}part-{}-{:02d}/'.format(oDir, s, i)
				oSubDirList.append(oSubDir)
	
				# Write the genes to file
				saveListToText(oSubDir, 'known.txt', gKnown)
				saveListToText(oSubDir, 'concealed.txt', gHidden)
				saveListToText(oSubDir, 'ignored.txt', list())
			# end loop range(numFolds)
		#end if
	# end loop sNames

	# 4) Create the z-score features

	#	Build the feature vector matrices for each sample
	gFeatures = np.zeros((len(geneDict), len(pathList),
						  len(oSampLists)), dtype=np.float32)

	# populate dimension 2 from each path
	dim2 = -1
	for p in pathList:
		dim2 += 1

		tpath = time.time()

		# load the path count matrix
		countMatrix = getPathMatrix(pathDict[p], ePath, eName, mxRows)

		# Create a probability matrix -- likelihood of path going from i -> j
		#	by dividing each row i by degree of gene i (for this MP)
		sumDegree = np.sum(countMatrix, axis=1)
		sumDegree = sumDegree.reshape((mxRows, 1))
		sumDegree = np.add(sumDegree, 1e-5)

		probMatrix = np.divide(countMatrix, sumDegree)
		np.fill_diagonal(probMatrix, 0)

		# populate dimension 3 from each sample
		dim3 = -1
		for giList in oSampLists:
			dim3 += 1

			# Get sum of probability of path ending in gene set
			probSet = np.sum(probMatrix[:, giList], axis=1)
			# if path not symmetric: matrix not symmetric ...
			#	so add sum from transpose, too
			if not isPathSymmetric(p):
				probSetTPose = np.sum(probMatrix[giList, :], axis=0)
				# probSet = np.add(probSet, probSetTPose)
				# TODO: Are A & B mutually exclusive? If not, P(A or B) = P(A) + P(B) - P(A and B)
				#	ie: gene-i connects to gene-j along path A-B-C or C-B-A
				probBoth = np.multiply(probSet, probSetTPose)
				probSet = np.add(probSet, probSetTPose)
				probSet = np.subtract(probSet, probBoth)
			gFeatures[:, dim2, dim3] = probSet[:]

		# end loop

		if verbosity and not (dim2 % 25):
			print("  Examined {} of {} paths...".format(dim2, len(pathList)))
			print("    --time per path: {:.3} (s)".format(time.time() - tpath))

	# end loop
	if verbosity:
		print("Finished examining matrix similarity matrices.")

	# 5) Save the feature matrix for each sub-directory
	tWrite = time.time()
	for i in range(len(oSubDirList)):
		saveMatrixNumpy(gFeatures[:, :, i], 'features_ZScoreSim',
						oSubDirList[i], False)

		# 4.2) Apply z-score across feature columns
		zScoreFeat = gFeatures[:, :, i]

		# TODO: breakout function
		colAvg = np.mean(zScoreFeat, axis=0)
		colAvg = colAvg.reshape((1, len(colAvg)))
		colStD = np.std(zScoreFeat, axis=0)
		colStD = colStD.reshape((1, len(colStD)))
		colStD = np.add(colStD, 1e-5)

		zScoreFeat = np.subtract(zScoreFeat, colAvg)
		zScoreFeat = np.divide(zScoreFeat, colStD)

		# write the file
		saveMatrixNumpy(zScoreFeat, 'features_ZScoreSim',
						oSubDirList[i], False)
	# end loop
	if verbosity:
		print("Finished writing Z-Score Similarity feature vector files.")
		print("    --time to write: {:.3} (s)".format(time.time() - tWrite))

	return oDir
# end def #################################



################################################################
# MAIN FUNCTION & CALL
def main(params, passedName = ''):
	# # getting command line arguments
	# params = readCommandLineFlags()

	if passedName == '' :
		nName = params.networkName
	else :
		splitName = passedName.split('/')
		nName = splitName[-1]
		if nName.endswith('.edge.txt'):
			nName = nName[0:-9]
	#end if
	
	numFolds = params.folds
	verbosity = params.verbose
	nPath = params.networkPath.rstrip('/') + '/'
	sPath = params.sample.rstrip('/') + '/'
	oRoot = params.output
	oRoot = oRoot.rstrip('/') + '/'

	_ = verifyDirectory(nPath, False, False)
	_ = verifyDirectory(sPath, False, False)
	_ = verifyDirectory(oRoot, True, False)
	
	if verbosity :
		print("\n-----------------------------------------")
		print("Building feature vectors from network ...")
		print("  network: {}".format(nName))
		print("  sample set: {}".format(sPath.rstrip('/')))
		print("-----------------------------------------\n")
	#end if
	
	oDir = createFeatureZScore(nName, nPath, sPath, oRoot, numFolds, verbosity)
	
	if verbosity :
		print("Features stored at: {}".format(oDir))
	
	return oDir
# end def #################################


if __name__ == "__main__":
	print("\nRunning GeneSet MAPR meta-path feature calculation ...")

	# getting command line arguments
	params = readCommandLineFlags()
	
	_ = main(params)
	
	print("\nDone.\n")
#end if
