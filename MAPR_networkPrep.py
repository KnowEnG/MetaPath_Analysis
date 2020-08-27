"""
GeneSet MAPR implementation
	Step 01: pre-process the network
	
Convert the network edge and node files into
  meta-path matrices. Convert indirect
  connections into direct edges.

author: Greg Linkowski
	for KnowEnG by UIUC & NIH
"""

import argparse
import sys
import os
import re
import math
import gzip
import numpy as np
from shutil import copyfile



################################################################
# GLOBAL PARAMETERS

# Data-type for the path matrices:
MATRIX_DTYPE = np.float32
# Max value when normalizing a meta-path matrix
MATRIX_NORM = 1.0
# Length to pad the matrix file names:
FNAME_ZPAD = 6
# Considering consecutive edges of same type
KEEP_DOUBLE = True
KEEP_TRIPLE = True
# File extension to use when saving the matrix
MATRIX_EXTENSION = '.gz'	# '.txt' or '.gz' (gz is compressed)
# Normalize matrix to this value & save as '%u' (if true)
MX_SAVE_AS_INT = True
MX_NORM_AS_INT = 6.5e4

# end params ##############################



################################################################
# ANCILLARY FUNCTIONS

def readCommandLineFlags() :

	parser = argparse.ArgumentParser()
	
	parser.add_argument('netEdgeFile', type=str,
	                    help='path & file name of network edge file')
	parser.add_argument('-k', '--keep', type=str, default='',
	                    help='path & file name of network keep file')
	parser.add_argument('-l', '--length', type=int, default=3,
	                    help='maximum meta-path depth')
	parser.add_argument('-v', '--verbose', type=int, default=0,
	                    help='enable verbose output to terminal: 0=none, 2=all')
	parser.add_argument('-n', '--networkPath', type=str, default='./networks',
	                    help='output directory to store processed network')
	parser.add_argument('-t', '--textSubNets', type=bool, default=False,
	                    help='whether to save separate subnetwork text files')
	
	flags = parser.parse_args()
	
	return flags
#end def #################################


def verifyFile(fName, verbose) :

	# Verify file exists
	exists = True
	if not os.path.isfile(fName) :
		if not verbose:
			exists = False
		else :
			print ( "ERROR: Specified file doesn't exist:" +
				" {}".format(fName))
			sys.exit()
	#end if

	return exists
#end def #################################


def stripQuotesFromString(inString) :
	
	outString = inString
	
	if len(outString) > 2 :
		# strip single quotes
		if (outString[0] == "'") and (outString[-1] == "'") :
			outString = outString[1:-1]
		# strip double quotes
		elif (outString[0] == '"') and (outString[-1] == '"') :
			outString = outString[1:-1]
		#end if
	#end if
	
	return outString
#end def #################################


def readKeepFile(fName) :

	# Verify keep file exists
	if not os.path.isfile(fName) :
		print("ERROR: Failed to read keep file: ", fName)
		sys.exit()
	#end if
	
	# Flags to indicate which sections were seen:
	sawSectionGenes = False
	sawSectionEdges = False

	# The lists to return
	humanGenes = list()
	keepGenes = list()
	loseGenes = list()
	keepEdges = list()
	indirEdges = list()
	tHold = 0.0
	cutoffRanges = dict()

	# Read the file         #TODO: python version check 2 vs 3
#	f = open(fname, "rb")
	f = open(fName, "r")

	# read file line by line
	section = 'header'
	for line in f :
#		line = str(line)    #TODO: ever necessary for python 2 ??
		line = line.rstrip()
		if line == '':
			continue
		#end if

		# split the line by columns
		lv = line.split('\t')

		# Sections headers (defined by all-caps)
		#	set the behavior for the lines that follow
		if lv[0] == 'GENE TYPES' :
			section = 'gene'
			sawSectionGenes = True
		elif lv[0] == 'EDGE TYPES' :
			section = 'edge'
			sawSectionEdges = True
		elif lv[0] == 'THRESHOLD' :
			section = 'threshold'
			tHold = float(lv[2])
		elif lv[0] == 'CUTOFF RANGE' :
			section = 'cutoff'
			
		elif section == 'gene' :
			# sort genes between kept & ignored
			if (lv[2] == 'keep') or (lv[2] == 'yes') :
				keepGenes.append(lv[1])
				# if lv[1] == '*' :
				# 	keepGenes.append('[ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789]')
				# else :
				# 	keepGenes.append(lv[1])
				if lv[0].startswith('human') :
					humanGenes.append(lv[1])
					# if lv[1] == '*' :
					# 	humanGenes.append('[ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789]')
					# else :
					# 	humanGenes.append(lv[1])
			else :
				loseGenes.append(lv[1])
			#end if
			
		elif section == 'edge' :
			# sort kept edges & note indirect edges
			if (lv[2] == 'keep') or (lv[2] == 'yes') :
				keepEdges.append(lv[0])
				if (lv[1] != 'direct') and (lv[1] != 'yes') :
					indirEdges.append(lv[0])
			
		elif section == 'cutoff' :
			cutoffRanges[lv[0]] = (int(lv[1]), int(lv[2]))
		#end if
	#end loop

	# Error checks
	if not sawSectionGenes :
		print("ERROR: Keep file is missing section GENE TYPES:" +
		      "    {}".format(fName))
		sys.exit()
	#end if
	if not sawSectionEdges :
		print("ERROR: Keep file is missing section EDGE TYPES:" +
		      "    {}".format(fName))
		sys.exit()
	#end if
	for ie in indirEdges :
		if ie not in cutoffRanges.keys() :
			cutoffRanges[ie] = (0, 1e15)
			print("WARNING: Cutoff ranges unspecified for edge type {}".format(ie))
			print("    Using default inclusive range 0 to 10^15")
	#end for
	
	return humanGenes, keepGenes, loseGenes, keepEdges, indirEdges, cutoffRanges, tHold
#end def #################################


def readEdgeFile(fName) :

	# get the number of lines in the file
	nLines = sum( 1 for line in open(fName, "r"))

	# assign space for edge list
	Edges = np.empty( (nLines,4), dtype=np.dtype('object'))

	# dictionary to hold Node indices
	Nodes = dict()
	nodeSet = set()

	# Start reading from the file
	# df = open(datafile, "rb") #TODO: python version 2 or 3 ?
	df = open(fName, "r")

	i = 0
	for line in df:
		# extract the data from the file
#		line = line.decode('UTF-8') #TODO: left over from python 2 ??

		line = line.rstrip()
		lv = line.split('\t')

		# insert into the edge list
		Edges[i,0] = lv[0]
#		Edges[i,0] = str(lv[0], encoding='utf-8')
#		print(np.char.decode(Edges[i,0], 'ascii'))  #TODO: left over from python 2 ??
		Edges[i,1] = lv[1]
		Edges[i,2] = lv[2]
		Edges[i,3] = lv[3]

		# add node locations to dict
		if lv[0] in nodeSet :
			Nodes[lv[0]].append(i)
		else :
			Nodes[lv[0]] = list()
			Nodes[lv[0]].append(i)
			nodeSet.add(lv[0])
		#end if
		if lv[1] in nodeSet :
			Nodes[lv[1]].append(i)
		else :
			Nodes[lv[1]] = list()
			Nodes[lv[1]].append(i)
			nodeSet.add(lv[1])
		#end if

		i += 1
	#end loop

	# close the data file
	df.close()

#TODO: what should be the behavior if using Python 2 ?
	# print(Edges[0,:]
	# # Decode the edge array from type=bytes to type=str
	# if sys.version_info[0] < 3 :
	# 	Edges = np.char.decode(Edges, 'ascii')

	return Edges, Nodes
#end def #################################


def applyTermCutoffRanges(edgeList, indirEdges, nodeDict, termCutoffs) :

	# the terms to be removed from the network
	dropTerms = set()

	# the terms to be removed from the network
	keepTerms = set()

	# the indices of the rows to keep from edgeList
	keepIdx = list()

	# step through the edge list
	i = -1
	for row in edgeList :
		i += 1
		# if an indirect edge, check the size of term
		if row[3] in indirEdges :
			termSize = len(nodeDict[row[0]])
			termMinMax = termCutoffs[row[3]]
			if termSize > termMinMax[1] :
				dropTerms.add( (row[0], row[3], termSize) )
			elif termSize < termMinMax[0] :
				dropTerms.add( (row[0], row[3], termSize) )
			else :
				keepIdx.append(i)
				keepTerms.add( (row[0], row[3], termSize) )
		# keep all direct edges
		else :
			keepIdx.append(i)
	#end loop

	newEdgeList = edgeList[keepIdx,:]
	dropTerms = list(dropTerms)
	dropTerms.sort()
	keepTerms = list(keepTerms)
	keepTerms.sort()

	return newEdgeList, dropTerms, keepTerms
#end def #################################


def applyKeepLists(edges, lGenes, kEdges, iEdges) :

	#TODO: remove or pass skipped edges

	keepIndex = list()
	kEdgeSet = set(kEdges)

	# Note which rows in edge list will be kept
	#       and skip/discard non-kept edges
	for i in range(0, edges.shape[0]) :
		if edges[i,3] not in kEdgeSet :
			continue
		#end if

		# list of matches to be found
		m0 = list()
		m1 = list()
		# Check nodes for matches (column 1 & 2)
		for gt in lGenes :
			m0.append( re.match(gt, edges[i,0]) )
			m1.append( re.match(gt, edges[i,1]) )
		#end loop
		
		# Skip/discard genes that match the non-keep list
		# Check for any match with the non-keep list
		if any(match is not None for match in m1) :
			continue
		#ASSUMPTION: for indirect edges, col 0 contains
		#	a non-gene node
		elif edges[i,3] not in iEdges :
			if any(match is not None for match in m0) :
				continue
		#end if

		# Finally, if no objections
		#	add this to the kept list
		keepIndex.append(i)
	#end loop

	newEdges = edges[keepIndex,:]
	return newEdges
#end def #################################


def createNodeLists(edges, aGenes) :

	nodeDict = dict()
	nodeSet = set()
	geneSet = set()

	for i in range(0, edges.shape[0]) :
		# Add the first node to the dictionary,
		#	using a set for look-up speed
		if edges[i,0] in nodeSet :
			nodeDict[edges[i,0]].append(i)
		else :
			nodeDict[edges[i,0]] = list()
			nodeDict[edges[i,0]].append(i)
			nodeSet.add(edges[i,0])
		#end if

		# Add the second node to the dictionary,
		#	using a set for look-up speed
		if edges[i,1] in nodeSet :
			nodeDict[edges[i,1]].append(i)
		else :
			nodeDict[edges[i,1]] = list()
			nodeDict[edges[i,1]].append(i)
			nodeSet.add(edges[i,1])
		#end if

		# list of matches to be found
		m0 = list()
		m1 = list()
		# Check nodes for matches (column 1 & 2)
		for gt in aGenes :
			m0.append( re.match(gt, edges[i,0]) )
			m1.append( re.match(gt, edges[i,1]) )
		#end loop
		# Matches mean node is a gene; add to set
		if any(match is not None for match in m0) :
			geneSet.add(edges[i,0])
		if any(match is not None for match in m1) :
			geneSet.add(edges[i,1])
		#end if
	#end loop

	geneList = list(geneSet)
	geneList.sort()
	return nodeDict, geneList
#end def #################################


def writeModEdgeFilePlus(oPath, oName, nDict, gList, eArray) :

	newPath = oPath + oName + '/'
	if not os.path.exists(newPath) :
		os.makedirs(newPath)
	#end if

	# Save output: ordered list of genes
	# Save output: row indices for each node in edge file
	gFile = 'genes.txt'
	nFile = 'indices.txt'
	gf = open(newPath + gFile, 'w')
	nf = open(newPath + nFile, 'w')
#	gf = open(newPath + gFile, 'wb')    #TODO: left over from python 2 ??
#	nf = open(newPath + nFile, 'wb')
	firstLine = True
	for gene in gList :
		if firstLine :
			firstLine = False
		else :
			gf.write("\n")
			nf.write("\n")
		#end if

		gf.write("{}".format(gene))

		nf.write("{}\t".format(gene, nDict[gene]))

		firstIndex = True
		for item in nDict[gene] :
			if firstIndex :
				firstIndex = False
			else :
				nf.write(",")
			#end if
			nf.write("{}".format(item))
		#end loop

	#end loop
	gf.close()
	nf.close()

	# Save output: list of edge types
	eTypes = np.unique(eArray[:,3])
	eTypes.sort()

	eFile = 'edges.txt'
	with open(newPath + eFile, 'w') as ef :
		firstLine = True
		for et in eTypes :
			if firstLine :
				firstLine = False
			else :
				ef.write("\n")
			#end if
	
			ef.write("{}".format(et))
		#end loop
	#end with

	# Save output: the network (as an edge list)
	oFile = 'network.txt'
	with open(newPath + oFile, 'w') as of :
		firstLine = True
		for i in range(0, eArray.shape[0]) :
			if firstLine :
				firstLine = False
			else :
				of.write("\n")
			#end if
	
			of.write("{}\t{}\t{}\t{}".format(eArray[i,0],
				eArray[i,1], eArray[i,2], eArray[i,3]))
		#end loop
	#end with

	return
#end def #################################

def saveSelectGeneDegrees(oPath, oName, edgeArray, genesAll, humanRegex) :
	
	textDelim = '\t'

	# If folder doesn't exist, create it
	if not os.path.exists(oPath + oName + "/") :
		os.makedirs(oPath + oName + "/")
	#end if

	# NOTE: Only considering human genes (at least for now)
	# Build an index dictionary from the human genes
	genesAll = np.unique(genesAll)
	genesAll.sort()
	gHumanDict = dict()
	index = 0
	for gene in genesAll :
		# Look for matches to regex expression
		ma = list()
		for exp in humanRegex :
			ma.append( re.match(exp, gene) )
		# add gene only if one of the matches is positive
		if any(match is not None for match in ma) :
			gHumanDict[gene] = index
			index += 1
	#end loop

	# Get list of edge types
	eTypes = np.unique(edgeArray[:,3])
	eTypes.sort()
	# Build an index dictionary from the edge types
	eDict = dict()
	index = 1 		# col 0 reserved for 'all'
	for et in eTypes :
		eDict[et] = index
		index += 1
	#end loop

	# matrix to store degree counts (col 0 reserved for 'all')
	degreeMatrix = np.zeros([ len(gHumanDict), len(eTypes)+1 ])

	# First, count degrees along SPECIFIED edge types
	for row in edgeArray :
		# by incrementing the matrix
		if row[0] in gHumanDict :
			degreeMatrix[ gHumanDict[row[0]], eDict[row[3]] ] += 1
		if row[1] in gHumanDict :
			degreeMatrix[ gHumanDict[row[1]], eDict[row[3]] ] += 1
	#end loop

	# Second, sum degrees along ALL edge types
	degreeMatrix[:, 0] = np.sum(degreeMatrix, axis=1)

	# Open the output file
	fname = oPath + oName + '/node-degree.txt'
	
	with open(fname, 'w') as fOut :     #TODO 'wb' for python 2 ??
	
		# Write column headers
		fOut.write("HEADER{}all".format(textDelim))
		for et in eTypes :
			fOut.write("{}{}".format(textDelim, et))
	
		# Write the matrix to file
		gHumanList = list(gHumanDict.keys())
		gHumanList.sort()
		for i in range(len(gHumanList)) :
			fOut.write( "\n{}".format(gHumanList[i]) )
	
			for j in range(degreeMatrix.shape[1]) :
				fOut.write( "{}{:.0f}".format(textDelim, degreeMatrix[i,j]) )
		#end loop
	#end with

	return
#end def #################################


def readFileAsIndexDict(fName) :

	verifyFile(fName, True)

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
#end def #################################


def convertMatrixFloatToUInt16(matrix, verbosity) :
	
	maxVal = np.amax(matrix)
	minVal = np.amin(matrix)
	if maxVal == minVal:
		if verbosity > 1:
			print(
				"WARNING: Uniform {:,}x{:,} matrix , all values = {}".format(matrix.shape[0], matrix.shape[1], maxVal))
	# end if
	
	if minVal != 0 :
		matrix = np.subtract(matrix, minVal)
		maxVal = np.amax(matrix)
	if maxVal > MX_NORM_AS_INT :
		matrix = np.divide(matrix, (maxVal - minVal))
		matrix = np.multiply(matrix, MX_NORM_AS_INT)
	#end if
	
	return matrix.astype(np.uint16)
#end def #################################


def buildDirectPathMatrix(eList, gDict, verbosity) :

	matrix = np.zeros( (len(gDict), len(gDict)), dtype=np.float32 )
	for row in eList :
		i = gDict[row[0]]
		j = gDict[row[1]]
		matrix[i,j] += float(row[2])
		matrix[j,i] += float(row[2])
	#end for
	
	return convertMatrixFloatToUInt16(matrix, verbosity)
#end def #################################


def saveSubNetworkDirect(snPath, snName, eList) :
	
	snPath = snPath.rstrip('/') + '/'
	fnSubNet = snPath + snName
	
	with open(fnSubNet, 'w') as fOut:
		firstLine = True
		
		for row in eList:
			if firstLine :
				firstLine = False
			else :
				fOut.write('\n')
			fOut.write('{}\t{}\t{}'.format(row[0], row[1], row[2]))
	#end with
	
	return
# end def #################################


def buildIndirectPathMatrix(eList, nNames, gDict, nDict, verbosity) :

	matrix = np.zeros( (len(gDict), len(gDict)), dtype=np.float32 )

	for node in nNames :
		idxList = nDict[node]
		listLen = len(idxList)
		
		for iIdx in range(listLen - 1):
			thisRowIdx = idxList[iIdx]
			thisGene = eList[thisRowIdx, 1]
			thisWeight = float(eList[thisRowIdx, 2])
			
			for jIdx in range((iIdx + 1), listLen):
				nextRowIdx = idxList[jIdx]
				nextGene = eList[nextRowIdx, 1]
				nextWeight = float(eList[nextRowIdx, 2])
				
				i = gDict[thisGene]
				j = gDict[nextGene]
				placeWeight = thisWeight * nextWeight
				matrix[i, j] += placeWeight
				matrix[j, i] += placeWeight
	# end loop
	
	return convertMatrixFloatToUInt16(matrix, verbosity)
#end def #################################


def saveSubNetworkIndirect(snPath, snName, eList, nNames, nDict) :
	
	snPath = snPath.rstrip('/') + '/'
	fnSubNet = snPath + snName
	
	with open(fnSubNet, 'w') as fIn :
		firstLine = True
		
		for node in nNames :
			idxList = nDict[node]
			listLen = len(idxList)
			
			for iIdx in range(listLen - 1) :
				thisRowIdx = idxList[iIdx]
				thisGene = eList[thisRowIdx, 1]
				thisWeight = float(eList[thisRowIdx, 2])
				
				for jIdx in range((iIdx + 1), listLen):
					nextRowIdx = idxList[jIdx]
					nextGene = eList[nextRowIdx, 1]
					nextWeight = float(eList[nextRowIdx, 2])
	
					placeWeight = thisWeight * nextWeight
					
					if firstLine :
						firstLine = False
					else :
						fIn.write('\n')
					fIn.write('{}\t{}\t{}'.format(thisGene, nextGene, placeWeight))
	#end with
	
	return
# end def #################################


def buildGeneTermMatrix(eList, gDict, verbosity=0) :
	
	nList = np.unique(eList[:,0])
	nDict = dict()
	idx = -1
	for n in nList:
		idx += 1
		nDict[n] = idx
	#end loop
	
	matrix = np.zeros( (len(gDict), len(nList)), dtype=np.float32 )
	for row in eList :
		i = gDict[row[1]]
		j = nDict[row[0]]
		matrix[i,j] += float(row[2])
	#end loop

	return convertMatrixFloatToUInt16(matrix, verbosity), nList
#end def #################################


def saveKeyFile(mDict, path):
	"""
	save the key file for the metapath matrices
	Creates a legend, mapping the path type on the right
	to the path matrix file on the left, where 't'
	indicates the transpose of that matrix should be used

	:param mDict: dict,
		key, str: metapath names
		value, [int, : corresponding index number for mList
				bool] : True means use matrix transpose
	:param path: str, path to the folder to save the file
	:return:
	"""
	
	# If folder doesn't exist, create it
	if not os.path.exists(path):
		os.makedirs(path)
	# end if
	
	# Get the sorted list of all paths
	nameList = list(mDict.keys())
	nameList.sort()
	
	# This file tells which matrix corresponds to which path
	fKey = open(path + "key.txt", "w")
	#	fKey = open(path+"key.txt", "wb")   #TODO: left over from python 2
	fKey.write("NOTE: 't' means use matrix transpose\n")
	firstLine = True
	for name in nameList:
		if firstLine:
			firstLine = False
		else:
			fKey.write("\n")
		# end if
		fKey.write("{}".format(str(mDict[name][0]).zfill(FNAME_ZPAD)))
		if mDict[name][1]:
			fKey.write(",t")
		else:
			fKey.write(", ")
		fKey.write("\t{}".format(name))
	# end loop
	fKey.close()
	
	return


# end def #################################


def readKeyFilePP(path):
	"""
	Read in the key.txt file regarding the	metapath matrices

	:param path: str, path to the network files
	:return: keyDict, dict
		key, str: name of metapath
		value, tuple: int is matrix/file ID number
			bool where True means use matrix transpose
	"""
	
	if not path.endswith('_MetaPaths/'):
		if path.endswith('/'):
			path = path[0:-1] + '_MetaPaths/'
		else:
			path = path + '_MetaPaths/'
	# end if
	fName = path + "key.txt"
	# ERROR CHECK: verify file exists
	if not os.path.isfile(fName):
		print("ERROR: Specified file doesn't exist:" +
		      " {}".format(fName))
		sys.exit()
	# end if
	
	# The item to return
	keyDict = dict()
	
	# Read in the file
	fk = open(fName, "r")
	#	fk = open(fName, "rb")  #TODO: left over from python 2
	firstLine = True
	for line in fk:
		# skip the first line
		if firstLine:
			firstLine = False
			continue
		# end if
		
		# separate the values
		line = line.rstrip()
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


def getPathMatrixV2(mpFName, mpPath, sizeOf) :

	preName = mpPath + mpFName
	if os.path.isfile(preName) :
		fName = preName
	elif os.path.isfile(preName + '.gz') :
		fName = preName + '.gz'
	elif os.path.isfile(preName + '.txt') :
		fName = preName + '.txt'
	else :
		# ERROR CHECK: verify file exists
		print ( "ERROR: Specified file doesn't exist:" +
			" {}  w/.txt/.gz".format(preName))
		sys.exit()
	#end if

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

	return matrix
#end def #################################


def saveMatrixNumpyV2(matrix, mFileName, mPath, mName, asUint) :
	"""
	save given matrix as a .npy file
	
	:param matrix: (NxN) list, the values to save
	:param mFileName: str, name of the file to save
	:param mPath: str, path to the folder to save the file
	:param mName: str, name of corresponding meta-path
	:param asUint: bool, whether to convert matrix as uint16 b/f saving
	:return:
	"""
	
	# If folder doesn't exist, create it
	if not os.path.exists(mPath) :
		os.makedirs(mPath)
	#end if

	# Matrix prep before saving
	#    - Warn if matrix contains zero connections #TODO: how to safely skip a 0 matrix ??
	#    - Convert to uint16 if flag is true (to save space)
	#    - (otherwise) Normalize according to defaults
	maxVal = np.amax(np.amax(matrix))
	if maxVal == 0 :
		print("WARNING: maximum value == 0  for saved matrix {}".format(mFileName))
		print("    Meta-path {} does not connect any nodes.\n".format(mName))
	elif asUint :
		matrix = convertMatrixFloatToUInt16(matrix, 0)
	else :
		matrix = np.divide(matrix, maxVal)
		matrix = np.multiply(matrix, MATRIX_NORM)
	#end if
	
	if mFileName.endswith('.gz') or mFileName.endswith('.txt') :
		np.savetxt(mPath + mFileName, matrix, fmt='%u')
	else :
		np.savetxt(mPath + mFileName + MATRIX_EXTENSION, matrix, fmt='%u')
	#end if
	
	return
#end def #################################


def NChooseK(N, K):
	"""
	Calculate (N choose K)

	:param N: int, the size of the overall set
	:param K: int, size of the combinations chosen from set
	:return: combos, int, number of combinations possible
	"""
	
	numer = math.factorial(N)
	denom = math.factorial(K) * math.factorial(N - K)
	
	combos = numer / denom
	return int(combos)
# end def #################################

def createMPLengthOneV2(pDict, pDir, verbosity):
	# Get the list of the primary path matrices
	pNames = list(pDict.keys())
	pNames.sort()
	
	# Check if metapath key file exists
	if verifyFile('{}key.txt'.format(pDir), True):
		# check if all expected 1-step paths were created
		#	exit if they are
		mNumExpected = len(pNames)
		mDict = readKeyFilePP(pDir)
		mp1Count = 0
		for mpName in list(mDict.keys()):
			if mpName.count('-') == 0:
				mp1Count += 1
		# end loop
		if mp1Count == mNumExpected:
			if verbosity:
				print("All 1-length paths already computed ...")
			return
		elif mp1Count >= mNumExpected:
			print("ERROR: Uh-oh, more 1-length paths than expected already exist!")
			sys.exit()
	# end if
	
	if verbosity > 1:
		print("    Creating key.txt for length-1 meta-paths")
	
	# Create the metapath dict
	mpDict = dict()
	for name in pNames:
		mpDict[name] = (pDict[name], False)
	# end loop
	
	# save the new mp key file
	saveKeyFile(mpDict, pDir)
	
	return
# end def #################################


def createMPLengthTwoV2(pDict, pDir, verbosity):
	# Get the list of the primary path matrices
	pNames = list(pDict.keys())
	pNames.sort()
	pNum = len(pNames)
	
	# Get the list of all metapaths created thus far
	mDict = readKeyFilePP(pDir)
	mNum = pNum
	
	# Check if all expected 2-step paths were created
	#	exit if they are
	mNumExpected = math.pow(pNum, 2)
	mp2Count = 0
	for mpName in list(mDict.keys()):
		# print(mpName)
		if mpName.count('-') == 1:
			mp2Count += 1
	# end if
	if mp2Count == int(mNumExpected):
		if verbosity:
			print("All 2-length paths already computed ...")
		return
	elif mp2Count >= mNumExpected:
		print("ERROR: Uh-oh, more 2-length paths than expected already exist!")
		sys.exit()
	# end if
	
	# Get the matrix dimensions from genes.txt file
	sizeOf = 0
	pDirAlt = pDir
	if pDirAlt.endswith("_MetaPaths/"):
		pDirAlt = pDirAlt[:-11] + '/'
	with open(pDirAlt + 'genes.txt') as fin:
		for line in fin:
			sizeOf += 1
	# end with
	
	# Multiply each matrix pair
	mNameSet = set()
	for p1 in pNames:
		
		# Load the first matrix
		mFileA = str(pDict[p1]).zfill(FNAME_ZPAD) + MATRIX_EXTENSION
		matrixA = getPathMatrixV2(mFileA, pDir, sizeOf)
		
		sizeGB = matrixA.nbytes / 1.0e9
		if verbosity > 1:
			print("-- loaded {}, size: {:.3f} GBytes".format(p1, sizeGB))
		
		for p2 in pNames:
			# Optionally skipping consecutive edges
			if not KEEP_DOUBLE:
				if p1 == p2:
					continue
			# end if
			
			# The name of this path
			name = p1 + '-' + p2
			# The name of the reversed path
			nameRev = p2 + '-' + p1
			
			# Skip if this mp already calculated
			if name in mNameSet:
				if verbosity > 1:
					print("-- {} exists, skipping".format(name))
				continue
			# end if
			
			# Create new matrix if file doesn't already exist
			newMFName = str(mNum).zfill(FNAME_ZPAD) + MATRIX_EXTENSION
			
			if not os.path.isfile(pDir + newMFName):
				
				# Load the second matrix
				if verbosity > 1:
					print("-- -- Loading matrix {}".format(p2))
				mFileB = str(pDict[p2]).zfill(FNAME_ZPAD) + MATRIX_EXTENSION
				matrixB = getPathMatrixV2(mFileB, pDir, sizeOf)
				
				# eliminate loops by removing self-connections
				np.fill_diagonal(matrixA, 0)
				np.fill_diagonal(matrixB, 0)
				# NOTE: Final saved matrix will contain self-connections
				
				if verbosity > 1:
					print("-- -- creating matrix {}".format(name))
				
				# Multiply the two matrices
				newM = np.dot(matrixA, matrixB)
				
				if verbosity > 1:
					print("-- -- -- saving matrix {}".format(newMFName))
				
				saveMatrixNumpyV2(newM, newMFName, pDir, name, MX_SAVE_AS_INT)
			# end if
			
			# Add the matrix name & number to mDict
			if name == nameRev:  # (ie: typeA-typeA)
				# Then add just this matrix to the list
				mDict[name] = [mNum, False]
				mNameSet.add(name)
			else:
				# Add this path & note the reverse path
				mDict[name] = [mNum, False]
				mNameSet.add(name)
				#	Reverse path uses transpose
				mDict[nameRev] = [mNum, True]
				mNameSet.add(nameRev)
			# end if
			mNum += 1
	# end loop
	
	saveKeyFile(mDict, pDir)
	return
# end def #################################


def createMPLengthThreeV2(pDict, pDir, verbosity):
	# Get the list of the primary path matrices
	pNames = list(pDict.keys())
	pNames.sort()
	pNum = len(pNames)
	
	# Get the list of all metapaths created thus far
	mDict = readKeyFilePP(pDir)
	
	# Set the number from which to start naming matrices
	mNum = NChooseK(pNum, 2) + (2 * pNum)
	
	# Check if all expected 3-step paths were created
	#	exit if they are
	mNumExpected = math.pow(pNum, 3)
	mp3Count = 0
	for mpName in list(mDict.keys()):
		if mpName.count('-') == 2:
			mp3Count += 1
	# end loop
	if mp3Count == mNumExpected:
		if verbosity:
			print("All 3-length paths already computed ...")
		return
	elif mp3Count >= mNumExpected:
		print("ERROR: Uh-oh, more 3-length paths than expected already exist!")
		sys.exit()
	# end if
	
	# Get the matrix dimensions from genes.txt file
	sizeOf = 0
	pDirAlt = pDir
	if pDirAlt.endswith("_MetaPaths/"):
		pDirAlt = pDirAlt[:-11] + '/'
	with open(pDirAlt + 'genes.txt') as fin:
		for line in fin:
			sizeOf += 1
	# end with
	
	# Build list of 2-step path names
	mp2Names = list()
	for mpName in list(mDict.keys()):
		if mpName.count('-') == 1:
			mp2Names.append(mpName)
	# end loop
	mp2Names.sort()
	
	# Multiply each matrix pair
	pNameSet = set()
	for p1 in pNames:
		
		# Load the first matrix
		mFileA = str(pDict[p1]).zfill(FNAME_ZPAD) + MATRIX_EXTENSION
		matrixA = getPathMatrixV2(mFileA, pDir, sizeOf)
		
		sizeGB = matrixA.nbytes / 1.0e9
		if verbosity > 1:
			print("-- loaded {}, size: {:.3f} GBytes".format(p1, sizeGB))
		
		for p2 in mp2Names:
			# The name of this path
			name = p1 + '-' + p2
			# The name of the reversed path
			p2v = p2.split('-')
			nameRev = p2v[1] + '-' + p2v[0] + '-' + p1
			
			# Optionally skipping consecutive edges
			if not KEEP_DOUBLE:
				if p1 == p2:
					continue
			# end if
			
			if name in pNameSet:
				continue
			# end if
			
			# Create new matrix if file doesn't already exist
			newMFName = str(mNum).zfill(FNAME_ZPAD) + MATRIX_EXTENSION
			
			if not os.path.isfile(pDir + newMFName):
				# Save a placeholder (allows multiple threads, to skip this one)
				fakeMx = np.ones((2, 2))
				saveMatrixNumpyV2(fakeMx, newMFName, pDir, name, MX_SAVE_AS_INT)
				
				# Load the second matrix
				mFileB = str(mDict[p2][0]).zfill(FNAME_ZPAD) + MATRIX_EXTENSION
				matrixB = getPathMatrixV2(mFileB, pDir, sizeOf)
				
				# eliminate loops by removing self-connections
				np.fill_diagonal(matrixA, 0)
				np.fill_diagonal(matrixB, 0)
				# NOTE: Final saved matrix will contain self-connections
				
				# Multiply the two matrices
				if verbosity > 1:
					print("-- -- creating matrix {}".format(name))
				newM = np.dot(matrixA, matrixB)
				
				if verbosity > 1:
					print("-- -- -- saving matrix {}".format(newMFName))
				saveMatrixNumpyV2(newM, newMFName, pDir, name, MX_SAVE_AS_INT)
			# end if
			
			# end if
			
			# Add the matrix name & number to mDict
			if name == nameRev:  # (ie: typeA-typeA)
				# Then add just this matrix to the list
				mDict[name] = [mNum, False]
				pNameSet.add(name)
			else:
				# Add this path & note the reverse path
				mDict[name] = [mNum, False]
				pNameSet.add(name)
				#	Reverse path uses transpose
				mDict[nameRev] = [mNum, True]
				pNameSet.add(nameRev)
			# end if
			mNum += 1
	# end loop
	
	saveKeyFile(mDict, pDir)
	return
# end def #################################


# def createKeepFile(fnEdgeIn, fnKeepOut) :
#
# 	#TODO: this
# #end def #################################



################################################################
# PRIMARY FUNCTION(S)

def buildNetworkUsingKeep(fnEdge, fnKeep, netPath, outName, verbosity) :

	##########################################
	# read in the edge & keep files
	#   build numpy array of objects from edge list
	
	if verbosity > 0 :
		print("\nReading in the following edge & keep files ...")
		print("    {}".format(fnEdge))
		print("    {}".format(fnKeep))
	#end if
	
	geneHuman, keepGenes, loseGenes, keepEdges, indirEdges, termCuttoffs, thresh = readKeepFile(fnKeep)
	edgeArray, nodeDict = readEdgeFile(fnEdge)
	
	if verbosity > 1 :
		print("    edge file contained {:,d} lines".format(edgeArray.shape[0]))
	
	
	##########################################
	# apply keep file restrictions to edge list
	#   keep indirect nodes if size within cutoff range
	#   keep only specified nodes & edge types
	
	if verbosity > 0 :
		print("Applying restrictions specified in keep file ...")
	numEdgesOrig = edgeArray.shape[0]
	
	edgeArray, lTerms, kTerms = applyTermCutoffRanges(edgeArray, indirEdges, nodeDict, termCuttoffs)
	del termCuttoffs
	
	edgeArray = applyKeepLists(edgeArray, loseGenes, keepEdges, indirEdges)
	if verbosity > 1 :
		print("    removed {:,d} of {:,d} edges".format((numEdgesOrig - edgeArray.shape[0]), numEdgesOrig))
	

	##########################################
	# save the network and associated files to folder
	#   update nodeDict & geneList from new edge list
	#   create the new directory
	#   save files: edge list, gene list, non-gene term list,
	#       copy of keep file,
	#       node degree matrix (shows degree for each sub-network)

	if verbosity > 0 :
		print("Saving network files to {}{}/".format(netPath, outName))
	
	nodeDict, geneList = createNodeLists(edgeArray, keepGenes)
	
	writeModEdgeFilePlus(netPath, outName, nodeDict, geneList, edgeArray)
	copyfile(fnKeep, netPath + outName + '/keep.txt')
	saveSelectGeneDegrees(netPath, outName, edgeArray, geneList, geneHuman)
	
	return
#end def #################################


def createPrimaryMatrices(outPath, outName, doSaveSubNets, verbosity) :
	##########################################
	# define IO file names & paths
	#   check if output path already exists
	
	kFile = 'keep.txt'
	eFile = 'network.txt'
	gFile = 'genes.txt'
	tFile = 'terms.txt'             #TODO: Do we want this ?
	pKeyFile = 'key_primaries.txt'
	tKeyFile = 'key_termNonGene.txt'
	subNetPrefix = outName.rstrip('/')

	netPath = outPath.rstrip('/') + '/' + outName.rstrip('/') + '/'
	mpPath = outPath.rstrip('/') + '/' + outName.rstrip('/') + '_MetaPaths/'

	#TODO: if mpPath_exists, then validate each file before overwriting
	mpPath_exists = True
	if not os.path.exists(mpPath) :
		os.makedirs(mpPath)
		mpPath_exists = False
	#end if


	##########################################
	# read in lists from network files
	#   get: kept genes, kept edge types, indirect edge types
	#   get: edge list
	#   get: node-index dictionary, gene-index dictionary

	gH, keepGenes, lG, keepEdges, indirEdges, tC, thresh = readKeepFile(netPath + kFile)
	del gH, lG, tC

	edgeList, nD = readEdgeFile(netPath + eFile)
	del nD
	
	nodeDict, gL = createNodeLists(edgeList, keepGenes)
	del gL
	
	geneDict = readFileAsIndexDict(netPath + gFile)
	

	##########################################
	# create primary matrix files (one per edge type)
	#   assign number/name to the matrix
	#   extract edges from full edge list
	#   if indirect edge, create & save gene-gene matrix
	#       plus non-gene term list & gene-term matrix
	#   else just save gene-gene matrix
	
	#TODO: check if each of the output files exists before calculating them
	
	pmDict = dict()
	pmCount = -1
	for eType in keepEdges:
		pmCount += 1
		
		pfName = str(pmCount).zfill(FNAME_ZPAD)
		if verbosity > 1 :
			print("  creating matrix {}: {} ...".format(pfName, eType))
		pmDict[eType] = pfName
		
		keepIdx = [i for i in range(len(edgeList)) if edgeList[i, 3] == eType]
		edgeCut = edgeList[keepIdx, :]
		
		if eType in indirEdges:
			selectNodes = np.unique(edgeCut[:, 0])
			pMatrix = buildIndirectPathMatrix(edgeList, selectNodes, geneDict, nodeDict, verbosity)
			tMatrix, tList = buildGeneTermMatrix(edgeCut, geneDict, verbosity)
			
			np.savetxt(mpPath + pfName + '.gz', pMatrix, fmt='%u')
			np.savetxt(mpPath + pfName + 'tm.gz', tMatrix, fmt='%u')
			with open(mpPath + pfName + 'tl.txt', 'w') as ftlOut:
				ftlOut.write('{}'.format(tList[0]))
				for t in tList[1:]:
					ftlOut.write('\n{}'.format(t))
			
			if doSaveSubNets :
				subNetName = '{}-{}.txt'.format(subNetPrefix, eType)
				if verbosity > 0 :
					print("  saving subnetwork edge list: {}".format(subNetName))
				saveSubNetworkIndirect(netPath, subNetName, edgeList, selectNodes, nodeDict)
		else:
			pMatrix = buildDirectPathMatrix(edgeCut, geneDict, verbosity)
			
			np.savetxt(mpPath + pfName + '.gz', pMatrix, fmt='%u')
			
			if doSaveSubNets :
				subNetName = '{}-{}.txt'.format(subNetPrefix, eType)
				if verbosity > 0 :
					print("  saving subnetwork edge list: {}".format(subNetName))
				saveSubNetworkDirect(netPath, subNetName, edgeCut)
	#end for
	

	##########################################
	# save the key files
	#   indicate which matrix files correspond to which meta-paths
	#   2nd key file specific to indirect gene-term edges
	
	if verbosity > 1 :
		print("    Finished saving primary matrices, writing key files")

	# Create the metapath dict & save the mp key file
	if not verifyFile(mpPath + 'key.txt', False) :
		mpDict = dict()
		for eType in keepEdges :
			mpDict[eType] = ( pmDict[eType], False )
		#end loop
		saveKeyFile(mpDict, mpPath)
	#end if
	
	with open(mpPath + pKeyFile, 'w') as fOut :
		firstLine = True
		for eType in keepEdges :
			if firstLine :
				fOut.write('{}\t{}'.format(pmDict[eType], eType))
				firstLine = False
			else :
				fOut.write('\n{}\t{}'.format(pmDict[eType], eType))
	#end with
	
	with open(mpPath + tKeyFile, 'w') as fOut :
		if indirEdges :
			eType = indirEdges[0]
			fOut.write('{}\t{}'.format(pmDict[eType], eType))
			for eType in indirEdges[1:] :
				fOut.write('\n{}\t{}'.format(pmDict[eType], eType))
		else :
			fOut.write("    Note: No indirect edges exist in this network.")
	#end with

	return
#end def #################################


def createMetaPathMatrices(ePath, eName, maxMPDepth, verbosity) :

	##########################################
	# define IO file names & paths
	#   check if output path already exists

	kFile = 'keep.txt'
	gFile = 'genes.txt'
	mpKeyFile = 'key.txt'
	ppKeyFile = 'key_primaries.txt'

	eDir = ePath.rstrip('/') + '/' + eName.rstrip('/') + '/'
	mpDir = eDir.rstrip('/') + '_MetaPaths/'
	
	#TODO: check existence of folder, files ?

	# 1) Read in the primary path dict
	pathDict = dict()
	with open(mpDir + ppKeyFile) as fin :
		for line in fin :
			line = line.rstrip()
			lv = line.split('\t')
			pathDict[lv[1]] = int(lv[0])
	#end with

	# 2) Create meta-path matrices by desired length
	createMPLengthOneV2(pathDict, mpDir, verbosity)
	if maxMPDepth >= 2 :
		createMPLengthTwoV2(pathDict, mpDir, verbosity)
	if maxMPDepth >= 3 :
		createMPLengthThreeV2(pathDict, mpDir, verbosity)

	return
#end def #################################



################################################################
# MAIN FUNCTION & CALL
def main(params) :
	
	##########################################
	# set parameter values & IO file locations
	#   read in the command-line arguments
	#   convert to file names and paths as appropriate
	
	# Upper bound on meta-path length to calculate (hard-coded)
	maxMPLen_UB = 3
	
	# # Collect parameters from the command line
	# params = readCommandLineFlags()
	
	maxMPLen = min(params.length, maxMPLen_UB)
	fnEdge = params.netEdgeFile
	fnKeep = params.keep
	verbosity = params.verbose
	netPath = params.networkPath
	flagSaveSubNet = params.textSubNets
	
	# Assign a name to the network
	fnEVect = fnEdge.split('/')
	netName = fnEVect[-1]
	if netName.endswith('.edge.txt') :
		netName = netName[0:-9]
	
	# Verify edge file exists
	fnEdge = stripQuotesFromString(fnEdge)
	verifyFile(fnEdge, True)
	
	# Check for keep file
	fnKeep = stripQuotesFromString(fnKeep)
	if fnKeep :
		keepExists = verifyFile(fnKeep, False)
	else :
		for i in range(len(fnEVect) - 1) :
			fnKeep = fnKeep + fnEVect[i] + '/'
		fnKeep = fnKeep + netName + '.keep.txt'
		keepExists = verifyFile(fnKeep, False)
	#end if
	
	# PREV: If not specified, set network output dir to input dir
	# CURR: Default processed network to ./networks dir
	netPath = stripQuotesFromString(netPath)
	# if not netPath :          # NOTE: prev default val was ''
	# 	for i in range(len(fnEVect) - 1):
	# 		netPath = netPath + fnEVect[i] + '/'
	netPath = netPath.rstrip('/') + '/'

	##########################################
	# build the basic network files from provided edge list
	#   two methods: one uses keep file to specify network
	#       other assumes complete edge file
	
	#TODO: check if this step has been done, skip and notify if so
	#TODO: perhaps show a note about renaming edge file, choosing different netPath, or deleting current network
	if keepExists :
		buildNetworkUsingKeep(fnEdge, fnKeep, netPath, netName, verbosity)
	else :
		#TODO: need func to read edge file in absence of keep file
		# buildNetworkWithoutKeep(fnEdge, netPath, verbosity)
		print("This part coming soon ...")
	#end if
	

	##########################################
	# create primary adjacency matrices
	
	#TODO: check for existence of these files, as above
	createPrimaryMatrices(netPath, netName, flagSaveSubNet, verbosity)
	

	##########################################
	# create meta-path matrices

	#TODO: check for existence of these files, as above
	createMetaPathMatrices(netPath, netName, maxMPLen, verbosity)
	
	return netName
# end def #################################


if __name__ == "__main__" :
	print("\nRunning GeneSet MAPR pre-processing ...")

	# Collect parameters from the command line
	params = readCommandLineFlags()
	
	_ = main(params)
	
	print("\nDone.\n")
#end if