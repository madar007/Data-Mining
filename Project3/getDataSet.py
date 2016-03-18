import os
from pprint import pprint

'''
Questions:
1. Replace any non alphanumeric characters with space - ex: I'm split into I m
2. How to do bags of words and n-gram representation
'''

#	This function will only get called with files that is not a reply
#	It also removes Subject tokens
def extractText( file ):
	text = ''
	with open( file, 'r' ) as f:
		line = f.readline()
		while( line != None and line != ''):
			lineSplit = line.split( " " )
			if( lineSplit[ 0 ] == 'Subject:' ):
				lineSplit = lineSplit[1:]
				text += " ".join( lineSplit )
				line = f.readline()
			elif( lineSplit[ 0 ] == "Lines:" ):
				try: 
					numberOfLines = int( lineSplit[ 1 ] ) 
				except ValueError:
					print 'ValueError in file: ' + file
					return ''
				for i in range( numberOfLines + 1 ):
					text += f.readline()
				line = None
			else:
				line = f.readline()
	f.close()
	return text

#	eliminates non ascii characters
def eliminateNonAsciiChars( originalText ):
	text = ''
	for ch in originalText:
		if( ord(ch) < 128 ):
			text += ch
	return text

#	replace non alphanumeric characters with space
def replaceNonAlphaCharWithSpace( originalText ):
	text = ''
	for ch in originalText:
		if ch.isalpha():
			text += ch
		else:
			text += ' '
	return text
	
#	split text by spaces and elmininating empty spaces
def splitIntoTokens( text ):
	tokens = []
	text = text.split(' ')
	for x in text:
		if( x == '' ):
			continue
		if( x.isdigit() ):
			continue
		tokens += [ x ]
	return tokens

#	write to bags.csv
def writeToFile( frequencyTokenInDocumentMap ):
	print "Writing bags.csv..."
	f = open( 'bags.csv', 'w' )
	for key in sorted( frequencyTokenInDocumentMap.iterkeys() ):
		f.write( str( key[ 0 ] ) + ',' + str( key[ 1 ] ) + ',' + str( frequencyTokenInDocumentMap[ key ] ) + '\n' )
	f.close()

#	This will write .rlabel and .class files 		
def writeNewsGroupLabelToFile( idLabelMap ):
	print "Writing newsgroups.rlabel and newsgroups.class..."
	f = open( 'newsgroups.rlabel', 'w' )
	c = open( 'newsgroups.class', 'w' )
	for key in sorted( idLabelMap.iterkeys() ):
		fileName = idLabelMap[ key ]
		fileName = fileName.replace( '__', '_' )
		#	write fileName[ 2: ] because first 2 chars is "._"
		f.write( str( key ) + ',' + str( fileName[ 2: ].split( '_', 2 )[-1] ) + '\n' )
		group = fileName.split( '_' )
		c.write( str( key ) + ',' + str( group[ 3 ] ) + '\n' )
	c.close()
	f.close() 
	
#	write to bag.clabel
def writeBagLabelToFile( tokenIdMap ):
	print "Writing bag.clabel..."
	idTokenMap = dict( ( y, x ) for x, y in tokenIdMap.iteritems() )
	f = open( 'bag.clabel', 'w' )
	for key in sorted( idTokenMap.keys() ):
		f.write( str( idTokenMap[ key ] ) + '\n' )
	f.close()
	
#	write ngrams.csv 
def writeNGramToFile( nGramTokenInDocumentMap, n ):
	fileName = 'char' + str( n ) + '.csv'
	print "Writing " + fileName + "..."
	f = open( fileName, 'w' )
	for key in sorted( nGramTokenInDocumentMap.iterkeys() ):
		f.write( str( key[ 0 ] ) + ',' + str( key[ 1 ] ) + ',' + str( nGramTokenInDocumentMap[ key ] ) + '\n' )
	f.close()	
#	write charN.label
def writeNGramLabelToFile( nGramIdMap, n ):
	idNGramMap = dict( ( y, x ) for x, y in nGramIdMap.iteritems() )
	fileName = 'char' + str( n ) + '.clabel'
	print "Writing " + fileName + "..."
	f = open( fileName, 'w' )
	for key in sorted( idNGramMap.keys() ):
		f.write( str( idNGramMap[ key ] ) + '\n' )
	f.close()
		

def getNGrams( text, n ):
	tokens = []
	while( len( text ) >= n ):
		tokens += [ text[:n] ]
		text = text[1:]
	return tokens
	
def computeNGrams( nGramsToken, dimensions, objectNum, nGramIdMap, nGramTokenInDocumentMap ):
	tokens = nGramsToken
	for token in tokens:
		tokenId = nGramIdMap.get( token )
		if( tokenId == None ):
			nGramIdMap[ token ] = dimensions
			dimensions += 1
			nGramTokenInDocumentMap[ objectNum, dimensions-1 ] = 1
		else:
			nGramTokenInDocumentMap[ objectNum, tokenId ] = nGramTokenInDocumentMap.get( ( objectNum, tokenId ), 0 ) + 1
		#if token not in nGramIdMap.keys():
		#	nGramIdMap[ token ] = dimensions
		#	dimensions += 1
		#	nGramTokenInDocumentMap[ objectNum, dimensions-1 ] = 1 
		#else:
		#	tokenId = nGramIdMap[ token ]
		#	nGramTokenInDocumentMap[ objectNum, tokenId ] = nGramTokenInDocumentMap.get( ( objectNum, tokenId ), 0 ) + 1
			#if (objectNum, tokenId) in nGramTokenInDocumentMap.keys():
			#	frequency = nGramTokenInDocumentMap[ objectNum, tokenId ]
			#	frequency += 1
			#	nGramTokenInDocumentMap[ objectNum, tokenId ] = frequency
			#else:
			#	nGramTokenInDocumentMap[ objectNum, tokenId ] = 1 

	return nGramTokenInDocumentMap, dimensions, nGramIdMap
		
def cleanTextForNGram( text ):
	newText = text.replace( "   ", " " )
	newText = newText.replace( "  ", " " )
	return newText
				
frequencyTokenInDocumentMap = {}
threeGramTokenInDocumentMap = {}
fiveGramTokenInDocumentMap = {}
sevenGramTokenInDocumentMap = {}
tokenIdMap = {}
threeGramIdMap = {}
fiveGramIdMap = {}
sevenGramIdMap = {}
objectIdMap = {}
idLabelMap = {}
objectNum = 0
dimensions = 0
threeGramsDimensions = 0
fiveGramsDimensions = 0
sevenGramsDimensions = 0

for subdir, dirs, files in os.walk('./20_newsgroups/'):
	for fileName in files:
		file = subdir + '/' + fileName
		print file
		try:
			f = open( file, 'r' )
		except IOError:
			continue
		objectNum += 1
		hasSubject = False
		for line in f:
			text = ''
			lineSplit = line.split(" ")
			if( lineSplit[0].upper() != 'SUBJECT:' ):
				continue
			if( lineSplit[1].lower() == 're:' ):
				continue
			hasSubject = True
			#	extract text from Subject: and Line:
			text = extractText( file )
			#	eliminate non-ascii characters from text
			text = eliminateNonAsciiChars( text )
			#	change the character case to lower-case
			text.lower()
			#	replace any non alphanumeric characters with space
			text = replaceNonAlphaCharWithSpace( text )
			#	split text into tokens, removing empty strings and digits
			tokens = splitIntoTokens( text )
			#	clean text for n grams
			nGramText = cleanTextForNGram( text )
			#	call3Grams
			threeGramsToken = getNGrams( nGramText, 3 );
			threeGramTokenInDocumentMap, threeGramsDimensions, threeGramIdMap = computeNGrams( threeGramsToken, threeGramsDimensions, objectNum, threeGramIdMap, threeGramTokenInDocumentMap )
			#	call5Grams
			fiveGramsToken = getNGrams( nGramText, 5 );
			fiveGramTokenInDocumentMap, fiveGramsDimensions, fiveGramIdMap = computeNGrams( fiveGramsToken, fiveGramsDimensions, objectNum, fiveGramIdMap, fiveGramTokenInDocumentMap )	
			#	call7Grams
			sevenGramsToken = getNGrams( nGramText, 7 );
			sevenGramTokenInDocumentMap, sevenGramsDimensions, sevenGramIdMap = computeNGrams( sevenGramsToken, sevenGramsDimensions, objectNum, sevenGramIdMap, sevenGramTokenInDocumentMap )		
			#	set dimensions for each token
			for token in tokens:
				tokenId = tokenIdMap.get( token )
				if( tokenId == None ):
					tokenIdMap[ token ] = dimensions
					dimensions += 1
					frequencyTokenInDocumentMap[ objectNum, dimensions-1 ] = 1 
				else:
					frequencyTokenInDocumentMap[ objectNum, tokenId ] = frequencyTokenInDocumentMap.get( ( objectNum, tokenId ), 0 ) + 1
			
			idLabelMap[ objectNum ] = file.replace( "/", "_" )
			#	this break is because other functions have already looked over all lines
			break
		if( not hasSubject ):
			objectNum -= 1
		f.close()


writeNewsGroupLabelToFile( idLabelMap )
writeBagLabelToFile( tokenIdMap )
writeNGramLabelToFile( threeGramIdMap, 3 )
writeNGramLabelToFile( fiveGramIdMap, 5 )
writeNGramLabelToFile( sevenGramIdMap, 7 )
writeNGramToFile( threeGramTokenInDocumentMap, 3 )
writeNGramToFile( fiveGramTokenInDocumentMap, 5 )
writeNGramToFile( sevenGramTokenInDocumentMap, 7 )
writeToFile( frequencyTokenInDocumentMap )


