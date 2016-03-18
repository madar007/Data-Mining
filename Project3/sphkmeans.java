import java.util.Random;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Collections;
import java.util.Arrays;
import java.io.*;

/* NOTE: objectId starts with 1 but termId starts with 0. */

class DocumentVector{
	public int objectId;
	public int termId;
	public float frequency;
	DocumentVector( String line ){
		splitLine( line );
	}
	void splitLine( String line ){
		String[] columns = line.split("\\,");
		objectId = Integer.parseInt( columns[ 0 ] );
		termId = Integer.parseInt( columns[ 1 ] );
		frequency = Integer.parseInt( columns[ 2 ] );
	}
	// this function is the inverse document frequency. It multiplies the frequency of 
	// each term i by log(N/dfi)
	void IDF( Map<Integer, Integer> termDocumentFrequency, int N ){
		//document frequency = number of documents that contain the term
		int documentFrequency = termDocumentFrequency.get( termId );
		float old = frequency;
		try{
			this.frequency = this.frequency * (float)Math.log( (float) N / (float)documentFrequency );
		}
		catch( Exception e ){
			System.out.println( "Exception caught in IDF()" );
		}
	}
}

public class sphkmeans{
	
	public static float[] addVector( float[] mean, float[] vector ){
		float[] sum = new float[ mean.length ];
		for( int i = 0; i < mean.length; i ++ ){
			try{
				sum[ i ] = mean[ i ] + vector[ i ];
			}
			catch( Exception e ){
				System.out.println( "Exception caught in addVector()" );
			}
		}
		return sum;
	}
	
	// computes mean and returns mean for each cluster
	// input: documentClusterMap = document id as key, cluster document vector as value 
	public static Map<Integer, float[]> computeMeanOfEachCluster( Map<Integer, Integer> documentClusterMap, float[][] documentTermMatrix ){
		Map<Integer, float[]> clusterMeanMap = new HashMap<Integer, float[]>();
		for( Integer documentId : documentClusterMap.keySet() ){
			// get cluster id from document cluster map
			int clusterId = documentClusterMap.get( documentId );
			// compute mean for each cluster
			float[] mean = clusterMeanMap.get( clusterId );
			if( mean != null ){
				mean = addVector( mean, documentTermMatrix[ documentId ] );
				clusterMeanMap.put( clusterId, mean );
			}
			else{
				clusterMeanMap.put( clusterId, documentTermMatrix[ documentId ] );
			}
		}
		Map<Integer, float[]> cleanClusterMeanMap = new HashMap<Integer, float[]>();	
		int index = 0;
		for( float[] mean : clusterMeanMap.values() ){
			float sum = 0;
			for( int i = 0; i < mean.length; i ++ ){
				sum += mean[ i ]; 
			}
			for( int i = 0; i < mean.length; i ++ ){
				mean[ i ] = mean[ i ] / sum; 
			}
			cleanClusterMeanMap.put( index, mean );
			index += 1;
		}
		return cleanClusterMeanMap;
		
	}
	
	// computes cosine similiartiy between two points, given cluster is a float array 
	public static float computeCosineSimilarity( float[][] documentTermMatrix, float[] clusterPoint, int documentId ){
		float similarity = 0;
		for( int i = 0; i < documentTermMatrix[1].length; i ++ ){
			try{
				similarity += clusterPoint[ i ] * documentTermMatrix[ documentId ][ i ];
			}
			catch( NullPointerException e ){
				System.out.println( "Null pointer exception caught in computeCosineSimilarity" );
			}
			catch( Exception g ){
				System.out.println( "Exception caught in computeCosineSimilarity" );
			}
		}
		return similarity;		
	}
	
	// computes cosine similarity between two documents, giving cluster id 
	public static float computeCosineSimilarity( float[][] documentTermMatrix, int clusterPoint, int documentId ){
		float similarity = 0;
		for( int i = 0; i < documentTermMatrix[1].length; i ++ ){
			try{
				similarity += documentTermMatrix[ clusterPoint ][ i ] * documentTermMatrix[ documentId ][ i ];
			}
			catch( Exception e ){
				System.out.println( "Exception caught in computeCosineSimilarity" );
			}
		}
		return similarity;
	}
	
	// makes matrix where row is document id and column is term id and the entry is the frequency of the term in the document
	// each row is (i.e document) sums up to 1 (since they are unit vector) 
	public static float[][] makeMatrix( ArrayList<DocumentVector> documents, int numberOfDocuments, int numberOfTerms ){
		System.out.println( "got into make matrix" );
		float[][] documentTermMatrix = new float[ numberOfDocuments + 1 ][ numberOfTerms ];
		System.out.println( "initiliazing in make matrix" );
		for( int i = 0; i < documents.size(); i ++ ){
			DocumentVector document = documents.get( i );
			int documentId = document.objectId;
			int termId = document.termId;
			float frequency = document.frequency;
			documentTermMatrix[ documentId ][ termId ] = frequency;
		}
		return documentTermMatrix;
	}
	
	// helper function for normalize document vector
	public static void alterDocumentVectors( ArrayList<DocumentVector> documents, float length, int objectId ){
		for( int i = 0; i < documents.size(); i ++ ){
			DocumentVector document = documents.get( i );
			if( document.objectId > objectId ){
				return;
			}
			if( document.objectId < objectId ){
				continue;
			}
			try{
				document.frequency = document.frequency / length;
			}
			catch( Exception e ){
				System.out.println( "Exception caught in alterDocumentVectors" );
			}
		}
	}
	
	// normalize or make unit vector to the document vectors
	public static void normalizeUnitVector( ArrayList<DocumentVector> documents ){
		DocumentVector previousDocument = null;
		float length = 0;
		for( int i = 0; i < documents.size(); i ++ ) {
			DocumentVector document = documents.get( i );
			if( previousDocument == null ){
				previousDocument = document;				
				length += document.frequency;
				continue;
			}
			if( previousDocument.objectId == document.objectId ){
				length += document.frequency;
			}
			else{
				alterDocumentVectors( documents, length, previousDocument.objectId );
				length = document.frequency;
				previousDocument = document;
			}
		}
		alterDocumentVectors( documents, length, previousDocument.objectId );
	}
	
	public static boolean isEqual( Map<Integer, float[]> current, Map<Integer, float[]> previous ){
		for( Integer key : current.keySet() ){
			if( previous.containsKey( key ) ){
				float[] previousMean = previous.get( key );
				float[] currentMean = current.get( key );
				for( int i = 0; i < currentMean.length; i++ ){
					if( currentMean[ i ] != previousMean[ i ] ){
						return false;
					}
				}
			}
			else{
				return false;
			}
		}
		return true;
	}
	
	public static float computeObjectiveFunctionValue( float[][] documentTermMatrix, Map<Integer, Integer> documentClusterMap, Map<Integer, float[]> clusterMeanMap ){
		float objectiveValue = 0;
		for( Integer key : documentClusterMap.keySet() ){
			int clusterId = documentClusterMap.get( key );
			objectiveValue += computeCosineSimilarity( documentTermMatrix, clusterMeanMap.get( clusterId ), key );
		}
		return objectiveValue;
	}
	
	public static void main( String [] args ) {
		
		if( args.length != 5 ){
			throw new IllegalArgumentException( "Please input exactly five arguments -- input-file class-file #clusters #trials output-file" );
		}
		String inputFile = args[ 0 ];
		String classFile = args[ 1 ];
		int clusters = Integer.parseInt( args[ 2 ] );
		int trials = Integer.parseInt( args[ 3 ] );
		String outputFile = args[ 4 ];
		
		// array to store document vectors
		ArrayList<DocumentVector> documents = new ArrayList<DocumentVector>();
		// total number of documents = N
		int numberOfDocuments = 0;
		// map between a term id and the number of documents it is in ( document frequency )
		Map<Integer, Integer> termDocumentFrequency = new HashMap<Integer, Integer>();
		try {
			FileReader fileReader = new FileReader( inputFile );
			BufferedReader bufferedReader = new BufferedReader( fileReader );
			String line = "";
			// put each line of input file into document vectors
			while(( line = bufferedReader.readLine()) != null ){
				DocumentVector document = new DocumentVector( line );
				documents.add( document );
				numberOfDocuments = Math.max( numberOfDocuments, document.objectId );
				int termId = document.termId;
				if ( termDocumentFrequency.containsKey( termId  ) ){
					int frequency = termDocumentFrequency.get( termId );
					frequency = frequency + 1;
					termDocumentFrequency.put( termId, frequency );
				}
				else{
					termDocumentFrequency.put( termId, 1 );
				}
			}
			bufferedReader.close();
		}
		catch( FileNotFoundException ex ){
			System.out.println( "Unable to open input file" );
		}
		catch( IOException ex ) {
			System.out.println( "Error reading input file" );
		}
		for( int i = 0; i < documents.size(); i ++ ){
			DocumentVector document = documents.get( i );
			document.IDF( termDocumentFrequency, numberOfDocuments );
		}
		// normalize each document vector 
		normalizeUnitVector( documents );
		
		int numberOfTrials = 0;
		float maxObjectiveValue = 0;
		int bestTrial = 0;
		Map<Integer, Integer> bestDocumentMapCluster = new HashMap<Integer, Integer>();
		while ( numberOfTrials < trials ) {
		
			// random choose initial K cluster points
			Random random = new Random( 2*numberOfTrials + 1 );
			ArrayList<Integer> clusterPoints = new ArrayList<Integer>();
			for( int i = 0 ; i < clusters ; i ++ ){
				int documentId = random.nextInt( numberOfDocuments );
				// plus 1 here because documentId starts from 1
				documentId = documentId + 1;
				clusterPoints.add( documentId );
			}	
			
			
			// 
			//float[][] documentTermMatrix = new float[ numberOfDocuments + 1 ][ termDocumentFrequency.size() ];
			System.out.println( "initialized matrix" );
			float[][] documentTermMatrix = makeMatrix( documents, numberOfDocuments, termDocumentFrequency.size() );
			System.out.println( "finished initializing matrix" );
			// maps document to a cluster 
			Map<Integer, Integer> documentClusterMap = new HashMap<Integer, Integer>();
			for( int i = 1; i < (numberOfDocuments + 1); i ++ ){
				float maxSimilarity = 0; 
				int clusterNo = -1;
				for( int j = 0; j < clusterPoints.size(); j ++ ){
					float similarity = computeCosineSimilarity( documentTermMatrix, clusterPoints.get( j ), i );
					if( maxSimilarity < similarity ){
						maxSimilarity = similarity;
						clusterNo = clusterPoints.get( j );
					}
				}
				if( clusterNo >= 0 ){
					documentClusterMap.put( i, clusterNo );
				}
			}
			
			// compute mean of each cluster
			Map<Integer, float[]> clusterMeanMap = computeMeanOfEachCluster( documentClusterMap, documentTermMatrix );		
			
			Map<Integer, float[]> previousMeanMap = new HashMap<Integer, float[]>();
			// keep doing this until mean of each cluster does not change from previous
			int limit = 0;
			while( true ){
				// reassign document to cluster 
				documentClusterMap = new HashMap<Integer, Integer>();
				for( int i = 1; i < (numberOfDocuments + 1); i ++ ){
					float maxSimilarity = 0; 
					int clusterNo = -1;
					for( Integer key : clusterMeanMap.keySet() ){
						float similarity = computeCosineSimilarity( documentTermMatrix, clusterMeanMap.get( key ), i );
						if( maxSimilarity < similarity ){
							maxSimilarity = similarity;
							clusterNo = key;
						}
					}
					if( clusterNo >= 0 ){
						documentClusterMap.put( i, clusterNo );
					}
				}
				
				// compute mean of each cluster
				clusterMeanMap = computeMeanOfEachCluster( documentClusterMap, documentTermMatrix );
				
				limit += 1;
				if( limit >= 20 ){
					break;
				}
				if( isEqual( clusterMeanMap, previousMeanMap ) ){
					break;
				}
				else{
					previousMeanMap = clusterMeanMap;
				}
				
			}
			float objectiveValue = computeObjectiveFunctionValue( documentTermMatrix, documentClusterMap, clusterMeanMap );
			if ( objectiveValue > maxObjectiveValue ){
				maxObjectiveValue = objectiveValue;
				bestTrial = 2*numberOfTrials + 1;
				bestDocumentMapCluster = documentClusterMap;
			}
			numberOfTrials += 1;
		}
		
		// write best trial to file 
		BufferedWriter output = null;
		try{
			File file = new File( outputFile );
			output = new BufferedWriter( new FileWriter( file ) ) ;
			ArrayList<Integer> keys = new ArrayList<Integer>( bestDocumentMapCluster.keySet() );
			Collections.sort( keys );
			for( Integer documentId : keys ){
				output.write( documentId + "," + bestDocumentMapCluster.get( documentId ) + "\n" );
			}
			output.close();
		}
		catch( IOException e ){
			System.out.println( "Could not write to outputfile" );
		} 
		
		// matrix of dimensions ( # of clusters ) * ( # of classes ), entry is number of documents that belong to the cluster
		int[][] clusterClassFrequencyMatrix = new int[ clusters ][ 20 ];
		//read from class-file
		try {
			FileReader fileReader = new FileReader( classFile );
			BufferedReader bufferedReader = new BufferedReader( fileReader );
			String line = "";
			// put each line of input file into document vectors
			String previousLabel = "";
			int indexLabel = -1; 
			while(( line = bufferedReader.readLine()) != null ){
				String[] columns = line.split("\\,");
				int objectId = Integer.parseInt( columns[ 0 ] );
				String label = columns[ 1 ];
				if( previousLabel.equals( label ) ){

					System.out.println( "label " + label + " indexLabel " + indexLabel + " cluster " + bestDocumentMapCluster.get( objectId ) );
					if( bestDocumentMapCluster.get( objectId ) != null ){
						clusterClassFrequencyMatrix[ bestDocumentMapCluster.get( objectId ) ][ indexLabel ]+= 1;
					}
				}
				else{
					indexLabel += 1;
					clusterClassFrequencyMatrix[ bestDocumentMapCluster.get( objectId ) ][ indexLabel ] += 1;
					previousLabel = label;
				}
			}
			bufferedReader.close();
		}
		catch( FileNotFoundException ex ){
			System.out.println( "Unable to open class file" );
		}
		catch( IOException ex ) {
			System.out.println( "Error reading class file" );
		}
		
		// compute the sum of the row of each cluster( i.e. total documents in the cluster
		// output matrix to screen
		System.out.println( "Two dimensional matrix of dimensions (# of clusters)*(# of classes)" );
		int[] valuesInEachCluster = new int[ clusters ];
		for( int i = 0; i < clusterClassFrequencyMatrix.length; i ++ ){
			int count = 0;
			for( int j = 0; j < clusterClassFrequencyMatrix[0].length; j ++ ){
				System.out.print( clusterClassFrequencyMatrix[i][j] + " " );
				count += clusterClassFrequencyMatrix[i][j];
			}
			valuesInEachCluster[ i ] = count;
			System.out.println();
		}
		
		// compute entropy and purity for each cluster
		float[] entropyCluster = new float[ clusters ];
		float[] purityCluster = new float[ clusters ];
		for( int i = 0; i < clusterClassFrequencyMatrix.length; i ++ ){
			entropyCluster[ i ] = 0;
			if( valuesInEachCluster[i] == 0 ){
				continue;
			}
			float[] Pij = new float[ clusterClassFrequencyMatrix[0].length ];
			for( int j = 0; j < clusterClassFrequencyMatrix[0].length; j ++ ){
				float probability = ( (float) clusterClassFrequencyMatrix[i][j] / (float) valuesInEachCluster[i] );
				Pij[ j ] = probability;
				if( probability == 0 ){
					continue;
				}
				entropyCluster[ i ] += Math.abs( probability * ( Math.log( probability ) / Math.log( 2 ) ) );
			}
			Arrays.sort( Pij );
			float purity = Pij[ Pij.length - 1 ]; 
			purityCluster[ i ] = purity;
		}
		
		float purity = 0;
		float entropy = 0;
		for( int i = 0; i < entropyCluster.length; i ++ ){
			purity += ( purityCluster[ i ] * ( (float) valuesInEachCluster[ i ] / numberOfDocuments ) );
			entropy += ( entropyCluster[ i ] * ( (float) valuesInEachCluster[ i ] / numberOfDocuments ) );
		}
		
		//write best objective function value to output
		System.out.println( "Best Objective function value: " + maxObjectiveValue );
		System.out.println( "Entropy: " + entropy );
		System.out.println( "Purity: " + purity );

		
	}
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
}
