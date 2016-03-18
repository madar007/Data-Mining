import java.util.Random;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Collections;
import java.util.Arrays;
import java.util.Comparator;
import java.util.TreeMap;
import java.util.Set;
import java.io.*;

// TODO: remove CHECKPOINT !'s

class ValueComparator implements Comparator<Integer> {
 
    Map<Integer, Float> map;
 
    public ValueComparator(Map<Integer, Float> base) {
        this.map = base;
    }
 
    public int compare(Integer a, Integer b) {
        if (map.get(a) >= map.get(b)) {
            return -1;
        } else {
            return 1;
        } // returning 0 would merge keys 
    }
}

class SparseVector{
    private final int N;             // length
    private Map<Integer, Float> map;  // the vector, represented by index-value pairs

    // initialize the all 0s vector of length N
    public SparseVector(int N) {
        this.N  = N;
        this.map = new HashMap<Integer, Float>();
    }

    // put st[i] = value
    public void put(int i, float value) {
        if (i < 0 || i >= N) throw new RuntimeException("Illegal index");
        if (value == 0.0f ) map.remove(i);
        else              	map.put(i, value);
    }

    // return st[i]
    public float get(int i) {
        if (i < 0 || i >= N) throw new RuntimeException("Illegal index");
        if (map.containsKey(i)) return map.get(i);
        else                	return 0.0f;
    }	
	
	public void add( SparseVector documentVector) {
		for( int i = 0; i < N; i ++ ){
			float newValue = documentVector.get( i ) + this.get( i );
			this.put( i, newValue );
		}
	}
	
	public SparseVector divide( float divisor ){
		SparseVector result = new SparseVector( N );
		for( Integer i : this.map.keySet() ){
			result.put( i , this.get( i ) / divisor );
		}
		return result;
	}
	
	public boolean isEmpty(){
		return map.isEmpty();
	}
	
	public int size(){
		return N;
	}
	
    // return the number of nonzero entries
    public int nnz() {
        return map.size();
    }
	
	public Map<Integer, Float> getMap(){
		return this.map;
	}
	
	public String toString(){
		String string = "( ";
		for( Integer i : this.map.keySet() ){
			string = string.concat( this.get( i ) + ", ");
		}
		string = string.concat( ")");
		return string;
	}
	
	// computes cosine similiartiy between two points, given cluster is a float array 
	// because documentVectors are unit length, cosine similarity is computed by just dot product
	public float computeCosineSimilarity( SparseVector centroidVector ){
		float similarity = 0;
		for( Integer i : centroidVector.getMap().keySet() ){
			similarity += centroidVector.get( i ) * this.get( i );
			/*try{
				similarity += clusterPoint[ i ] * documentTermMatrix[ documentId ][ i ];
			}
			catch( NullPointerException e ){
				System.out.println( "Null pointer exception caught in computeCosineSimilarity" );
			}
			catch( Exception g ){
				System.out.println( "Exception caught in computeCosineSimilarity" );
			}*/
		}
		return similarity;		
	}
}


class SparseMatrix{
	private final int N;           // rows of the matrix
	private final int M;		   // columns of the matrix
	private SparseVector[] rows;   // the rows, each row is a sparse vector

	// initialize an N-by-N matrix of all 0s
	public SparseMatrix(int row, int col) {
		this.N  = row;
		this.M = col;
		rows = new SparseVector[N];
		for (int i = 0; i < N; i++)
			rows[i] = new SparseVector(col);
	}

	// put A[i][j] = value
	public void put(int i, int j, float value) {
		if (i < 0 || i >= N) throw new RuntimeException("Illegal index of row " + i );
		if (j < 0 || j >= M) throw new RuntimeException("Illegal index of column " + j);
		rows[i].put(j, value);
	}

	// return A[i][j]
	public float get(int i, int j) {
		if (i < 0 || i >= N) throw new RuntimeException("Illegal index");
		if (j < 0 || j >= M) throw new RuntimeException("Illegal index");
		return rows[i].get(j);
	}
	
	// return the whole row
	public SparseVector get( int i ){
		if (i < 0 || i >= N) throw new RuntimeException("Illegal index");
		return rows[i];
	}
	
	public int size(){
		return N;
	}
	
	public int columnSize(){
		return M;
	}
	
    public int nnz() { 
        int sum = 0;
        for (int i = 0; i < N; i++)
            sum += rows[i].nnz();
        return sum;
    }
}


class DocumentVector{
	public int objectId;
	public int termId;
	public float frequency;
	DocumentVector( String line ){
		splitLine( line );
	}
	void splitLine( String line ){
		String[] columns = line.split(" ");
		objectId = Integer.parseInt( columns[ 0 ] );
		termId = Integer.parseInt( columns[ 1 ] );
		frequency = Integer.parseInt( columns[ 2 ] );
	}
	
	float IDF( Map<Integer, Integer> trainingTermDocumentFrequency, int N ){
		int documentFrequency = trainingTermDocumentFrequency.get( termId );
		float IDFTerm = (float) Math.log( (float) N / (float)documentFrequency ) / (float) Math.log( 2 ); 
		this.frequency = this.frequency * IDFTerm;
		return IDFTerm;
	}
	
	/* THIS IS THE OLD IDF METHOD. THE NEW ONE ONLY COMPUTE USING TRAINING SET. 
	// this function is the inverse document frequency. It multiplies the frequency of 
	// each term i by log(N/dfi)
	void IDF( Map<Integer, Integer> termDocumentFrequency, int N ){
		//document frequency = number of documents that contain the term
		int documentFrequency = termDocumentFrequency.get( termId );
		float old = frequency;
		this.frequency = this.frequency * (float)Math.log( (float) N / (float)documentFrequency );
	}*/
}

public class centroid{
	
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
			document.frequency = document.frequency / length;
		}
	}
	
	// normalize or make unit vector to the document vectors
	public static void normalizeUnitVector( ArrayList<DocumentVector> documents ){
		DocumentVector previousDocument = null;
		ArrayList<DocumentVector> temp = new ArrayList<DocumentVector>();
		float length = 0;
		for( int i = 0; i < documents.size(); i ++ ) {
			DocumentVector document = documents.get( i );
			if( previousDocument == null ){
				previousDocument = document;				
				length += document.frequency;
				temp.add( document );
				continue;
			}
			if( previousDocument.objectId == document.objectId ){
				length += document.frequency;
				temp.add( document );
			}
			else{
				alterDocumentVectors( temp, length, previousDocument.objectId );
				length = document.frequency;
				previousDocument = document;
				temp = new ArrayList<DocumentVector>();
				temp.add( document );
			}
		}
		alterDocumentVectors( temp, length, previousDocument.objectId );
	}
	
	// normalize centroids
	/*public static Map<String, SparseVector> normalizeCentroids( Map<String, SparseVector> centroids, int N ){
		//Map<String, SparseVector> unitCentroids = new HashMap<String, SparseVector>();
		float length = 0;
		for( String key : centroids.keySet() ){
			SparseVector vector = centroids.get( key );
			for( Integer i : vector.getMap().keySet() ){
				length += vector.get( i );
			}
			vector.divide( length );
			centroids.put( key, vector );
		}
		return centroids;
	}*/
	
	// compute centroids of every cluster and return centroids as an array of floats 
	public static Map<String, SparseVector> computeCentroids( SparseMatrix trainSparseMatrix, 
				Map<String, ArrayList<Integer>> classDocumentRelation, int numberOfTerms ){
		Map<String, SparseVector> centroids = new HashMap<String, SparseVector>();
		// even though this is nested loop, it shouldn't take too long because outer loop is only 20, inner loop is partial documents
		int i = 0;
		for( String key : classDocumentRelation.keySet() ){
			ArrayList<Integer> documentsId = classDocumentRelation.get( key );
			SparseVector sumDocumentVector = new SparseVector( numberOfTerms );
			int length = 0;
			for( int j = 0; j < documentsId.size(); j ++ ){
				// get sparse vector from row of sparse matrix -- given document id which is the row number
				SparseVector documentVector = trainSparseMatrix.get( documentsId.get( j ) );
				// when document vector is not in the train set, don't add
				if( documentVector.isEmpty() ){
					continue;
				}
				length += 1;
				sumDocumentVector.add( documentVector );			
			}
			// TODO: is centroid suppose to be unit vector? Or just average so just divide by size? 
			centroids.put( key, sumDocumentVector.divide( (float) length ) );
			//centroids.put( key, sumDocumentVector );
			//System.out.println( sumDocumentVector.toString());
			i ++ ;
			//break;
		}
		return centroids;
	}
	
	// compute the centroid for the rest classes or the -ve classifiers
	// TODO: right now centroid is calculated by mean of the rest classes. Probably need to change this!
	// TODO: does this need to be unit vector? If centroid need not be unit vector, then this probably doesn't either 
	public static SparseVector computeOneVsRestCentroid( String oneClass, Map<String, SparseVector> centroids, int numberOfTerms ){
		SparseVector restCentroid = new SparseVector( numberOfTerms );
		for( String key : centroids.keySet() ){
			if( key.equals( oneClass ) ){
				continue;
			}
			restCentroid.add( centroids.get( key ) );
		}
		// TODO: divide by 19 here because of remaining classifiers
		restCentroid.divide( 19.0f );
		return restCentroid;
	}
	
	// returns the class name with where the test document has the maximum score
	// TODO: does this prediction score method needs to be changed to what Sara was trying to say???
	public static String getClassWithMaxPredictionScore( Map<String, SparseVector> oneVsRestCentroid, 
				SparseVector testDocumentVector, Map<String, SparseVector> centroids ){
		// prediction score now is measured by difference between its similarity to the +ve centroid and its similarity to the -ve centroid
		float maxPredictionScore = 0.0f;
		String maxSimilarityClass = "";
		for( String key : centroids.keySet() ){
			// similarity of + ve centroid
			float positiveSimilarity = testDocumentVector.computeCosineSimilarity( centroids.get( key ) );
			// similarity of - ve centroid
			float negativeSimilarity = testDocumentVector.computeCosineSimilarity( oneVsRestCentroid.get( key ) ); 
			float predictionScore = positiveSimilarity - negativeSimilarity;
			//System.out.println( predictionScore );
			if( predictionScore > maxPredictionScore ){
				maxPredictionScore = predictionScore;
				maxSimilarityClass = key;
			}
		}
		return maxSimilarityClass;
	}
	
	// positiveVE is a list of object ids that is +ve
	// documentClassNameRelation tells what class each object id belongs to (Ground Truth) 
	// className is the +ve class in consideration
	// this function computes precision for +ve (i.e prec(+ve))
	public static float computePrecision( String className, ArrayList<Integer> positiveVE,
		String[] documentClassNameRelation ){
		int TP = 0;
		int FP = 0;
		for( int i = 0; i < positiveVE.size(); i ++ ){
			// if document id in +ve has that className, increment TP, otherwise increment FP
			if( documentClassNameRelation[ positiveVE.get( i ) ].equals( className )  ){
				TP ++ ;
			}
			else{
				FP ++;
			}
		}
		float precision = (float) TP / (float) ( TP + FP );
		return precision;
	}
	
	// positiveVE is a list of object ids that is +ve
	// negativeVE is a list of object ids that is -ve, the objects below point in rankedList
	// documentClassNameRelation tells what class each object id belongs to (Ground Truth) 
	// className is the +ve class in consideration
	// this function computes recall for +ve (i.e rec(+ve))
	public static float computeRecall( String className, ArrayList<Integer> positiveVE,
		ArrayList<Integer> negativeVE, String[] documentClassNameRelation ){
		int TP = 0;
		int FN = 0;
		for( int i = 0; i < positiveVE.size(); i ++ ){
			// if document id in +ve has that className, increment TP
			if( documentClassNameRelation[ positiveVE.get( i ) ].equals( className )  ){
				TP ++ ;
			}
		}
		for( int i = 0; i < negativeVE.size(); i ++ ){
			// if document id in -ve has that className, increment FN because that means that 
			// the document belongs to that class but was classified as -ve. 
			if( documentClassNameRelation[ negativeVE.get( i ) ].equals( className )  ){
				FN ++ ;
			}
		}		
		float recall = (float) TP / (float) ( TP + FN );
		return recall;
	}
	
	// this function computes F1 score : F1(+ve) only 
	public static float computeF1Score( String className, ArrayList<Integer> positiveVE, ArrayList<Integer> negativeVE, 
			TreeMap<Integer, Float> sortedMap, HashMap<Integer, Float> objectPredictionScore, 
			String[] documentClassNameRelation ){
		float precision = computePrecision( className, positiveVE, documentClassNameRelation );
		float recall = computeRecall( className, positiveVE, negativeVE, documentClassNameRelation );
		float f1Score = ( 2 * precision * recall ) / ( precision + recall );
		return f1Score;		
	}
	
	// helper function to sort by value
	public static TreeMap<Integer, Float> SortByValue 
			(HashMap<Integer, Float> map) {
		ValueComparator vc =  new ValueComparator(map);
		TreeMap<Integer,Float> sortedMap = new TreeMap<Integer,Float>(vc);
		sortedMap.putAll(map);
		return sortedMap;
	}
	
	// compute threshold from only objects that were not used during training. i.e TEST sets
	// sort objects in decreasing order according to prediction score
	// at each point in the ranked list, compute the F1 score assuming that all objects above are +ve
	// and all objects below are predicted -ve. 
	// threshold is the value associated with the point that has the maximum F1 score. 
	// parameter className is the class in which we are considering as +ve
	// TODO: Make loop testSparseMatrix faster. 
	public static float computeThreshold( String className, SparseVector oneCentroid, SparseVector restCentroid, 
		SparseMatrix testSparseMatrix, String[] documentClassNameRelation ){
		HashMap<Integer, Float> objectPredictionScore = new HashMap<Integer, Float>();
		for( int i = 0; i < testSparseMatrix.size(); i ++ ){
			SparseVector documentVector = testSparseMatrix.get( i );
			if( documentVector.isEmpty() ){
				continue;
			}
			float positiveSimilarity = documentVector.computeCosineSimilarity( oneCentroid );
			float negativeSimilarity = documentVector.computeCosineSimilarity( restCentroid );
			float predictionScore = positiveSimilarity - negativeSimilarity;
			objectPredictionScore.put( i, predictionScore );
		}
		//System.out.println();
		// BE CAREFUL! CANNOT GET VALUE FROM SORTED MAP, MUST GET FROM OBJECTPREDICTIONSCORE MAP
		// JUST GET SORTED KEY FROM TREEMAP (I.E SORTED MAP)
		TreeMap<Integer, Float> sortedMap = SortByValue(objectPredictionScore); 
		Set<Integer> rankedList = sortedMap.keySet();
		int maxPoint = 0; 
		float maxF1Score = 0.0f;
		ArrayList<Integer> positiveVE = new ArrayList<Integer>();
		ArrayList<Integer> negativeVE = new ArrayList<Integer>( rankedList );
		for( Integer point : rankedList ){
			positiveVE.add( point );
			negativeVE.remove( point );
			float f1Score = computeF1Score( className, positiveVE, negativeVE, sortedMap, objectPredictionScore, documentClassNameRelation );
			if( f1Score > maxF1Score ){
				maxF1Score = f1Score;
				maxPoint = point;
			}
		}
		// TODO: Change this return value to return threshold!
		return maxF1Score;
	}
	
	public static void main( String args[] ){
		
		// parse input arguments
		if( args.length != 8 && args.length != 9 ){
			System.out.println( "Incorrect # of arguments: java classifier-name input-file input-rlabel-file train-file test-file class-file features-label-file feature-representation-option output-file [options] " );
			return;
		}
		String inputFile = args[ 0 ];
		String inputRLabel = args[ 1 ];
		String trainFile = args[ 2 ];
		String testFile = args[ 3 ];
		String classFile = args[ 4 ];
		String featuresLabel = args[ 5 ];
		String featureReprOption = args[ 6 ];
		String outputFile = args[ 7 ];
		String options = "";
		if( args.length == 9 ){
			options = args[ 8 ];
		}
		
		// array to store document vectors
		ArrayList<DocumentVector> documents = new ArrayList<DocumentVector>();
		// total number of documents = N
		int numberOfDocuments = 0;
		// total number of terms
		int numberOfTerms = 0; 

		// put training documents in a map so it is fast to see if a document is in the training set 
		Map<Integer, Integer> trainingDocuments = new HashMap<Integer, Integer>();
		try {
			FileReader fileReader = new FileReader( trainFile );
			BufferedReader bufferedReader = new BufferedReader( fileReader );
			String line = "";
			// put each line of input file into document vectors
			while(( line = bufferedReader.readLine()) != null ){
				int documentId = Integer.parseInt( line );
				trainingDocuments.put( documentId, 0 ); 
			}
			bufferedReader.close();
		}
		catch( FileNotFoundException ex ){
			System.out.println( "Unable to open training input file" );
		}
		catch( IOException ex ) {
			System.out.println( "Error reading training input file" );
		}
		
		// put test documents in a map so it is fast to see if a document is in the test set 
		Map<Integer, Integer> testDocuments = new HashMap<Integer, Integer>();
		try {
			FileReader fileReader = new FileReader( testFile );
			BufferedReader bufferedReader = new BufferedReader( fileReader );
			String line = "";
			// put each line of input file into document vectors
			while(( line = bufferedReader.readLine()) != null ){
				int documentId = Integer.parseInt( line );
				testDocuments.put( documentId, 0 ); 
			}
			bufferedReader.close();
		}
		catch( FileNotFoundException ex ){
			System.out.println( "Unable to open test input file" );
		}
		catch( IOException ex ) {
			System.out.println( "Error reading test input file" );
		}
		
		// map between a term id and the number of documents it is in ( document frequency )
		Map<Integer, Integer> termDocumentFrequency = new HashMap<Integer, Integer>();
		// map TF on training documents only 
		Map<Integer, Integer> trainingTermDocumentFrequency = new HashMap<Integer, Integer>();
		try {
			FileReader fileReader = new FileReader( inputFile );
			BufferedReader bufferedReader = new BufferedReader( fileReader );
			String line = "";
			// put each line of input file into document vectors
			while(( line = bufferedReader.readLine()) != null ){
				DocumentVector document = new DocumentVector( line );
				documents.add( document );
				numberOfDocuments = Math.max( numberOfDocuments, document.objectId );
				numberOfTerms = Math.max( numberOfTerms, document.termId );
				int termId = document.termId;
				if ( termDocumentFrequency.containsKey( termId  ) ){
					int frequency = termDocumentFrequency.get( termId );
					frequency = frequency + 1;
					termDocumentFrequency.put( termId, frequency );
				}
				else{
					termDocumentFrequency.put( termId, 1 );
				}
				// if document is in the training set, map its TF into trainingTermDocumentFrequency
				if( trainingDocuments.get( document.objectId ) != null ){
					if( trainingTermDocumentFrequency.containsKey( termId ) ){
						int frequency = trainingTermDocumentFrequency.get( termId );
						frequency = frequency + 1;
						trainingTermDocumentFrequency.put( termId, frequency );
					}
					else{
						trainingTermDocumentFrequency.put( termId, 1 );
					}
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
		
		// CHECKPOINT !
		/*System.out.println( termDocumentFrequency.get( 15412 ) ); // output should be 24
		System.out.println( trainingTermDocumentFrequency.get( 15412 ) ); // output should be 13
		System.out.println( termDocumentFrequency.size() ); // output: 71944
		System.out.println( trainingTermDocumentFrequency.size() ); // output: 51257
		System.out.println( termDocumentFrequency.get( 1 ) ); // 5
		System.out.println( trainingDocuments.size() ); // 3367
		DocumentVector test = documents.get( 0 ); 
		test.IDF( trainingTermDocumentFrequency, trainingDocuments.size()  );
		System.out.println( test.frequency ); // 9.39532*/
		
		// map term-IDF value for TRAINING DOCUMENTS ONLY
		Map<Integer, Float> termIDFValue = new HashMap<Integer, Float>();
		// computes IDF of each term that is in TRAINING DOCUMENTS ONLY
		if( featureReprOption.equals( "tfidf" ) ){
			for( int i = 0; i < documents.size(); i ++ ){
				DocumentVector document = documents.get( i );
				if( trainingDocuments.get( document.objectId ) != null ){
					float IDFValue = document.IDF( trainingTermDocumentFrequency, trainingDocuments.size() );
					termIDFValue.put( document.termId, IDFValue );
				}
			}
			// TODO: What should happen to terms not in training documents? Do they just have TF values
			// when computing the unit length. 
			for( int i = 0; i < documents.size(); i ++ ){
				DocumentVector document = documents.get( i );
				if( testDocuments.get( document.objectId ) != null ){
					if( termIDFValue.get( document.termId ) != null ){
						document.frequency = document.frequency * termIDFValue.get( document.termId );
					}
					else{
						document.frequency = document.frequency * (float) Math.log( 
							(float) trainingDocuments.size() / 1.0f ) 
								/ (float) Math.log( 2 ); 
					}
				}
			}
		}
		
		if( featureReprOption.equals( "binary" )){
			for( int i = 0; i < documents.size(); i ++ ){
				DocumentVector document = documents.get( i );
				document.frequency = 1;
			}
		}
		
		if( featureReprOption.equals( "sqrt" )){
			for( int i = 0; i < documents.size(); i ++ ){
				DocumentVector document = documents.get( i );
				document.frequency = (float) Math.sqrt( document.frequency );
			}
		}
		
		if( featureReprOption.equals( "binaryidf" )){
			for( int i = 0; i < documents.size(); i ++ ){
				DocumentVector document = documents.get( i );
				if( termIDFValue.get( document.termId ) != null ){
					document.frequency = termIDFValue.get( document.termId );
				}
				else{
					document.frequency = 0;
				}
			}
		}
		
		if( featureReprOption.equals( "sqrtidf" )){
			for( int i = 0; i < documents.size(); i ++ ){
				DocumentVector document = documents.get( i );
				document.frequency = (float) Math.sqrt( document.frequency );
				if( termIDFValue.get( document.termId ) != null ){
					document.frequency = document.frequency * termIDFValue.get( document.termId );
				}
				else{
					document.frequency = 0;
				}
			}
		}
		
	
		
		// CHECKPOINT ! 
		//DocumentVector test3 = documents.get( 0 ); 
		//System.out.println( test3.objectId + " " + test3.termId + " " + test3.frequency ); // 9.39532
		
		// normalize each document vector 
		normalizeUnitVector( documents );
		
		
		// make trainSparseMatrix and testSparseMatrix from document vectors
		SparseMatrix trainSparseMatrix = new SparseMatrix( numberOfDocuments + 1, numberOfTerms + 1 );
		SparseMatrix testSparseMatrix = new SparseMatrix( numberOfDocuments + 1, numberOfTerms + 1 );
		for( int i = 0; i < documents.size(); i ++ ){
			DocumentVector document = documents.get( i );
			try{
				if( trainingDocuments.get( document.objectId ) != null ) {
					trainSparseMatrix.put( document.objectId, document.termId, document.frequency );
				}
				else {
					testSparseMatrix.put( document.objectId, document.termId, document.frequency );
				}
			}
			catch( OutOfMemoryError e )	{
				System.out.println( "Out of memory caught in sparseMatrix" );
			}
			catch( Exception g ){
				System.out.println( "Exception caught in sparseMatrix" );
			}
				
		}
		
		// CHECKPOINT ! 
		/*System.out.println( sparseMatrix.nnz() ); // 805310
		System.out.println( trainSparseMatrix.nnz() ); // 417652
		System.out.println( testSparseMatrix.nnz() ); // 387658
		System.out.println( termIDFValue.size() ); // 51257*/
		
		// CHECKPOINT ! Sum should all be 1
		/*DocumentVector test2 = documents.get( 0 ); 
		System.out.println( test2.objectId + " " + test2.termId + " " + test2.frequency ); // 0.0012813566
		System.out.println( sparseMatrix.get( test2.objectId, test2.termId ));
		float sum = 0;
		for( int i = 0; i < sparseMatrix.size(); i ++ ){
			sum = 0;
			for( int j = 0; j < sparseMatrix.columnSize(); j ++ ){
				sum += sparseMatrix.get( i, j );
			}
			System.out.println( sum );
		}*/ 
		
		
		// Remove features not in the training documents from the test sparse matrix 
		
		// 20 iterations - each having one +ve classifier and the rest as -ve classifier (one-vs-rest);
		// compute prediction score for each and output classifier with max or min prediction score? 
		
		// store class-document relation in map. 
		Map<String, ArrayList<Integer>> classDocumentRelation = new HashMap<String, ArrayList<Integer>>();
		// can get class name relating to document by getting the index of that document id in the array 
		String[] documentClassNameRelation = new String[ numberOfDocuments + 1 ];
		try {
			FileReader fileReader = new FileReader( classFile );
			BufferedReader bufferedReader = new BufferedReader( fileReader );
			String line = "";
			// put each line of input file into document vectors
			while(( line = bufferedReader.readLine()) != null ){
				String[] columns = line.split(" ");
				int documentId = Integer.parseInt( columns[ 0 ] );
				String className = columns[ 1 ];
				if( classDocumentRelation.get( className ) != null ){
					ArrayList<Integer> documentsId = classDocumentRelation.get( className );
					documentsId.add( documentId );
					classDocumentRelation.put( className, documentsId );	
				}
				else{
					ArrayList<Integer> documentsId = new ArrayList<Integer>();
					documentsId.add( documentId );
					classDocumentRelation.put( className, documentsId );
				}
				documentClassNameRelation[ documentId ] = className;
			}
			bufferedReader.close();
		}
		catch( FileNotFoundException ex ){
			System.out.println( "Unable to open class file" );
		}
		catch( IOException ex ) {
			System.out.println( "Error reading class file" );
		}
		
		// CHECKPOINT ! should get documents id that is in class comp.graphics
		/*
		ArrayList<Integer> testDocs = classDocumentRelation.get( "comp.graphics" );
		for( int i = 0; i < testDocs.size() ; i ++ ){
			System.out.println( testDocs.get( i ) );
		}*/
		
		// compute centroids 
		Map<String, SparseVector> centroids = computeCentroids( trainSparseMatrix, classDocumentRelation, numberOfTerms + 1);
		
		// CHECKPOINT ! 
		//System.out.println( centroids.size() ); // should equal 20 
		
		// oneVsRestCentroid map 
		// computes one vs rest where one is a +ve centroid and rest is center of -ve centroids
		Map<String, SparseVector> oneVsRestCentroid = new HashMap<String, SparseVector>();
		for( String key : centroids.keySet() ){
			SparseVector restCentroid = computeOneVsRestCentroid( key, centroids, numberOfTerms + 1 );
			oneVsRestCentroid.put( key, restCentroid.divide( 19.0f ) );
		}
		
		// 	CHECKPOINT !
		// 	all centroids are unit length so sum should be 1
		/*for( String key : centroids.keySet() ){
			SparseVector vector = centroids.get( key );
			float sum = 0.0f;
			for ( Integer i : vector.getMap().keySet() ){
				sum += vector.get( i );
			}
			System.out.println( key + " \t\t" + sum );
		}*/
		
		// CHECKPOINT !
		// rest-centroids should be unit length so their sum should equal 1. 
		/*for( String key : oneVsRestCentroid.keySet() ){
			SparseVector vector = oneVsRestCentroid.get( key );
			float sum = 0.0f;
			for ( Integer i : vector.getMap().keySet() ){
				sum += vector.get( i );
			}
			System.out.println( "rest " + key + " \t\t" + sum );
		}*/
		
		
		// compute threshold for each binary classifier and put in map. 
		// compute threshold from only objects that were not used during training. i.e TEST sets
		// sort objects in decreasing order according to prediction score
		// at each point in the ranked list, compute the F1 score assuming that all objects above are +ve
		// and all objects below are predicted -ve. 
		// threshold is the value associated with the point that has the maximum F1 score. 
		Map<String, Float> classThreshold = new HashMap<String, Float>();
		for( String key : centroids.keySet() ){
			float threshold = computeThreshold( key, centroids.get( key ), oneVsRestCentroid.get( key ), testSparseMatrix, documentClassNameRelation );
			classThreshold.put( key, threshold );
			System.out.println( key + " " + threshold );
		}
		
		
		// TODO: make this faster by not looping through the whole sparseMatrix
		// TODO: check that you get same answer as when computing threshold by output to different file and make sure both are the same 
		// test document -> class map 
		Map<Integer, String> testDocumentClass = new HashMap<Integer, String>();
		for(int i = 0; i < testSparseMatrix.size(); i ++ ){
			if( testSparseMatrix.get( i ).isEmpty() ){
				continue;
			}
			String className = getClassWithMaxPredictionScore( oneVsRestCentroid, testSparseMatrix.get( i ), centroids );
			//System.out.println( className );
			testDocumentClass.put( i, className );
		}
		
		// write best trial to file 
		BufferedWriter output = null;
		try{
			File file = new File( outputFile );
			output = new BufferedWriter( new FileWriter( file ) ) ;
			ArrayList<Integer> keys = new ArrayList<Integer>( testDocumentClass.keySet() );
			Collections.sort( keys );
			for( Integer documentId : keys ){
				output.write( documentId + " " + testDocumentClass.get( documentId ) + "\n" );
			}
			output.close();
		}
		catch( IOException e ){
			System.out.println( "Could not write to outputfile" );
		} 
		
	}
}
