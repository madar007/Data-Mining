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
	
	public void minus( SparseVector documentVector) {
		for( int i = 0; i < N; i ++ ){
			float newValue = this.get( i ) - documentVector.get( i );
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
	
	public float computeDistance(){
		float distance = 0.0f;
		for( Integer i : this.map.keySet() ){
			distance += this.get( i ) * this.get( i );
		}
		distance = (float) Math.sqrt( distance );
		return distance;
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
		}
		return similarity;		
	}
}


class SparseMatrix{
	private final int N;         		 	// rows of the matrix
	private final int M;		  		 	// columns of the matrix
	private SparseVector[] rows; 	  		// the rows, each row is a sparse vector
	private ArrayList<Integer> columns; 	// used to track the columns used in the matrix 

	// initialize an N-by-N matrix of all 0s
	public SparseMatrix(int row, int col) {
		this.N  = row;
		this.M = col;
		rows = new SparseVector[N];
		columns = new ArrayList<Integer>();
		for (int i = 0; i < N; i++)
			rows[i] = new SparseVector(col);
	}

	// put A[i][j] = value
	public void put(int i, int j, float value) {
		if (i < 0 || i >= N) throw new RuntimeException("Illegal index of row " + i );
		if (j < 0 || j >= M) throw new RuntimeException("Illegal index of column " + j);
		rows[i].put(j, value);
		columns.add( j );
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
	
	public SparseVector multiply( SparseVector w ){
		SparseVector vector = new SparseVector( w.size() );
		for( Integer i : w.getMap().keySet() ) {
			float temp = 0.0f;
			for( int j = 0; j < columns.size(); j ++ ){
				int k = columns.get( j );
				temp += this.get( i, k ) * w.get( i );
			}
			vector.put( i, temp );
		}
		return vector;
	}
	
	public ArrayList<Integer> getColumns(){
		return columns;
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
}

public class regression{
	
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
	
	public static float computeRidgeRegression( SparseVector w, SparseVector wNew, 
						SparseMatrix testSparseMatrix, float lambda, SparseVector y ){
		SparseVector wVector = testSparseMatrix.multiply( w );
		wVector.minus( y );
		SparseVector wNewVector = testSparseMatrix.multiply( w );
		wNewVector.minus( y );
		float wDistance = wVector.computeDistance();
		float wLambdaDistance = w.computeDistance() * lambda;
		float wFunction = wDistance + wLambdaDistance;
		float wNewDistance = wNewVector.computeDistance();
		float wNewLambdaDistance = wNew.computeDistance() * lambda;
		float wNewFunction = wNewDistance + wNewLambdaDistance;	
		return ( wFunction - wNewFunction );	
	}
	
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
			String[] documentClassNameRelation ){
		float precision = computePrecision( className, positiveVE, documentClassNameRelation );
		float recall = computeRecall( className, positiveVE, negativeVE, documentClassNameRelation );
		float f1Score = ( 2 * precision * recall ) / ( precision + recall );
		return f1Score;		
	}
	
	public static void main( String args[] ){
		
		// parse input arguments
		if( args.length != 9 ){
			System.out.println( "Incorrect # of arguments: java classifier-name input-file input-rlabel-file train-file test-file class-file features-label-file feature-representation-option output-file val-file " );
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
		String valFile = args[ 8 ];
		
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
		
		// put validation documents in a map so it is fast to see if a document is in the test set 
		Map<Integer, Integer> valDocuments = new HashMap<Integer, Integer>();
		try {
			FileReader fileReader = new FileReader( valFile );
			BufferedReader bufferedReader = new BufferedReader( fileReader );
			String line = "";
			// put each line of input file into document vectors
			while(( line = bufferedReader.readLine()) != null ){
				int documentId = Integer.parseInt( line );
				valDocuments.put( documentId, 0 ); 
			}
			bufferedReader.close();
		}
		catch( FileNotFoundException ex ){
			System.out.println( "Unable to open validation input file" );
		}
		catch( IOException ex ) {
			System.out.println( "Error reading validation input file" );
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
		
		// normalize each document vector 
		normalizeUnitVector( documents );
		
		// make trainSparseMatrix and testSparseMatrix from document vectors
		SparseMatrix trainSparseMatrix = new SparseMatrix( numberOfDocuments + 1, numberOfTerms + 1 );
		SparseMatrix valSparseMatrix = new SparseMatrix( numberOfDocuments + 1, numberOfTerms + 1 );
		SparseMatrix testSparseMatrix = new SparseMatrix( numberOfDocuments + 1, numberOfTerms + 1 );
		for( int i = 0; i < documents.size(); i ++ ){
			DocumentVector document = documents.get( i );
			if( trainingDocuments.get( document.objectId ) != null ) {
				trainSparseMatrix.put( document.objectId, document.termId, document.frequency );
			}
			else if( testDocuments.get( document.objectId ) != null ){
				testSparseMatrix.put( document.objectId, document.termId, document.frequency );
			}
			else{
				valSparseMatrix.put( document.objectId, document.termId, document.frequency );
			}
		}
		 
		// store document - class relation in a map
		Map<Integer, String> documentClassRelation = new HashMap<Integer, String>();
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
				documentClassRelation.put( documentId, className );
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
		
		// computes y vector for each model -- which will represent 1 if that document is in that class
		Map<String, SparseVector> y = new HashMap<String, SparseVector>();
		SparseVector yVector;
		// for each training documents
		for( Integer i : trainingDocuments.keySet() ){
			String className = documentClassRelation.get( i );
			if( y.get( className ) != null ){
				yVector = y.get( className );
				yVector.put( i , 1.0f );
				y.put( className, yVector );
			}
			else{
				yVector = new SparseVector( numberOfTerms + 1 );
				yVector.put( i, 1.0f );
				y.put( className, yVector );
			}
		}
		
		// CHECKPOINT ! // should equal 2694
		int sum  = 0;
		for( String key : y.keySet() ){
			sum += y.get( key ).nnz();
		}
		System.out.println( sum );
		

		for( String className : classDocumentRelation.keySet() ){
			// initialize each dimension in vector w to a random number between 0 and 1.  /
			SparseVector w = new SparseVector( numberOfTerms + 1 );
			ArrayList<Integer> wColumns = trainSparseMatrix.getColumns();
			for( int i = 0; i < wColumns.size(); i ++ ){
				w.put( wColumns.get( i ), (float) Math.random() );
			}
			System.out.println( className );
			float f1Score = 0.0f;
			float[] lambdaArray = new float[ 6 ];
			lambdaArray[0] = 0.01f;
			lambdaArray[1] = 0.05f;
			lambdaArray[2] = 0.1f;
			lambdaArray[3] = 0.5f;
			lambdaArray[4] = 1f;
			lambdaArray[5] = 10f ;
			int lambdaIndex = 0;
			float lambda = lambdaArray[ lambdaIndex ];
			ArrayList<Integer> kColumns;
			while( lambdaIndex <= 5 ){
			while( true ){
				SparseVector newW = new SparseVector( numberOfTerms + 1 );
				// **********		Storing variables for formula ************ // 
				
				float firstTerm = 0.0f;
				float secondTerm = 0.0f;
				float thirdTerm = 0.0f;
				
				// ***********************************************************//
				/* loop only column trainSparseMatrix that have nonzeros
				 kColumns will only have integers of the column number that is nonzero in 
				 the trainSparseMatrix. */
				kColumns = trainSparseMatrix.getColumns();
				yVector = y.get( className );
				/*
					This formula is done in the way that Professor George explained in class. 
				*/
				// this is like looping for the w values
				for( int x = 0; x < kColumns.size(); x ++ ){
					int k = kColumns.get( x );
					for( Integer i : yVector.getMap().keySet() ){
						firstTerm += yVector.get( i ) * trainSparseMatrix.get( i, k );
					}
					for( Integer i : trainingDocuments.keySet() ){
						for( Integer j : w.getMap().keySet() ){
							if( j == k ){
								continue;
							}
							secondTerm += trainSparseMatrix.get( i , j ) * 	
								w.get( i ) * trainSparseMatrix.get( i, k );
						}
						thirdTerm += trainSparseMatrix.get( i , k ) * trainSparseMatrix.get( i , k ); 
					}
					float wPrime = ( firstTerm - secondTerm ) / ( thirdTerm + lambda );
					wPrime = Math.max( wPrime, 0.0f );
					newW.put( k, wPrime );
					//System.out.println( wPrime );
				}
				// check that error between the two is less than 0.001
				float distance = computeRidgeRegression( w, newW, testSparseMatrix, lambda, yVector );
				if( distance < 0.001 ){
					break;
				}
			}
			float newf1Score = computeF1Score( className, kColumns, wColumns, documentClassNameRelation );
			f1Score = Math.max( f1Score, newf1Score );

		}

			lambda = lambdaArray[ lambdaIndex + 1 ];
			lambdaIndex += 1;

		}
		
	}
}
