JFLAGS = -g
JC = javac

preprocess: 
	python getDataSet.py

default: kmeans

kmeans: sphkmeans.java
	$(JC) $(JFLAGS) sphkmeans.java

clean:
	$(RM) *.class
