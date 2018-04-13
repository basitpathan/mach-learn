import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;

import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class PollingInterval {

	public static void main(String[] args) throws Exception {

		Classifier classifier = new J48();
		classifier.buildClassifier(getInstances("polling.txt"));

		System.out.println(classifier.toString());

		Instances testInstances = getInstances("2017-01-01 13","CT600", 1234567);
//		Instances testInstances = getInstances("pollingTest.txt");

		System.out.println("Prediction : " + classifier.classifyInstance(testInstances.instance(0)));
	}

	private static Instances getInstances(String fileName) throws Exception{
		Instances instances = new Instances(new BufferedReader(new FileReader("data/" + fileName)));
		instances.setClassIndex(instances.numAttributes() - 1);
		return instances;
	}

	private static Instances getInstances(String messageTime, String messageClass, double messageSize) throws Exception{

		ArrayList<Attribute> attributeList = new ArrayList<Attribute>();
		attributeList.add(new Attribute("messageTime", "yyyy-MM-dd HH"));
		attributeList.add(new Attribute("messageClass", Arrays.asList("CT600","SA100","SA800")));
		attributeList.add(new Attribute("messageSize"));
		attributeList.add(new Attribute("responseTime", Arrays.asList("1","2","3","4","5","6","7","8","9")));

		Instances instances = new Instances("polling",attributeList, 1);
		instances.setClassIndex(instances.numAttributes() - 1);

		Instance denseInstance = new DenseInstance(instances.numAttributes() -1);
		denseInstance.setValue(attributeList.get(0), attributeList.get(0).parseDate(messageTime));
		denseInstance.setValue(attributeList.get(1), messageClass);
		denseInstance.setValue(attributeList.get(2), messageSize);

		instances.add(denseInstance);

		return instances;
	}
}