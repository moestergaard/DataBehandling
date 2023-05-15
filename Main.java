public class Main {

    public static void main(String[] args) {

        // create a new instance of the SVM classifier
        SVMClassifier svm = new SVMClassifier();


        // do something with the classifier, for example, load a model from a file
        svm.loadModel("svm_model.pkl");

        // use the classifier to predict the class label of a new data instance
        double[] newData = {1.2, 3.4, 5.6, 7.8};
        // double predictedLabel = svm.predict(newData);

        // System.out.println("Predicted label: " + predictedLabel);
    }
}