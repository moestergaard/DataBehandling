package ModelsForUbiMusic;
public class Main {

    public static void main(String[] args) {

        SVMExample svm = new SVMExample();

        double[] newData = {1.2, 3.4, 5.6, 7.8};
        int predictedClass = svm.predictClass(newData);

        System.out.println("Predicted label: " + predictedClass);
    }
}