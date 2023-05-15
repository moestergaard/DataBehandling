import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Scanner;

import libsvm.svm_model;
import libsvm.svm;
import libsvm.svm_node;
import libsvm.svm_parameter;


public class SVMClassifier {
    public static void main(String[] args) {
        // Load the trained SVM model from file
        loadModel("svm_model.pkl");

        // Test the SVM model on new data
        
        // double[] new_data = ...; // New datapoint to classify
        // svm_node[] nodes = new svm_node[new_data.length];
        // for (int i = 0; i < new_data.length; i++) {
        //     svm_node node = new svm_node();
        //     node.index = i+1;
        //     node.value = new_data[i];
        //     nodes[i] = node;
        // }
        // double prediction = svm.svm_predict(model, nodes);
        // System.out.println("Prediction: " + prediction);
    }

    /*
    // Load a trained SVM model from a pickle file
    public static void loadModel(String filename) {
        try {
            FileInputStream file = new FileInputStream(filename);
            ObjectInputStream in = new ObjectInputStream(file);
            var model = in.readObject();
            in.close();
            file.close();
            //return model;
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
            //return null;
        }
    }
     */

    public static svm_model loadModel(String filePath) {
        svm_model model = null;
        try {
            String content = new String(Files.readAllBytes(Paths.get(filePath)));
            Scanner scanner = new Scanner(content);

            // Read SVM parameters from the file
            svm_parameter param = new svm_parameter();
            param.parse(scanner.nextLine());

            // Read the support vectors from the file
            int svCount = Integer.parseInt(scanner.nextLine());
            svm_node[][] sv = new svm_node[svCount][];
            double[] svCoefs = new double[svCount];
            for (int i = 0; i < svCount; i++) {
                String[] line = scanner.nextLine().trim().split("\\s+");
                svCoefs[i] = Double.parseDouble(line[0]);
                sv[i] = new svm_node[line.length - 1];
                for (int j = 1; j < line.length; j++) {
                    String[] field = line[j].split(":");
                    svm_node node = new svm_node();
                    node.index = Integer.parseInt(field[0]);
                    node.value = Double.parseDouble(field[1]);
                    sv[i][j - 1] = node;
                }
            }

            // Create the SVM model and set its parameters
            model = new svm_model();
            model.param = param;
            model.SV = sv;
            model.sv_coef = new double[][]{svCoefs};
            model.nr_sv = new int[]{svCount};

            scanner.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return model;
    }
}


