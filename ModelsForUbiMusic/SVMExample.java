package ModelsForUbiMusic;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.FileReader;
import java.io.IOException;

public class SVMExample {

    
    // public public static void main(String[] args) {

    //     double[] newData = {1.2, 3.4, 5.6, 7.8};
    //     int predictedClass = predictClass(newData);

    //     System.out.println("Predicted label: " + predictedLabel);
    // }


    private static int predictClassCalculations(double[] sample, double[][] supportVectors, double[][] coefficients, double[] intercepts) {
        double maxDistance = Double.NEGATIVE_INFINITY;
        int predictedClass = -1;
    
        for (int classIndex = 0; classIndex < intercepts.length; classIndex++) {
            double distance = 0.0;
    
            for (int i = 0; i < supportVectors.length; i++) {
                double dotProduct = 0.0;
                for (int j = 0; j < supportVectors[i].length; j++) {
                    dotProduct += supportVectors[i][j] * sample[j];
                }
                distance += dotProduct * coefficients[classIndex][i];
            }
    
            distance += intercepts[classIndex];
    
            if (distance > maxDistance) {
                maxDistance = distance;
                predictedClass = classIndex;
            }
        }
    
        return predictedClass;
    }

    public int predictClass(double[] sample) {
        // Initialize the support vectors, coefficients, and intercepts with the provided data
        double[][] supportVectors = {
            // Insert the support vectors from the given information
        };
        double[][] coefficients = {
            [[0.8205128205128205, 0.8205128205128205, 0.20296604549694083, 0.06665786858841243, 0.18092483474450682, 0.4293060923281209, 0.2349971813173972, 0.8205128205128205, 0.8205128205128205, 0.8205128205128205, 0.8205128205128205, 0.24535858486261947, 0.47025219487895326, -0.2183676401922463, -0.8888888888888888, -0.23613753096016968, -0.6949316955210124, -0.8110865193890832, -0.7736531017639208, -0.8888888888888888, -0.8888888888888888, -0.18893321178747513, -0.0, -0.44379548761647886, -0.7199678713968206, -1.5238095238095237, -1.060737022735028, -1.5238095238095237, -0.0, -0.025786317815877438, -1.5238095238095237, -1.5238095238095237], [0.8205128205128205, 0.8205128205128205, 0.6742283993314611, 0.6308982861903888, 0.0, 0.25088370123790177, 0.0, 0.8205128205128205, 0.8205128205128205, 0.8205128205128205, 0.8205128205128205, 0.22465952513284937, 0.47801460081947617, 0.44049578510329407, 0.8888888888888888, 0.8888888888888888, 0.44236809175508274, 0.8888888888888888, 0.8888888888888888, 0.8888888888888888, 0.8888888888888888, 0.8888888888888888, 0.8888888888888888, 0.8888888888888888, 0.24720689273830906, -1.421403382308881, -1.5238095238095237, -0.9547590658933837, -1.5238095238095237, -1.5238095238095237, -1.5238095238095237, -0.6586702261563261]]
            // Insert the coefficients from the given information
        };
        double[] intercepts = {
            // Insert the intercepts from the given information
        };
    
        return predictClassCalculations(sample, supportVectors, coefficients, intercepts);
    }

    private static void loadModelFromFile(String filePath) {
        try (FileReader fileReader = new FileReader(filePath)) {
            JsonParser jsonParser = new JsonParser();
            JsonObject jsonObject = jsonParser.parse(fileReader).getAsJsonObject();
    
            JsonObject supportVectorsObject = jsonObject.getAsJsonObject("support_vectors");
            double[][] supportVectors = new Gson().fromJson(supportVectorsObject, double[][].class);
    
            JsonObject coefficientsObject = jsonObject.getAsJsonObject("coefficients");
            double[][] coefficients = new Gson().fromJson(coefficientsObject, double[][].class);
    
            double[] intercepts = new Gson().fromJson(jsonObject.getAsJsonArray("intercepts"), double[].class);
    
            // Assign the loaded values to the variables
            // supportVectors, coefficients, intercepts
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}