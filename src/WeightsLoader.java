import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import java.io.FileReader;
import java.lang.reflect.Type;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class WeightsLoader {

    // Load weights from JSON file
    public static Map<String, Object> loadWeights(String filePath) throws Exception {
        Gson gson = new Gson();
        FileReader reader = new FileReader(filePath);
        Type type = new TypeToken<Map<String, Object>>() {}.getType();
        Map<String, Object> weightsMap = gson.fromJson(reader, type);
        reader.close();
        return weightsMap;
    }

    // Convert a single JSON array to a 1D double array
    public static double[] toDoubleArray(Object obj, int expectedSize) {
        if (obj == null || !(obj instanceof List)) {
            // Replace null or invalid objects with random values
            System.err.println("Warning: Encountered invalid or null object. Filling with random values.");
            return randomDoubleArray(expectedSize);
        }

        List<?> list = (List<?>) obj;
        if (list.size() != expectedSize) {
            System.err.println("Warning: Unexpected array size. Expected " + expectedSize + ", but got " + list.size() + ". Filling with random values.");
            return randomDoubleArray(expectedSize);
        }

        double[] array = new double[expectedSize];
        for (int i = 0; i < expectedSize; i++) {
            Object value = list.get(i);
            if (value instanceof Number) {
                array[i] = ((Number) value).doubleValue();
            } else {
                // Replace invalid entries with random values
                array[i] = randomDouble();
            }
        }
        return array;
    }

    // Convert a nested JSON array to a 2D double array
    public static double[][] to2DDoubleArray(Object obj, int expectedRows, int expectedCols) {
        if (obj == null || !(obj instanceof List)) {
            // Replace null or invalid objects with random values
            System.err.println("Warning: Encountered invalid or null object. Filling with random values.");
            return random2DDoubleArray(expectedRows, expectedCols);
        }

        List<?> list = (List<?>) obj;
        if (list.size() != expectedRows) {
            System.err.println("Warning: Unexpected number of rows. Expected " + expectedRows + ", but got " + list.size() + ". Filling with random values.");
            return random2DDoubleArray(expectedRows, expectedCols);
        }

        double[][] array = new double[expectedRows][expectedCols];
        for (int i = 0; i < expectedRows; i++) {
            array[i] = toDoubleArray(list.get(i), expectedCols);
        }
        return array;
    }

    // Helper method to generate a random double array
    private static double[] randomDoubleArray(int size) {
        Random random = new Random();
        double[] array = new double[size];
        for (int j = 0; j < size; j++) {
            array[j] = randomDouble();
        }
        return array;
    }

    // Helper method to generate a random 2D double array
    private static double[][] random2DDoubleArray(int rows, int cols) {
        double[][] array = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            array[i] = randomDoubleArray(cols);
        }
        return array;
    }

    // Helper method to generate a random double value
    private static double randomDouble() {
        Random random = new Random();
        return random.nextGaussian() * 0.02; // Using standard deviation similar to typical weight initialization
    }
}
