import java.io.FileReader;
import com.google.gson.JsonElement;
import com.google.gson.JsonParser;

public class WeightsValidator {
    public static void main(String[] args) {
        try {
            JsonElement element = JsonParser.parseReader(new FileReader("gpt2_weights.json"));
            System.out.println("Weights JSON structure is valid.");
        } catch (Exception e) {
            System.err.println("Error in weights JSON file: " + e.getMessage());
        }
    }
}
