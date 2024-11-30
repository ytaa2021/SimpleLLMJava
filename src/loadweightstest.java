import java.util.Map;

public class loadweightstest {
    public static void main(String[] args) {
        System.out.println("Starting weight loading test...");

        try {
            // Path to your weights file
            String weightsFilePath = "gpt2_weights.json";

            // Load weights
            System.out.println("Loading model weights...");
            Map<String, Object> weightsMap = WeightsLoader.loadWeights(weightsFilePath);

            // Initialize model parameters (use GPT-2 small specs)
            int vocabSize = 50257;
            int embDim = 768;
            int contextLength = 1024;
            int numLayers = 12;
            int numHeads = 12;
            double dropoutRate = 0.0;

            // Initialize the GPTModel
            GPTModel model = new GPTModel(vocabSize, embDim, contextLength, numLayers, numHeads, dropoutRate);

            // Load weights into the model
            System.out.println("Loading weights into the model...");
            model.loadWeights(weightsMap);

            System.out.println("Weights successfully loaded into the model!");
        } catch (OutOfMemoryError e) {
            System.err.println("OutOfMemoryError: Increase the heap size with -Xmx (e.g., -Xmx8g).");
            e.printStackTrace();
        } catch (Exception e) {
            System.err.println("An error occurred while loading the model weights.");
            e.printStackTrace();
        }
    }
}
