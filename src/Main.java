import java.util.*;
import java.io.IOException;

public class Main {
    public static void main(String[] args) throws Exception {
        // Load the weights map
        Map<String, Object> weightsMap = WeightsLoader.loadWeights("gpt2_weights.json");

        // Initialize your GPTModel
        int vocabSize = 50257; // GPT-2 uses 50257 tokens
        int embDim = 768;      // For GPT-2 small
        int contextLength = 1024;
        int numLayers = 12;
        int numHeads = 12;
        double dropoutRate = 0.0;

        GPTModel model = new GPTModel(vocabSize, embDim, contextLength, numLayers, numHeads, dropoutRate);

        try {
            // Load weights into the model
            model.loadWeights(weightsMap);
        } catch (Exception e) {
            System.err.println("Error loading model weights: " + e.getMessage());
            e.printStackTrace();
            return;
        }

        // Tokenize input text using BytePairEncoding
        BytePairEncoding.Encoder encoder = BytePairEncoding.getEncoder("gpt2", "models");
        String inputText = "Hello, world!";
        List<Integer> tokenIndicesList = encoder.encode(inputText);
        int[] tokenIndices = tokenIndicesList.stream().mapToInt(Integer::intValue).toArray();

        // Run the model
        Matrix logits = model.forward(tokenIndices);

        // Process logits to generate text
        // For simplicity, we'll perform greedy decoding to generate the next token
        int generatedToken = getNextToken(logits);

        // Decode the generated token back to text
        String generatedText = encoder.decode(Collections.singletonList(generatedToken));

        // Print the generated text
        System.out.println("Input Text: " + inputText);
        System.out.println("Generated Text: " + generatedText);
    }

    private static int getNextToken(Matrix logits) {
        // Assuming logits shape is (seqLength, vocabSize)
        // We'll take the last token's logits for prediction
        double[] lastLogits = logits.getData()[logits.getRows() - 1];

        // Find the index with the highest logit (greedy decoding)
        int maxIndex = 0;
        double maxLogit = lastLogits[0];
        for (int i = 1; i < lastLogits.length; i++) {
            if (lastLogits[i] > maxLogit) {
                maxLogit = lastLogits[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }
}
