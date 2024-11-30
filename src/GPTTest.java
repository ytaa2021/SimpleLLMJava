import java.util.*;
import java.io.IOException;

public class GPTTest {
    public static void main(String[] args) {
        // Display the JVM's max heap size
        System.out.println("Max Heap Size: " + Runtime.getRuntime().maxMemory() / (1024 * 1024) + " MB");

        try {
            // Step 1: Load the weights map
            System.out.println("Loading model weights...");
            Map<String, Object> weightsMap = WeightsLoader.loadWeights("gpt2_weights.json");

            // Step 2: Initialize your GPTModel
            int vocabSize = 50257; // GPT-2 uses 50257 tokens
            int embDim = 768;      // For GPT-2 small
            int contextLength = 1024;
            int numLayers = 12;
            int numHeads = 12;
            double dropoutRate = 0.0;

            GPTModel model = new GPTModel(vocabSize, embDim, contextLength, numLayers, numHeads, dropoutRate);

            // Step 3: Load weights into the model
            System.out.println("Loading weights into the model...");
            model.loadWeights(weightsMap);
            System.out.println("Model weights loaded successfully!");

            // Step 4: Initialize the BytePairEncoding tokenizer
            System.out.println("Initializing BytePairEncoding tokenizer...");
            BytePairEncoding.Encoder encoder = BytePairEncoding.getEncoder("gpt2", "models");

            // Step 5: Encode a sample input text
            String inputText = "Once upon a time";
            System.out.println("Input Text: " + inputText);
            List<Integer> inputTokenIndicesList = encoder.encode(inputText);
            int[] inputTokenIndices = inputTokenIndicesList.stream().mapToInt(Integer::intValue).toArray();

            // Step 6: Run the model to generate text
            System.out.println("Running the model...");
            int numTokensToGenerate = 20; // Number of tokens to generate
            List<Integer> generatedTokenIndices = new ArrayList<>();
            for (int i = 0; i < numTokensToGenerate; i++) {
                // Get the current input tokens
                int[] currentInput = concatenateArrays(inputTokenIndices, generatedTokenIndices);

                // Ensure the input does not exceed context length
                if (currentInput.length > contextLength) {
                    throw new IllegalArgumentException("Input exceeds the model's context length.");
                }

                // Run the model
                Matrix logits = model.forward(currentInput);

                // Get the logits for the last token
                double[] lastLogits = logits.getRow(logits.getRows() - 1);

                // Convert logits to probabilities using softmax
                double[] probabilities = softmax(lastLogits);

                // Sample the next token from the probability distribution
                int nextToken = sampleFromDistribution(probabilities);

                // Add the generated token to the list
                generatedTokenIndices.add(nextToken);
            }

            // Step 7: Decode the generated tokens back into text
            List<Integer> allTokenIndices = new ArrayList<>(inputTokenIndicesList);
            allTokenIndices.addAll(generatedTokenIndices);
            String generatedText = encoder.decode(allTokenIndices);

            // Step 8: Print the generated text
            System.out.println("Generated Text:");
            System.out.println(generatedText);

        } catch (OutOfMemoryError e) {
            System.err.println("Out of memory error! Consider increasing the heap size.");
        } catch (IOException e) {
            System.err.println("File error while loading model or weights: " + e.getMessage());
        } catch (IllegalArgumentException e) {
            System.err.println("Illegal argument error: " + e.getMessage());
        } catch (Exception e) {
            System.err.println("An unexpected error occurred:");
            e.printStackTrace();
        }
    }

    // Helper method to concatenate two arrays
    private static int[] concatenateArrays(int[] array1, List<Integer> list2) {
        int[] array2 = list2.stream().mapToInt(Integer::intValue).toArray();
        int[] result = new int[array1.length + array2.length];
        System.arraycopy(array1, 0, result, 0, array1.length);
        System.arraycopy(array2, 0, result, array1.length, array2.length);
        return result;
    }

    // Helper method to compute softmax
    private static double[] softmax(double[] logits) {
        double maxLogit = Arrays.stream(logits).max().orElse(0.0);
        double sumExp = 0.0;
        double[] expLogits = new double[logits.length];
        for (int i = 0; i < logits.length; i++) {
            expLogits[i] = Math.exp(logits[i] - maxLogit);
            sumExp += expLogits[i];
        }
        for (int i = 0; i < logits.length; i++) {
            expLogits[i] /= sumExp;
        }
        return expLogits;
    }

    // Helper method to sample an index from a probability distribution
    private static int sampleFromDistribution(double[] probabilities) {
        double r = Math.random();
        double cumulative = 0.0;
        for (int i = 0; i < probabilities.length; i++) {
            cumulative += probabilities[i];
            if (r < cumulative) {
                return i;
            }
        }
        // Should not reach here
        return probabilities.length - 1;
    }
}
