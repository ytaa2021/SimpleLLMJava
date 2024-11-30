import java.util.*;
import java.io.IOException;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.Scanner;


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
            Scanner scanner = new Scanner(System.in);
            while (true) {
                System.out.print("Enter a prompt: ");
                String inputText = scanner.nextLine(); // Read the user input
                System.out.print("Enter a number of tokens to generate: ");
                String TokensToGenerate = scanner.nextLine(); // Read the user input
                int numTokensToGenerate;
                try {
                    numTokensToGenerate = Integer.parseInt(TokensToGenerate); // Cast String to int
                } catch (NumberFormatException e) {
                    System.out.println("Invalid input. Going with 10.");
                    numTokensToGenerate = 10;
                }
                System.out.println("Input Text: " + inputText);
                List<Integer> inputTokenIndicesList = encoder.encode(inputText);
                int[] inputTokenIndices = inputTokenIndicesList.stream().mapToInt(Integer::intValue).toArray();

                // Step 6: Run the model to generate text
                System.out.println("Running the model...");
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
                
                    // Select the next token deterministically
                    /*
                    int nextToken = IntStream.range(0, probabilities.length)
                        .reduce((index1, index2) -> probabilities[index1] > probabilities[index2] ? index1 : index2)
                        .orElse(-1);
                    */
                    // Sample the next token from the top N probabilities
                    int nextToken = sampleFromTopN(probabilities, 30); // Replace 5 with your desired top-N value

                
                    // Add the generated token to the list
                    generatedTokenIndices.add(nextToken);
                
                    // Decode and print the generated token so far
                    List<Integer> allTokenIndices = new ArrayList<>(inputTokenIndicesList);
                    allTokenIndices.addAll(generatedTokenIndices);
                    String currentOutput = encoder.decode(allTokenIndices);
                    System.out.print(currentOutput + "\r");
                }
                System.out.println();
                // Step 7: Decode the generated tokens back into text
                List<Integer> allTokenIndices = new ArrayList<>(inputTokenIndicesList);
                allTokenIndices.addAll(generatedTokenIndices);
                String generatedText = encoder.decode(allTokenIndices);

                // Step 8: Print the generated text
                System.out.println("Generated Text:");
                System.out.println(generatedText);
                
            }
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
    // Helper function to sample from the top N probabilities
    private static int sampleFromTopN(double[] probabilities, int topN) {
        // Create a list of indices and their associated probabilities
        List<Integer> indices = IntStream.range(0, probabilities.length).boxed().collect(Collectors.toList());
        indices.sort((i, j) -> Double.compare(probabilities[j], probabilities[i])); // Sort descending by probability

        // Retain only the top N indices
        List<Integer> topIndices = indices.subList(0, Math.min(topN, indices.size()));
        double[] topProbabilities = topIndices.stream().mapToDouble(i -> probabilities[i]).toArray();

        // Normalize the top probabilities
        double sum = Arrays.stream(topProbabilities).sum();
        for (int i = 0; i < topProbabilities.length; i++) {
            topProbabilities[i] /= sum;
        }

        // Sample from the top N indices using the normalized probabilities
        double r = Math.random();
        double cumulative = 0.0;
        for (int i = 0; i < topIndices.size(); i++) {
            cumulative += topProbabilities[i];
            if (r < cumulative) {
                return topIndices.get(i);
            }
        }

        // Fallback (should not reach here)
        return topIndices.get(topIndices.size() - 1);
    }
/*
    // Helper method to sample an index from a probability distribution
    private static int sampleFromDistribution(double[] probabilities, double temperature, int topK) {
        // Apply temperature
        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] = Math.pow(probabilities[i], 1.0 / temperature);
        }
    
        // Normalize probabilities after applying temperature
        double sum = Arrays.stream(probabilities).sum();
        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] /= sum;
        }
    
        // Top-k filtering
        PriorityQueue<Integer> topKIndices = new PriorityQueue<>(
            Comparator.comparingDouble(i -> probabilities[i])
        );
        for (int i = 0; i < probabilities.length; i++) {
            if (topKIndices.size() < topK) {
                topKIndices.add(i);
            } else if (probabilities[i] > probabilities[topKIndices.peek()]) {
                topKIndices.poll();
                topKIndices.add(i);
            }
        }
    
        // Normalize over the top-k probabilities
        double[] filteredProbabilities = new double[probabilities.length];
        double filteredSum = 0.0;
        for (int idx : topKIndices) {
            filteredProbabilities[idx] = probabilities[idx];
            filteredSum += probabilities[idx];
        }
        for (int i = 0; i < filteredProbabilities.length; i++) {
            filteredProbabilities[i] /= filteredSum;
        }
    
        // Sample from the filtered probabilities
        double r = Math.random();
        double cumulative = 0.0;
        for (int i = 0; i < filteredProbabilities.length; i++) {
            cumulative += filteredProbabilities[i];
            if (r < cumulative) {
                return i;
            }
        }
    
        return probabilities.length - 1; // Fallback
    }
    */
}