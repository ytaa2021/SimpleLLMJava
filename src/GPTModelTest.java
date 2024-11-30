public class GPTModelTest {
    public static void main(String[] args) {
        int vocabSize = 1000;
        int embDim = 16;
        int contextLength = 20;
        int numLayers = 4;
        int numHeads = 4;
        double dropoutRate = 0.1;

        // Initialize the GPT model
        GPTModel model = new GPTModel(vocabSize, embDim, contextLength, numLayers, numHeads, dropoutRate);

        // Sample input token indices (e.g., a random sequence)
        int[] tokenIndices = new int[]{5, 23, 456, 789, 12, 34, 678, 90};

        // Forward pass
        Matrix logits = model.forward(tokenIndices);

        // Print output shape
        System.out.println("Logits shape: (" + logits.getRows() + ", " + logits.getCols() + ")");
        // Expected shape: (sequence length, vocabSize)
    }
}
