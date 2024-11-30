public class TestFeedForward {
    public static void main(String[] args) {
        int batchSize = 2;
        int embDim = 8;

        // Create dummy input
        Matrix input = Matrix.random(batchSize, embDim, 0.0, 1.0);

        // Initialize FeedForward network
        FeedForward ff = new FeedForward(embDim);

        // Forward pass
        Matrix output = ff.forward(input);

        // Print shapes
        System.out.println("Input shape: (" + input.getRows() + ", " + input.getCols() + ")");
        System.out.println("Output shape: (" + output.getRows() + ", " + output.getCols() + ")");
    }
}
