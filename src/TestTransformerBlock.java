public class TestTransformerBlock {
    public static void main(String[] args) {
        int seqLength = 10;
        int embDim = 16;
        int numHeads = 4;
        int contextLength = seqLength;
        double dropoutRate = 0.1;

        // Create dummy input
        Matrix x = Matrix.random(seqLength, embDim, 0.0, 1.0);

        // Initialize TransformerBlock
        TransformerBlock transformerBlock = new TransformerBlock(embDim, numHeads, contextLength, dropoutRate);

        // Forward pass
        Matrix output = transformerBlock.forward(x);

        // Print shapes
        System.out.println("Input shape: (" + x.getRows() + ", " + x.getCols() + ")");
        System.out.println("Output shape: (" + output.getRows() + ", " + output.getCols() + ")");
    }
}
