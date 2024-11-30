public class FeedForward {
    public final Linear fc1; // First linear layer
    public final Linear fc2; // Second linear layer

    public FeedForward(int embDim) {
        this.fc1 = new Linear(embDim, 4 * embDim);
        this.fc2 = new Linear(4 * embDim, embDim);
    }

    public Matrix forward(Matrix x) {
        // x: (batchSize, embDim)
        // First linear layer
        Matrix out = fc1.forward(x);     // Shape: (batchSize, 4 * embDim)
        // Activation function (GELU)
        out = GELU.forward(out);         // Shape: (batchSize, 4 * embDim)
        // Second linear layer
        out = fc2.forward(out);          // Shape: (batchSize, embDim)
        return out;
    }
}
