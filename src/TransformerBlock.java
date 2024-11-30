public class TransformerBlock {
    public final LayerNorm norm1;
    public final MultiHeadAttention attention;
    public final LayerNorm norm2;
    public final FeedForward feedForward;

    public TransformerBlock(int embDim, int numHeads, int contextLength, double dropoutRate) {
        this.norm1 = new LayerNorm(embDim);
        this.attention = new MultiHeadAttention(embDim, embDim, contextLength, dropoutRate, numHeads);
        this.norm2 = new LayerNorm(embDim);
        this.feedForward = new FeedForward(embDim);
    }

    public Matrix forward(Matrix x) {
        // x: (seqLength, embDim)

        // First LayerNorm
        Matrix normed1 = norm1.forward(x); // Shape: (seqLength, embDim)

        // MultiHeadAttention
        Matrix attentionOut = attention.forward(normed1); // Shape: (seqLength, embDim)

        // Residual connection
        Matrix add1 = x.add(attentionOut); // Shape: (seqLength, embDim)

        // Second LayerNorm
        Matrix normed2 = norm2.forward(add1); // Shape: (seqLength, embDim)

        // FeedForward
        Matrix feedForwardOut = feedForward.forward(normed2); // Shape: (seqLength, embDim)

        // Second residual connection
        Matrix output = add1.add(feedForwardOut); // Shape: (seqLength, embDim)

        return output; // Shape: (seqLength, embDim)
    }
}
