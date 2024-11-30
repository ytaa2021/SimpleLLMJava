public class MultiHeadAttention {
    private final int dOut;
    private final int numHeads;
    private final int headDim;
    public final Linear WQuery;
    public final Linear WKey;
    public final Linear WValue;

    public final Linear outProj;
    private final Matrix mask;
    private final double dropoutRate;

    public MultiHeadAttention(int dIn, int dOut, int contextLength, double dropout, int numHeads) {
        if (dOut % numHeads != 0) {
            throw new IllegalArgumentException("dOut must be divisible by numHeads.");
        }
        this.dOut = dOut;
        this.numHeads = numHeads;
        this.headDim = dOut / numHeads;
        this.dropoutRate = dropout;

        // Initialize weights
        this.WQuery = new Linear(dIn, dOut);
        this.WKey = new Linear(dIn, dOut);
        this.WValue = new Linear(dIn, dOut);
        this.outProj = new Linear(dOut, dOut);

        // Create causal mask
        this.mask = new Matrix(contextLength, contextLength);
        for (int i = 0; i < contextLength; i++) {
            for (int j = i + 1; j < contextLength; j++) {
                this.mask.getData()[i][j] = 1; // 1 means masked
            }
        }
    }

    public void loadCattnWeights(double[][] cAttnWeight, double[] cAttnBias) {
        int hiddenSize = cAttnWeight[0].length / 3;
    
        // Split weights
        double[][] qWeight = new double[cAttnWeight.length][hiddenSize];
        double[][] kWeight = new double[cAttnWeight.length][hiddenSize];
        double[][] vWeight = new double[cAttnWeight.length][hiddenSize];
    
        for (int i = 0; i < cAttnWeight.length; i++) {
            System.arraycopy(cAttnWeight[i], 0, qWeight[i], 0, hiddenSize);
            System.arraycopy(cAttnWeight[i], hiddenSize, kWeight[i], 0, hiddenSize);
            System.arraycopy(cAttnWeight[i], 2 * hiddenSize, vWeight[i], 0, hiddenSize);
        }
    
        // Set weights
        this.WQuery.setWeights(qWeight);
        this.WKey.setWeights(kWeight);
        this.WValue.setWeights(vWeight);
    
        // Split biases
        double[] qBias = new double[hiddenSize];
        double[] kBias = new double[hiddenSize];
        double[] vBias = new double[hiddenSize];
    
        System.arraycopy(cAttnBias, 0, qBias, 0, hiddenSize);
        System.arraycopy(cAttnBias, hiddenSize, kBias, 0, hiddenSize);
        System.arraycopy(cAttnBias, 2 * hiddenSize, vBias, 0, hiddenSize);
    
        // Set biases
        this.WQuery.setBias(qBias);
        this.WKey.setBias(kBias);
        this.WValue.setBias(vBias);
    }
    public void loadOutProjWeights(double[][] outProjWeights, double[] outProjBias) {
        this.outProj.setWeights(outProjWeights);
        this.outProj.setBias(outProjBias);
    }
    

    public Matrix forward(Matrix x) {
        // x: (seqLength, embDim)
        int seqLength = x.getRows();
    
        // Linear projections
        Matrix Q = WQuery.forward(x); // (seqLength, dOut)
        Matrix K = WKey.forward(x);   // (seqLength, dOut)
        Matrix V = WValue.forward(x); // (seqLength, dOut)
    
        // Reshape Q, K, V to (seqLength, numHeads, headDim)
        Matrix[] Q_heads = splitHeads(Q);
        Matrix[] K_heads = splitHeads(K);
        Matrix[] V_heads = splitHeads(V);
    
        // For each head
        Matrix[] attentionOutputs = new Matrix[numHeads];
        for (int i = 0; i < numHeads; i++) {
            // Compute attention scores
            Matrix scores = Q_heads[i].matMul(Matrix.transpose(K_heads[i])); // (seqLength, seqLength)
            scores = scores.divide(Math.sqrt(headDim));
    
            // Apply mask
            Matrix adjustedMask = mask.getSubMatrix(0, seqLength, 0, seqLength);
            scores = Matrix.applyMask(scores, adjustedMask);
    
            // Softmax
            Matrix weights = Matrix.softmax(scores);
    
            // Dropout
            weights = Matrix.dropout(weights, this.dropoutRate);
    
            // Compute attention output
            attentionOutputs[i] = weights.matMul(V_heads[i]); // (seqLength, headDim)
        }
    
        // Concatenate attention outputs
        Matrix concatAttention = concatHeads(attentionOutputs); // (seqLength, dOut)
    
        // Final linear projection
        Matrix output = this.outProj.forward(concatAttention); // Use Linear's forward method
    
        return output; // (seqLength, dOut)
    }
    
    private Matrix[] splitHeads(Matrix x) {
        // x: (seqLength, dOut)
        // Split x into numHeads matrices of shape (seqLength, headDim)
        Matrix[] heads = new Matrix[numHeads];
        for (int i = 0; i < numHeads; i++) {
            double[][] headData = new double[x.getRows()][headDim];
            for (int j = 0; j < x.getRows(); j++) {
                System.arraycopy(x.getData()[j], i * headDim, headData[j], 0, headDim);
            }
            heads[i] = new Matrix(headData);
        }
        return heads;
    }

    private Matrix concatHeads(Matrix[] heads) {
        // Concatenate numHeads matrices of shape (seqLength, headDim) into one matrix of shape (seqLength, dOut)
        int seqLength = heads[0].getRows();
        double[][] concatData = new double[seqLength][dOut];
        for (int i = 0; i < seqLength; i++) {
            int offset = 0;
            for (Matrix head : heads) {
                System.arraycopy(head.getData()[i], 0, concatData[i], offset, headDim);
                offset += headDim;
            }
        }
        return new Matrix(concatData);
    }
}
