public class Linear {
    private final Matrix weight; // Shape: (inFeatures, outFeatures)
    private final Matrix bias;   // Shape: (1, outFeatures)

    public Linear(int inFeatures, int outFeatures) {
        // Initialize weights with small random values (mean=0, std=0.02)
        this.weight = Matrix.random(inFeatures, outFeatures, 0.0, 0.02);
        // Initialize biases to zeros
        this.bias = new Matrix(1, outFeatures);
    }
    public void setWeights(double[][] weightData) {
        this.weight.setData(weightData);
    }
    
    public void setBias(double[] biasData) {
        if (biasData.length != this.bias.getCols()) {
            throw new IllegalArgumentException("Bias dimensions do not match.");
        }
        this.bias.setData(new double[][] { biasData });
    }
    
    public Matrix forward(Matrix input) {
        // Input shape: (batchSize, inFeatures)
        // Weight shape: (inFeatures, outFeatures)
        // Output shape: (batchSize, outFeatures)
        Matrix output = input.matMul(this.weight); // Matrix multiplication
        output = output.addRowVector(this.bias);   // Add bias to each row
        return output;
    }
}
