public class LayerNorm {
    private final double eps = 1e-5;
    private final Matrix scale;
    private final Matrix shift;

    public LayerNorm(int embDim) {
        this.scale = new Matrix(1, embDim); // Initialize to 1
        this.shift = new Matrix(1, embDim); // Initialize to 0
        for (int i = 0; i < embDim; i++) {
            this.scale.getData()[0][i] = 1.0;
            this.shift.getData()[0][i] = 0.0;
        }
    }
    public void setScale(double[] scaleData) {
        this.scale.setData(new double[][] { scaleData });
    }
    
    public void setShift(double[] shiftData) {
        this.shift.setData(new double[][] { shiftData });
    }
    
    public Matrix forward(Matrix x) {
        Matrix mean = x.mean(-1);
        Matrix variance = x.variance(-1, false);
        Matrix normX = x.subtract(mean).divide(Matrix.sqrt(variance.add(eps)));
        return normX.multiply(scale).add(shift);
    }
}
