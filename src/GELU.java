public class GELU {
    public static Matrix forward(Matrix x) {
        double sqrt2OverPi = Math.sqrt(2.0 / Math.PI);
        // Compute x^3
        Matrix xCubed = x.multiply(x).multiply(x); // Since pow is element-wise, x^3 = x * x * x

        // Compute inner term: sqrt(2/π) * (x + 0.044715 * x^3)
        Matrix inner = x.add(xCubed.multiply(0.044715)).multiply(sqrt2OverPi);

        // Compute tanh(inner)
        Matrix tanhInner = inner.applyFunction(Math::tanh);

        // Compute 0.5 * x * (1 + tanhInner)
        Matrix result = x.multiply(0.5).multiply(tanhInner.add(1));

        return result;
    }
}
