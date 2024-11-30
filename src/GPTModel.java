import java.util.*;
public class GPTModel {
    private final int vocabSize;
    private final int embDim;
    private final int contextLength;
    private final int numLayers;
    private final TransformerBlock[] transformerBlocks;
    private final Matrix tokenEmbedding;
    private final Matrix positionEmbedding;
    private final LayerNorm finalLayerNorm;
    private final Linear outputProjection;

    public GPTModel(int vocabSize, int embDim, int contextLength, int numLayers, int numHeads, double dropoutRate) {
        this.vocabSize = vocabSize;
        this.embDim = embDim;
        this.contextLength = contextLength;
        this.numLayers = numLayers;

        // Initialize token embeddings: Shape (vocabSize, embDim)
        this.tokenEmbedding = Matrix.random(vocabSize, embDim, 0.0, 0.02);

        // Initialize positional embeddings: Shape (contextLength, embDim)
        this.positionEmbedding = Matrix.random(contextLength, embDim, 0.0, 0.02);

        // Initialize Transformer blocks
        this.transformerBlocks = new TransformerBlock[numLayers];
        for (int i = 0; i < numLayers; i++) {
            this.transformerBlocks[i] = new TransformerBlock(embDim, numHeads, contextLength, dropoutRate);
        }

        // Initialize final LayerNorm
        this.finalLayerNorm = new LayerNorm(embDim);

        // Initialize output projection layer
        this.outputProjection = new Linear(embDim, vocabSize);
    }
    

    public Matrix forward(int[] tokenIndices) {
        // Input: tokenIndices of shape (seqLength)
        int seqLength = tokenIndices.length;

        // Check if seqLength exceeds contextLength
        if (seqLength > contextLength) {
            throw new IllegalArgumentException("Sequence length exceeds model's context length.");
        }

        // Get token embeddings
        Matrix tokenEmbeddings = getTokenEmbeddings(tokenIndices); // Shape: (seqLength, embDim)

        // Get positional embeddings
        Matrix positionalEmbeddings = getPositionalEmbeddings(seqLength); // Shape: (seqLength, embDim)

        // Combine embeddings
        Matrix x = tokenEmbeddings.add(positionalEmbeddings); // Shape: (seqLength, embDim)

        // Pass through Transformer blocks
        for (TransformerBlock block : transformerBlocks) {
            x = block.forward(x); // Shape: (seqLength, embDim)
        }

        // Apply final LayerNorm
        x = finalLayerNorm.forward(x); // Shape: (seqLength, embDim)

        // Output projection to vocabulary size
        Matrix logits = x.matMul(Matrix.transpose(tokenEmbedding)); // Shape: (seqLength, vocabSize)

        return logits;
    }
    public void loadWeights(Map<String, Object> weightsMap) {
        try {
            System.out.println("Loading token embedding weights...");
            int vocabSize = this.vocabSize;
            int embDim = this.embDim;
            double[][] tokenEmbeddingWeights = WeightsLoader.to2DDoubleArray(
                weightsMap.get("wte.weight"), vocabSize, embDim
            );
            this.tokenEmbedding.setData(tokenEmbeddingWeights);
    
            System.out.println("Loading position embedding weights...");
            int maxPositionEmbeddings = this.contextLength;
            double[][] positionEmbeddingWeights = WeightsLoader.to2DDoubleArray(
                weightsMap.get("wpe.weight"), maxPositionEmbeddings, embDim
            );
            this.positionEmbedding.setData(positionEmbeddingWeights);
    
            System.out.println("Loading final LayerNorm weights...");
            double[] ln_f_weight = WeightsLoader.toDoubleArray(
                weightsMap.get("ln_f.weight"), embDim
            );
            double[] ln_f_bias = WeightsLoader.toDoubleArray(
                weightsMap.get("ln_f.bias"), embDim
            );
            this.finalLayerNorm.setScale(ln_f_weight);
            this.finalLayerNorm.setShift(ln_f_bias);
    
            for (int i = 0; i < this.numLayers; i++) {
                System.out.println("Loading weights for transformer block " + i + "...");
                TransformerBlock block = this.transformerBlocks[i];
    
                double[] ln1_weight = WeightsLoader.toDoubleArray(
                    weightsMap.get("h." + i + ".ln_1.weight"), embDim
                );
                double[] ln1_bias = WeightsLoader.toDoubleArray(
                    weightsMap.get("h." + i + ".ln_1.bias"), embDim
                );
                block.norm1.setScale(ln1_weight);
                block.norm1.setShift(ln1_bias);
    
                double[][] c_attn_weight = WeightsLoader.to2DDoubleArray(
                    weightsMap.get("h." + i + ".attn.c_attn.weight"), embDim, embDim * 3
                );
                double[] c_attn_bias = WeightsLoader.toDoubleArray(
                    weightsMap.get("h." + i + ".attn.c_attn.bias"), embDim * 3
                );
                block.attention.loadCattnWeights(c_attn_weight, c_attn_bias);
    
                double[][] c_proj_weight = WeightsLoader.to2DDoubleArray(
                    weightsMap.get("h." + i + ".attn.c_proj.weight"), embDim, embDim
                );
                double[] c_proj_bias = WeightsLoader.toDoubleArray(
                    weightsMap.get("h." + i + ".attn.c_proj.bias"), embDim
                );
                block.attention.outProj.setWeights(c_proj_weight);
                block.attention.outProj.setBias(c_proj_bias);
    
                double[] ln2_weight = WeightsLoader.toDoubleArray(
                    weightsMap.get("h." + i + ".ln_2.weight"), embDim
                );
                double[] ln2_bias = WeightsLoader.toDoubleArray(
                    weightsMap.get("h." + i + ".ln_2.bias"), embDim
                );
                block.norm2.setScale(ln2_weight);
                block.norm2.setShift(ln2_bias);
    
                double[][] fc1_weight = WeightsLoader.to2DDoubleArray(
                    weightsMap.get("h." + i + ".mlp.c_fc.weight"), embDim, embDim * 4
                );
                double[] fc1_bias = WeightsLoader.toDoubleArray(
                    weightsMap.get("h." + i + ".mlp.c_fc.bias"), embDim * 4
                );
                block.feedForward.fc1.setWeights(fc1_weight);
                block.feedForward.fc1.setBias(fc1_bias);
    
                double[][] fc2_weight = WeightsLoader.to2DDoubleArray(
                    weightsMap.get("h." + i + ".mlp.c_proj.weight"), embDim * 4, embDim
                );
                double[] fc2_bias = WeightsLoader.toDoubleArray(
                    weightsMap.get("h." + i + ".mlp.c_proj.bias"), embDim
                );
                block.feedForward.fc2.setWeights(fc2_weight);
                block.feedForward.fc2.setBias(fc2_bias);
            }
    
            System.out.println("Model weights loaded successfully!");
        } catch (Exception e) {
            System.err.println("An error occurred while loading the model weights: " + e.getMessage());
            e.printStackTrace();
            // Optionally, you can proceed without throwing the exception
            // Or re-initialize the model weights randomly
        }
    }
    
    
    public static double[][] transposeMatrix(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] transposed = new double[cols][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposed[j][i] = matrix[i][j];
            }
        }
        return transposed;
    }
    
    private Matrix getTokenEmbeddings(int[] tokenIndices) {
        int seqLength = tokenIndices.length;
        double[][] embeddings = new double[seqLength][embDim];
        for (int i = 0; i < seqLength; i++) {
            int tokenIndex = tokenIndices[i];
            if (tokenIndex < 0 || tokenIndex >= vocabSize) {
                throw new IllegalArgumentException("Token index out of bounds.");
            }
            embeddings[i] = tokenEmbedding.getRow(tokenIndex);
        }
        return new Matrix(embeddings);
    }

    private Matrix getPositionalEmbeddings(int seqLength) {
        // Return the first seqLength positional embeddings
        double[][] embeddings = new double[seqLength][embDim];
        for (int i = 0; i < seqLength; i++) {
            embeddings[i] = positionEmbedding.getRow(i);
        }
        return new Matrix(embeddings);
    }
}
