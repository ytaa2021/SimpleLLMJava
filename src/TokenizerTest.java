import java.util.List;
import java.util.*;
import java.io.IOException;

public class TokenizerTest {
    public static void main(String[] args) throws IOException {
        // Assuming you have the necessary model files in the "models/gpt2" directory
        BytePairEncoding.Encoder tokenizer = BytePairEncoding.getEncoder("gpt2", "models");

        String sampleText = "Hello, world! This is a test.";

        // Encode the text
        List<Integer> tokenIds = tokenizer.encode(sampleText);
        System.out.println("Token IDs: " + tokenIds);

        // Decode the token IDs back to text
        String decodedText = tokenizer.decode(tokenIds);
        System.out.println("Decoded Text: " + decodedText);
    }
}
