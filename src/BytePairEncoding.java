import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.regex.*;
import java.util.stream.Collectors;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;


public class BytePairEncoding {

    // method to map utf-8 bytes to unicode characters
    public static Map<Integer, String> bytesToUnicode() {
        List<Integer> bs = new ArrayList<>();
        // add ranges of bytes that correspond to printable characters
        for (int i = (int) '!'; i <= (int) '~'; i++) bs.add(i);
        for (int i = (int) '¡'; i <= (int) '¬'; i++) bs.add(i);
        for (int i = (int) '®'; i <= (int) 'ÿ'; i++) bs.add(i);

        List<Integer> cs = new ArrayList<>(bs); // copy the list of bytes
        int n = 0;

        // handle other bytes that aren't already in the list
        for (int b = 0; b < 256; b++) {
            if (!bs.contains(b)) {
                bs.add(b);
                cs.add(256 + n); // map to a unicode value
                n++;
            }
        }

        // create a map of byte -> unicode character
        Map<Integer, String> byteToUnicode = new HashMap<>();
        for (int i = 0; i < bs.size(); i++) {
            byteToUnicode.put(bs.get(i), String.valueOf((char) (int) cs.get(i)));
        }

        return byteToUnicode; // return the mapping
    }

    // helper method to find all adjacent pairs of symbols in a word
    public static Set<Pair<String, String>> getPairs(String[] word) {
        Set<Pair<String, String>> pairs = new HashSet<>();
        String prevChar = word[0]; // start with the first character
        for (int i = 1; i < word.length; i++) {
            pairs.add(new Pair<>(prevChar, word[i])); // add the pair to the set
            prevChar = word[i]; // move to the next character
        }
        return pairs; // return the set of pairs
    }

    // main encoder class
    public static class Encoder {
        private final Map<String, Integer> encoder; // maps symbols to integers
        private final Map<Integer, String> decoder; // reverse map: integers to symbols
        private final Map<Integer, String> byteEncoder; // maps bytes to unicode
        private final Map<String, Integer> byteDecoder; // reverse map: unicode to bytes
        private final Map<Pair<String, String>, Integer> bpeRanks; // ranking of symbol pairs
        private final Map<String, String> cache = new HashMap<>(); // cache for processed tokens
        private final Pattern pattern; // regex pattern for tokenizing text

        // constructor to initialize the encoder
        public Encoder(Map<String, Integer> encoder, List<Pair<String, String>> bpeMerges, String errors) {
            this.encoder = encoder;
            this.decoder = encoder.entrySet().stream()
                    .collect(Collectors.toMap(Map.Entry::getValue, Map.Entry::getKey)); // reverse the encoder map
            this.byteEncoder = bytesToUnicode(); // get byte -> unicode mapping
            this.byteDecoder = byteEncoder.entrySet().stream()
                    .collect(Collectors.toMap(Map.Entry::getValue, Map.Entry::getKey)); // reverse byte mapping
            this.bpeRanks = new HashMap<>();
            for (int i = 0; i < bpeMerges.size(); i++) {
                bpeRanks.put(bpeMerges.get(i), i); // create rank for each pair
            }
            // define regex pattern for splitting text into tokens
            this.pattern = Pattern.compile("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");
        }

        // method to apply byte pair encoding to a single token
        public String bpe(String token) {
            if (cache.containsKey(token)) return cache.get(token); // return cached result if available
            String[] word = token.split(""); // split token into characters
            Set<Pair<String, String>> pairs = getPairs(word); // get initial symbol pairs

            if (pairs.isEmpty()) return token; // return token if no pairs

            while (true) {
                // find the pair with the lowest rank
                Pair<String, String> bigram = pairs.stream()
                        .min(Comparator.comparingInt(p -> bpeRanks.getOrDefault(p, Integer.MAX_VALUE)))
                        .orElse(null);

                if (bigram == null || !bpeRanks.containsKey(bigram)) break; // stop if no valid pair

                String first = bigram.first, second = bigram.second;
                List<String> newWord = new ArrayList<>();
                for (int i = 0; i < word.length; ) {
                    // find the first occurrence of the pair
                    int j = indexOf(word, first, i);
                    if (j == -1) {
                        newWord.addAll(Arrays.asList(word).subList(i, word.length)); // add remaining characters
                        break;
                    }
                    newWord.addAll(Arrays.asList(word).subList(i, j)); // add characters before the pair
                    if (j < word.length - 1 && word[j].equals(first) && word[j + 1].equals(second)) {
                        newWord.add(first + second); // merge the pair
                        i = j + 2;
                    } else {
                        newWord.add(word[j]); // add single character
                        i = j + 1;
                    }
                }
                word = newWord.toArray(new String[0]); // update the word
                pairs = getPairs(word); // update the pairs
            }

            String result = String.join(" ", word); // join the word back into a string
            cache.put(token, result); // cache the result
            return result;
        }

        // method to encode text into BPE tokens
        public List<Integer> encode(String text) {
            List<Integer> bpeTokens = new ArrayList<>();
            Matcher matcher = pattern.matcher(text); // match tokens in the text
            while (matcher.find()) {
                String token = matcher.group();
                // convert token to bytes and apply BPE
                token = token.chars()
                        .mapToObj(c -> byteEncoder.get(c))
                        .collect(Collectors.joining());
                String[] splitBpeTokens = bpe(token).split(" "); // split BPE token
                for (String bpeToken : splitBpeTokens) {
                    bpeTokens.add(encoder.get(bpeToken)); // add encoded token
                }
            }
            return bpeTokens; // return list of encoded tokens
        }

        // method to decode BPE tokens back into text
        public String decode(List<Integer> tokens) {
            // Step 1: Map token IDs to BPE tokens
            StringBuilder bpeTokensBuilder = new StringBuilder();
            for (Integer token : tokens) {
                String bpeToken = decoder.get(token);
                if (bpeToken == null) {
                    System.err.println("Warning: Token ID " + token + " not found in decoder.");
                    continue; // Skip unknown tokens
                }
                bpeTokensBuilder.append(bpeToken);
            }
            // Step 2: Concatenate BPE tokens into a single string
            String bpeText = bpeTokensBuilder.toString();

            // Step 3: Map Unicode characters back to bytes
            List<Byte> byteList = new ArrayList<>();
            for (int i = 0; i < bpeText.length(); i++) {
                String s = String.valueOf(bpeText.charAt(i));
                Integer byteValue = byteDecoder.get(s);
                if (byteValue == null) {
                    System.err.println("Warning: Character '" + s + "' not found in byteDecoder.");
                    continue; // Skip unknown characters
                }
                byteList.add(byteValue.byteValue());
            }

            // Step 4: Convert byte array to string using UTF-8 encoding
            byte[] byteArray = new byte[byteList.size()];
            for (int i = 0; i < byteList.size(); i++) {
                byteArray[i] = byteList.get(i);
            }
            String text = new String(byteArray, StandardCharsets.UTF_8);

            return text;
        }
    }
    private static int indexOf(String[] array, String element, int startIndex) {
        for (int i = startIndex; i < array.length; i++) {
            if (array[i].equals(element)) {
                return i;
            }
        }
        return -1;
    }    

    // utility method to load an encoder from files
    public static Encoder getEncoder(String modelName, String modelsDir) throws IOException {
        // read the encoder.json file
        BufferedReader encoderReader = new BufferedReader(new FileReader(modelsDir + "/" + modelName + "/encoder.json"));
        Map<String, Integer> encoder = new HashMap<>(new Gson().fromJson(encoderReader, new TypeToken<Map<String, Integer>>() {}.getType()));
        encoderReader.close();

        // read the vocab.bpe file
        BufferedReader vocabReader = new BufferedReader(new FileReader(modelsDir + "/" + modelName + "/vocab.bpe"));
        List<Pair<String, String>> bpeMerges = vocabReader.lines()
                .skip(1) // skip the first line
                .filter(line -> !line.isEmpty()) // ignore empty lines
                .map(line -> {
                    String[] split = line.split(" ");
                    return new Pair<>(split[0], split[1]); // create pair from line
                })
                .collect(Collectors.toList());
        vocabReader.close();

        return new Encoder(encoder, bpeMerges, "replace"); // return an encoder instance
    }

    // helper class to represent pairs of symbols
    public static class Pair<F, S> {
        public final F first; // first element of the pair
        public final S second; // second element of the pair

        public Pair(F first, S second) {
            this.first = first;
            this.second = second;
        }

        @Override
        public int hashCode() {
            return Objects.hash(first, second);
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            Pair<?, ?> pair = (Pair<?, ?>) o;
            return Objects.equals(first, pair.first) && Objects.equals(second, pair.second);
        }
    }
}
