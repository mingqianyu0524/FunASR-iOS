import Foundation

enum CERHelper {
    /// Character Error Rate via Levenshtein distance.
    /// CER = edit_distance(ref_chars, hyp_chars) / len(ref_chars)
    /// Whitespace is stripped before comparison.
    static func cer(hypothesis: String, reference: String) -> Double {
        let ref = Array(reference.filter { !$0.isWhitespace })
        let hyp = Array(hypothesis.filter { !$0.isWhitespace })
        guard !ref.isEmpty else { return hyp.isEmpty ? 0 : 1 }

        var prev = Array(0...ref.count)
        var curr = [Int](repeating: 0, count: ref.count + 1)

        for i in 1...hyp.count {
            curr[0] = i
            for j in 1...ref.count {
                if hyp[i - 1] == ref[j - 1] {
                    curr[j] = prev[j - 1]
                } else {
                    curr[j] = 1 + min(prev[j - 1], prev[j], curr[j - 1])
                }
            }
            swap(&prev, &curr)
        }
        return Double(prev[ref.count]) / Double(ref.count)
    }
}
