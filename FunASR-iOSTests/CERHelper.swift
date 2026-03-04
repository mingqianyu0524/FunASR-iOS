import Foundation

enum CERHelper {
    /// Character Error Rate via Levenshtein distance.
    /// CER = edit_distance(ref_chars, hyp_chars) / len(ref_chars)
    /// Whitespace is stripped before comparison.
    static func cer(hypothesis: String, reference: String) -> Double {
        cerWithDetails(hypothesis: hypothesis, reference: reference).cer
    }

    /// CER with D/I/S decomposition via Levenshtein DP.
    /// Returns (cer, deletions, insertions, substitutions).
    static func cerWithDetails(
        hypothesis: String,
        reference: String
    ) -> (cer: Double, deletions: Int, insertions: Int, substitutions: Int) {
        let ref = Array(reference.filter { !$0.isWhitespace })
        let hyp = Array(hypothesis.filter { !$0.isWhitespace })
        guard !ref.isEmpty else {
            return (hyp.isEmpty ? 0 : 1, 0, hyp.count, 0)
        }
        guard !hyp.isEmpty else {
            return (1.0, ref.count, 0, 0)
        }

        let n = ref.count
        let m = hyp.count

        // Each cell: (distance, deletions, insertions, substitutions)
        typealias Cell = (dist: Int, del: Int, ins: Int, sub: Int)
        var dp = [[Cell]](repeating: [Cell](repeating: (0, 0, 0, 0), count: m + 1), count: n + 1)

        for i in 0...n { dp[i][0] = (i, i, 0, 0) }
        for j in 0...m { dp[0][j] = (j, 0, j, 0) }

        for i in 1...n {
            for j in 1...m {
                if ref[i - 1] == hyp[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1]
                } else {
                    let s   = dp[i - 1][j - 1]  // substitution
                    let d   = dp[i - 1][j]       // deletion (ref token not in hyp)
                    let ins = dp[i][j - 1]        // insertion (extra hyp token)
                    let sCost = s.dist + 1
                    let dCost = d.dist + 1
                    let iCost = ins.dist + 1
                    if sCost <= dCost && sCost <= iCost {
                        dp[i][j] = (sCost, s.del, s.ins, s.sub + 1)
                    } else if dCost <= iCost {
                        dp[i][j] = (dCost, d.del + 1, d.ins, d.sub)
                    } else {
                        dp[i][j] = (iCost, ins.del, ins.ins + 1, ins.sub)
                    }
                }
            }
        }

        let result = dp[n][m]
        let cerValue = Double(result.dist) / Double(n)
        return (cerValue, result.del, result.ins, result.sub)
    }
}
