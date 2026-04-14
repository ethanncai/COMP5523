//
// Matches trainer/sprite_prompt_example.txt: only the quoted user-goal line changes.
//

import Foundation

enum ConcisePromptTemplate {

    /// Fixed body; ``userGoal`` is the user's speech (or a single-line paraphrase).
    static func prompt(userGoal: String) -> String {
        let line = userGoal
            .replacingOccurrences(of: "\n", with: " ")
            .replacingOccurrences(of: "\"", with: "'")
            .trimmingCharacters(in: .whitespacesAndNewlines)

        return """
You guide a blind user to grasp the drink they asked for. Reply with one short spoken command only (e.g. move left, grab now).

User goal:
"\(line)"
"""
    }
}
