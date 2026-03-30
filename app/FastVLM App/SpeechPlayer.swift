//
//  SpeechPlayer.swift
//  FastVLM App
//
//  Reads assistant text with AVSpeechSynthesis after generation completes.
//

import AVFoundation
import Foundation

/// Speaks assistant replies one at a time. Exposes an async `waitUntilDone()` so the guidance
/// loop can block until the current utterance finishes before requesting the next inference.
///
/// Does NOT touch AVAudioSession.
@MainActor
final class SpeechPlayer: NSObject, AVSpeechSynthesizerDelegate {

    private let synthesizer = AVSpeechSynthesizer()
    private var continuation: CheckedContinuation<Void, Never>?

    var isSpeaking: Bool { synthesizer.isSpeaking }

    override init() {
        super.init()
        synthesizer.delegate = self
        #if os(iOS)
        synthesizer.usesApplicationAudioSession = true
        #endif
    }

    /// Speaks the text. If something is already playing, stops it first.
    func speak(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        synthesizer.stopSpeaking(at: .immediate)
        resumeWaiter()

        let utterance = AVSpeechUtterance(string: trimmed)
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate * 1.05
        utterance.volume = 1.0
        synthesizer.speak(utterance)
    }

    /// Blocks until the current utterance finishes (or returns immediately if nothing is playing).
    func waitUntilDone() async {
        guard synthesizer.isSpeaking else { return }
        await withCheckedContinuation { cont in
            self.continuation = cont
        }
    }

    func stop() {
        synthesizer.stopSpeaking(at: .immediate)
        resumeWaiter()
    }

    private func resumeWaiter() {
        continuation?.resume()
        continuation = nil
    }

    // MARK: - AVSpeechSynthesizerDelegate

    nonisolated func speechSynthesizer(
        _ synthesizer: AVSpeechSynthesizer,
        didFinish utterance: AVSpeechUtterance
    ) {
        Task { @MainActor in
            self.resumeWaiter()
        }
    }

    nonisolated func speechSynthesizer(
        _ synthesizer: AVSpeechSynthesizer,
        didCancel utterance: AVSpeechUtterance
    ) {
        Task { @MainActor in
            self.resumeWaiter()
        }
    }
}
