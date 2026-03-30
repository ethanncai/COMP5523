//
//  SpeechRecognizer.swift
//  FastVLM App
//
//  Adapted from STT example; push-to-talk: release ends the session (not isFinal).
//

import AVFoundation
import Foundation
import Speech

@Observable
@MainActor
final class SpeechRecognizer {

    enum RecognizerError: LocalizedError {
        case notAuthorized
        case notAvailable
        case noMicrophoneInput
        case invalidAudioFormat
        case audioEngineError(Error)

        var errorDescription: String? {
            switch self {
            case .notAuthorized:
                return "Speech recognition is not authorized."
            case .notAvailable:
                return "Speech recognition is not available on this device."
            case .noMicrophoneInput:
                return "No microphone input is available on this device."
            case .invalidAudioFormat:
                return "Audio input format is invalid."
            case .audioEngineError(let error):
                return "Audio engine error: \(error.localizedDescription)"
            }
        }
    }

    enum State: Equatable {
        case idle
        case preparing
        case recording
        case finalizing
    }

    var transcript: String = ""
    var state: State = .idle
    var errorMessage: String?

    var isRecording: Bool { state == .recording }

    /// When false, recognition does not end the session when Apple reports a final segment (push-to-talk).
    var endsSessionOnFinalResult: Bool = false

    private var audioEngine: AVAudioEngine?
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private var finalizeStopTask: Task<Void, Never>?
    private let speechRecognizer: SFSpeechRecognizer?

    init(locale: Locale = Locale(identifier: "en-US")) {
        self.speechRecognizer = SFSpeechRecognizer(locale: locale)
    }

    func requestAuthorization() async -> Bool {
        async let speechAuthorized = requestSpeechAuthorization()
        async let microphoneAuthorized = requestMicrophoneAuthorization()
        let isSpeechAuthorized = await speechAuthorized
        let isMicrophoneAuthorized = await microphoneAuthorized
        return isSpeechAuthorized && isMicrophoneAuthorized
    }

    /// Primes AVAudioSession, audio engine, and speech recognition so the first real hold-to-talk is responsive.
    func warmup() async {
        guard SFSpeechRecognizer.authorizationStatus() == .authorized,
              AVCaptureDevice.authorizationStatus(for: .audio) == .authorized else {
            return
        }
        guard let speechRecognizer, speechRecognizer.isAvailable else { return }

        do {
            try startRecording()

            var ticks = 0
            while state == .preparing && ticks < 80 {
                try await Task.sleep(for: .milliseconds(25))
                ticks += 1
            }
            guard state == .recording else { return }

            try await Task.sleep(for: .milliseconds(280))

            stopRecording()

            ticks = 0
            while state != .idle && ticks < 120 {
                try await Task.sleep(for: .milliseconds(25))
                ticks += 1
            }

            transcript = ""
            errorMessage = nil
        } catch {
            transcript = ""
            errorMessage = nil
        }
    }

    func startRecording() throws {
        let speechStatus = SFSpeechRecognizer.authorizationStatus()
        let microphoneStatus = AVCaptureDevice.authorizationStatus(for: .audio)
        guard speechStatus == .authorized, microphoneStatus == .authorized else {
            throw RecognizerError.notAuthorized
        }
        guard let speechRecognizer, speechRecognizer.isAvailable else {
            throw RecognizerError.notAvailable
        }
        guard AVCaptureDevice.default(for: .audio) != nil else {
            throw RecognizerError.noMicrophoneInput
        }

        finalizeStopTask?.cancel()
        finalizeStopTask = nil
        resetRecordingSessionSync()
        transcript = ""
        state = .preparing
        errorMessage = nil

        Task {
            await beginRecordingAsync(speechRecognizer: speechRecognizer)
        }
    }

    /// Ends capture and tears down audio; allows a short window for a final transcript after `endAudio`.
    func stopRecording() {
        audioEngine?.inputNode.removeTap(onBus: 0)
        audioEngine?.stop()
        recognitionRequest?.endAudio()

        if state == .recording || state == .preparing {
            state = .finalizing
        }

        finalizeStopTask?.cancel()
        finalizeStopTask = Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(450))
            guard !Task.isCancelled else { return }
            self.recognitionTask?.cancel()
            self.recognitionTask = nil
            self.recognitionRequest = nil
            self.audioEngine = nil
            // Do NOT deactivate the shared audio session — camera + TTS depend on it.
            if self.state == .finalizing {
                self.state = .idle
            }
        }
    }

    private func resetRecordingSessionSync() {
        audioEngine?.inputNode.removeTap(onBus: 0)
        audioEngine?.stop()
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        recognitionTask = nil
        recognitionRequest = nil
        audioEngine = nil
        // Do NOT deactivate the shared audio session — camera + TTS depend on it.
        if state != .idle {
            state = .idle
        }
    }

    private func beginRecordingAsync(speechRecognizer: SFSpeechRecognizer) async {
        do {
            let (engine, request) = try await Task.detached {
                try self.setupAudioEngine()
            }.value

            guard state == .preparing else { return }

            self.audioEngine = engine
            self.recognitionRequest = request
            self.state = .recording

            recognitionTask = speechRecognizer.recognitionTask(with: request) {
                [weak self] result, error in
                Task { @MainActor in
                    guard let self else { return }
                    if let result {
                        self.transcript = self.formatTranscript(
                            result.bestTranscription.formattedString
                        )
                    }
                    if let error {
                        let ns = error as NSError
                        if ns.domain == "kAFAssistantErrorDomain", ns.code == 216 {
                            return
                        }
                        self.errorMessage = error.localizedDescription
                        if self.state == .recording {
                            self.stopRecording()
                        }
                        return
                    }
                    if result?.isFinal == true, self.endsSessionOnFinalResult {
                        self.stopRecording()
                    }
                }
            }
        } catch {
            errorMessage = error.localizedDescription
            state = .idle
        }
    }

    /// The shared AVAudioSession is configured once at app launch (ContentView.configureSharedAudioSession).
    /// This method only creates the engine + tap; it never changes the session category or deactivates it.
    private nonisolated func setupAudioEngine() throws -> (AVAudioEngine, SFSpeechAudioBufferRecognitionRequest) {
        let engine = AVAudioEngine()
        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        request.addsPunctuation = true

        let inputNode = engine.inputNode
        let format = resolvedRecordingFormat(from: inputNode)
        guard format.sampleRate > 0, format.channelCount > 0 else {
            throw RecognizerError.invalidAudioFormat
        }

        inputNode.installTap(onBus: 0, bufferSize: 2048, format: format) { buffer, _ in
            guard buffer.frameLength > 0 else { return }
            request.append(buffer)
        }

        engine.prepare()
        try engine.start()

        return (engine, request)
    }

    private nonisolated func resolvedRecordingFormat(from inputNode: AVAudioInputNode) -> AVAudioFormat {
        let inputFormat = inputNode.inputFormat(forBus: 0)
        if inputFormat.sampleRate > 0, inputFormat.channelCount > 0 {
            return inputFormat
        }
        return inputNode.outputFormat(forBus: 0)
    }

    private nonisolated func formatTranscript(_ text: String) -> String {
        let pattern = #"([.!?;。！？；])\s*"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else {
            return text
        }
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        let formatted = regex.stringByReplacingMatches(
            in: text, options: [], range: range, withTemplate: "$1\n"
        )
        return formatted.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func requestSpeechAuthorization() async -> Bool {
        await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { status in
                continuation.resume(returning: status == .authorized)
            }
        }
    }

    private func requestMicrophoneAuthorization() async -> Bool {
        let status = AVCaptureDevice.authorizationStatus(for: .audio)
        switch status {
        case .authorized:
            return true
        case .notDetermined:
            return await withCheckedContinuation { continuation in
                AVCaptureDevice.requestAccess(for: .audio) { granted in
                    continuation.resume(returning: granted)
                }
            }
        case .denied, .restricted:
            return false
        @unknown default:
            return false
        }
    }
}
