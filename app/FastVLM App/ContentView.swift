//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2025 Apple Inc. All Rights Reserved.
//

import AVFoundation
import SwiftUI
import Video

extension CVImageBuffer: @unchecked @retroactive Sendable {}
extension CMSampleBuffer: @unchecked @retroactive Sendable {}

let GUIDANCE_INTERVAL = Duration.milliseconds(100)

struct ContentView: View {
    @State private var camera = CameraController()
    @State private var model = RemoteVLMModel()
    @State private var framesToDisplay: AsyncStream<CVImageBuffer>?
    @State private var latestFrame: CVImageBuffer?
    @State private var recognizer = SpeechRecognizer(locale: Locale(identifier: "en-US"))
    @State private var pushToTalkArmed = false
    @State private var hasAudioPermission = false
    @State private var preparationComplete = false
    @State private var speechPlayer = SpeechPlayer()

    /// The object the user asked to pick up (extracted from speech).
    @State private var targetObject: String = ""
    /// Whether the continuous guidance loop is active.
    @State private var guidanceActive = false
    /// Handle for the running guidance loop task.
    @State private var guidanceTask: Task<Void, Never>?

    private let cameraType: CameraType = .continuous

    private var statusTextColor: Color {
        model.evaluationState == .processingPrompt ? .black : .white
    }

    private var statusBackgroundColor: Color {
        switch model.evaluationState {
        case .idle:
            return .gray
        case .generatingResponse:
            return .green
        case .processingPrompt:
            return .yellow
        }
    }

    var body: some View {
        ZStack {
            VStack(spacing: 12) {
                if let framesToDisplay {
                    VideoFrameView(
                        frames: framesToDisplay,
                        cameraType: cameraType,
                        action: nil
                    )
                    .aspectRatio(4 / 3, contentMode: .fit)
                    .overlay(alignment: .bottom) {
                        inferenceCapsule
                            .padding(.bottom, 8)
                    }
                    #if !os(macOS)
                    .overlay(alignment: .topTrailing) {
                        if preparationComplete, !guidanceActive {
                            Button {
                                targetObject = extractObjectName(from: "I want to pick up badge")
                                model.output = ""
                                startGuidanceLoop()
                            } label: {
                                Image(systemName: "ladybug.fill")
                                    .font(.title3)
                                    .padding(10)
                                    .background(.ultraThinMaterial)
                                    .clipShape(Circle())
                            }
                            .padding(8)
                        }
                    }
                    #endif
                } else {
                    ProgressView()
                        .controlSize(.large)
                        .frame(maxWidth: .infinity)
                        .aspectRatio(4 / 3, contentMode: .fit)
                }

                dialogueSection

                if let err = recognizer.errorMessage, !err.isEmpty {
                    Text(err)
                        .font(.footnote)
                        .foregroundStyle(.red)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }

                talkButton
            }
            .padding()

            if !preparationComplete {
                ZStack {
                    Color.black.opacity(0.58)
                        .ignoresSafeArea(.all)
                    VStack(spacing: 12) {
                        ProgressView()
                            .controlSize(.regular)
                            .tint(.white)
                        Text("Preparing…")
                            .font(.subheadline.weight(.medium))
                            .foregroundStyle(.white)
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .allowsHitTesting(true)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .task {
            configureSharedAudioSession()
            camera.start()
        }
        .task {
            preparationComplete = false
            hasAudioPermission = await recognizer.requestAuthorization()
            if !hasAudioPermission {
                recognizer.errorMessage =
                    "Microphone or speech recognition permission is not granted."
            }
            let loaded = await model.load()
            if loaded {
                await model.warmup()
            }
            if hasAudioPermission {
                await recognizer.warmup()
            }
            preparationComplete = true
        }
        .task {
            await distributeVideoFrames()
        }
        #if !os(macOS)
        .onAppear {
            UIApplication.shared.isIdleTimerDisabled = true
        }
        .onDisappear {
            UIApplication.shared.isIdleTimerDisabled = false
        }
        #endif
    }

    // MARK: - Sub-views

    private var inferenceCapsule: some View {
        Group {
            if guidanceActive {
                HStack(spacing: 6) {
                    Image(systemName: "hand.point.up.left.fill")
                        .font(.caption)
                    Text("Guiding: \(targetObject)")
                }
            } else if model.evaluationState == .processingPrompt {
                HStack {
                    ProgressView()
                        .tint(statusTextColor)
                        .controlSize(.small)
                    Text(model.evaluationState.rawValue)
                }
            } else if model.evaluationState == .idle {
                HStack(spacing: 6) {
                    Image(systemName: "clock.fill")
                        .font(.caption)
                    Text(model.evaluationState.rawValue)
                }
            } else {
                HStack(spacing: 6) {
                    Image(systemName: "lightbulb.fill")
                        .font(.caption)
                    Text(model.evaluationState.rawValue)
                }
            }
        }
        .foregroundStyle(guidanceActive ? .white : statusTextColor)
        .font(.caption)
        .bold()
        .padding(.vertical, 6)
        .padding(.horizontal, 8)
        .background(guidanceActive ? .orange : statusBackgroundColor)
        .clipShape(.capsule)
    }

    private var dialogueSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Target")
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(targetObject.isEmpty ? "Say: \"I want to pick up [item]\"" : targetObject)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(8)
                .background(Color.secondary.opacity(0.12))
                .clipShape(RoundedRectangle(cornerRadius: 10))

            Text("Guidance")
                .font(.caption)
                .foregroundStyle(.secondary)
            ScrollView {
                if model.output.isEmpty, model.running {
                    ProgressView()
                        .controlSize(.regular)
                        .frame(maxWidth: .infinity)
                } else {
                    Text(model.output.isEmpty ? " " : model.output)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                }
            }
            .frame(minHeight: 56, maxHeight: 120)
            .padding(8)
            .background(Color.secondary.opacity(0.12))
            .clipShape(RoundedRectangle(cornerRadius: 10))
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var talkButton: some View {
        Group {
            if guidanceActive {
                Text("Tap to stop guidance")
                    .font(.subheadline.weight(.semibold))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
                    .background(Color.orange)
                    .foregroundStyle(.white)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                    .contentShape(Rectangle())
                    .onTapGesture {
                        stopGuidance()
                    }
            } else {
                Text(recognizer.isRecording ? "Release to start" : "Hold to speak")
                    .font(.subheadline.weight(.semibold))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
                    .background(recognizer.isRecording ? Color.red.opacity(0.9) : Color.accentColor)
                    .foregroundStyle(.white)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                    .contentShape(Rectangle())
                    .opacity(
                        preparationComplete
                            && hasAudioPermission
                            && recognizer.state != .preparing
                            && recognizer.state != .finalizing
                            ? 1
                            : 0.45
                    )
                    .gesture(
                        DragGesture(minimumDistance: 0)
                            .onChanged { _ in
                                guard hasAudioPermission else { return }
                                guard recognizer.state != .preparing,
                                      recognizer.state != .finalizing else { return }
                                if !pushToTalkArmed {
                                    pushToTalkArmed = true
                                    speechPlayer.stop()
                                    do {
                                        try recognizer.startRecording()
                                    } catch {
                                        recognizer.errorMessage = error.localizedDescription
                                        pushToTalkArmed = false
                                    }
                                }
                            }
                            .onEnded { _ in
                                guard pushToTalkArmed else { return }
                                pushToTalkArmed = false
                                recognizer.stopRecording()
                                Task { await handleUserCommand() }
                            }
                    )
                    .disabled(
                        !preparationComplete
                            || !hasAudioPermission
                            || recognizer.state == .preparing
                            || recognizer.state == .finalizing
                    )
            }
        }
    }

    // MARK: - Guidance logic

    /// Extracts the target object from user speech and starts the continuous guidance loop.
    private func handleUserCommand() async {
        try? await Task.sleep(for: .milliseconds(500))
        let spoken = recognizer.transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !spoken.isEmpty else { return }

        targetObject = extractObjectName(from: spoken)
        model.output = ""
        startGuidanceLoop()
    }

    /// Strips common prefixes like "I want to pick up (the) ..." to get just the object name.
    private func extractObjectName(from sentence: String) -> String {
        let lowered = sentence.lowercased()
        let prefixes = [
            "i want to pick up the ",
            "i want to pick up ",
            "i want to grab the ",
            "i want to grab ",
            "i want to get the ",
            "i want to get ",
            "pick up the ",
            "pick up ",
            "grab the ",
            "grab ",
            "get the ",
            "get ",
            "help me pick up the ",
            "help me pick up ",
            "help me grab the ",
            "help me grab ",
            "help me get the ",
            "help me get ",
            "please pick up the ",
            "please pick up ",
            "please grab the ",
            "please grab ",
        ]
        for prefix in prefixes {
            if lowered.hasPrefix(prefix) {
                let remainder = String(sentence.dropFirst(prefix.count))
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                    .trimmingCharacters(in: .punctuationCharacters)
                if !remainder.isEmpty {
                    return remainder
                }
            }
        }
        return sentence.trimmingCharacters(in: .punctuationCharacters)
    }

    private func startGuidanceLoop() {
        guidanceTask?.cancel()
        guidanceActive = true

        guidanceTask = Task {
            while !Task.isCancelled {
                guard let frame = latestFrame else {
                    try? await Task.sleep(for: .milliseconds(200))
                    continue
                }

                let prompt = buildGuidancePrompt(for: targetObject)

                model.cancel()
                await model.generate(prompt: prompt, pixelBuffer: frame)

                if Task.isCancelled { break }

                let reply = model.output.trimmingCharacters(in: .whitespacesAndNewlines)
                if !reply.isEmpty {
                    speechPlayer.speak(reply)
                    await speechPlayer.waitUntilDone()
                }

                if Task.isCancelled { break }
            }

            await MainActor.run {
                guidanceActive = false
            }
        }
    }

    private func stopGuidance() {
        guidanceTask?.cancel()
        guidanceTask = nil
        guidanceActive = false
        model.cancel()
        speechPlayer.stop()
    }

    private func buildGuidancePrompt(for object: String) -> String {
        "System: You guide a blind person's hand to grab \"\(object)\".\n"
            + "Rules:\n"
            + "1. Find the \"\(object)\" in the image.\n"
            + "2. Find the user's hand in the image.\n"
            + "3. Compare their positions and output ONE directional command.\n"
            + "4. Allowed commands ONLY: \"move left\", \"move right\", \"move up\", \"move down\", "
            + "\"move closer\", \"move back\", \"almost there\", \"open hand\", \"close hand\", \"grab now\".\n"
            + "5. Output ONLY the command. No other words. No explanation. Maximum 3 words.\n"
            + "6. If you cannot see the hand, say \"show your hand\".\n"
            + "7. If you cannot see the object, say \"object not found\".\n\n"
            + "User: Guide my hand."
    }

    // MARK: - Video distribution

    private func distributeVideoFrames() async {
        if Task.isCancelled { return }

        let frames = AsyncStream<CMSampleBuffer>(bufferingPolicy: .bufferingNewest(1)) {
            camera.attach(continuation: $0)
        }

        let (stream, continuation) = AsyncStream.makeStream(
            of: CVImageBuffer.self,
            bufferingPolicy: .bufferingNewest(1)
        )
        framesToDisplay = stream

        for await sampleBuffer in frames {
            if Task.isCancelled { break }
            if let frame = sampleBuffer.imageBuffer {
                continuation.yield(frame)
                latestFrame = frame
            }
        }

        await MainActor.run {
            framesToDisplay = nil
            camera.detatch()
        }
        continuation.finish()
    }

    private func configureSharedAudioSession() {
        #if os(iOS)
        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(
                .playAndRecord,
                mode: .spokenAudio,
                options: [.defaultToSpeaker, .allowBluetooth, .mixWithOthers]
            )
            try session.setActive(true, options: [])
        } catch {
            // Best-effort; camera will still work without mic.
        }
        #endif
    }
}

#Preview {
    ContentView()
}
