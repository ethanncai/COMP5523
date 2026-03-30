//
// Client for smol/server FastAPI: POST /infer (multipart image + prompt).
//

import CoreImage
import CoreVideo
import Foundation
import ImageIO
import UniformTypeIdentifiers

@Observable
@MainActor
final class RemoteVLMModel {

    public var running = false
    public var modelInfo = ""
    public var output = ""
    public var promptTime = ""

    enum EvaluationState: String, CaseIterable {
        case idle = "Idle"
        case processingPrompt = "Processing Prompt"
        case generatingResponse = "Generating Response"
    }

    public var evaluationState = EvaluationState.idle

    /// Sent as ``max_new_tokens`` to the server (matches previous on-device cap).
    public var maxNewTokens = 240

    private let session: URLSession

    public init() {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 300
        config.timeoutIntervalForResource = 600
        session = URLSession(configuration: config)
    }

    /// Verifies ``ServerConfig`` and that ``GET /health`` succeeds.
    @discardableResult
    public func load() async -> Bool {
        do {
            let (data, response) = try await session.data(from: ServerConfig.healthURL)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                modelInfo = "Server health check failed"
                return false
            }
            if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let status = json["status"] as? String, status == "ok"
            {
                modelInfo = "Connected to server"
                return true
            }
            modelInfo = "Connected (unexpected health payload)"
            return true
        } catch {
            modelInfo = "Cannot reach server: \(error.localizedDescription)"
            return false
        }
    }

    /// No local model; kept for API compatibility with ``ContentView``.
    public func warmup() async {}

    /// POST ``/infer`` with JPEG frame + prompt; updates ``output`` with JSON ``text``.
    public func generate(prompt: String, pixelBuffer: CVPixelBuffer) async {
        guard !running else { return }

        running = true
        evaluationState = .processingPrompt
        let start = Date()

        defer {
            running = false
            evaluationState = .idle
        }

        guard let jpegData = Self.jpegData(from: pixelBuffer) else {
            output = "Failed: could not encode image"
            return
        }

        do {
            try Task.checkCancellation()
        } catch {
            return
        }

        evaluationState = .generatingResponse
        output = ""

        do {
            let text = try await Self.postInfer(
                session: session,
                prompt: prompt,
                imageData: jpegData,
                filename: "frame.jpg",
                maxNewTokens: maxNewTokens
            )
            try Task.checkCancellation()
            let ms = Int(Date().timeIntervalSince(start) * 1000)
            promptTime = "\(ms) ms"
            output = text
        } catch is CancellationError {
            output = ""
        } catch {
            output = "Failed: \(error.localizedDescription)"
        }
    }

    public func cancel() {
        running = false
        output = ""
        promptTime = ""
        evaluationState = .idle
    }

    // MARK: - Encoding

    private static func jpegData(from pixelBuffer: CVPixelBuffer, quality: CGFloat = 0.88) -> Data? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext(options: [.useSoftwareRenderer: false])
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
            return nil
        }
        let data = NSMutableData()
        let type = UTType.jpeg.identifier as CFString
        guard let dest = CGImageDestinationCreateWithData(data, type, 1, nil) else {
            return nil
        }
        let props: [CFString: Any] = [
            kCGImageDestinationLossyCompressionQuality: quality,
        ]
        CGImageDestinationAddImage(dest, cgImage, props as CFDictionary)
        guard CGImageDestinationFinalize(dest) else {
            return nil
        }
        return data as Data
    }

    // MARK: - HTTP

    private static func postInfer(
        session: URLSession,
        prompt: String,
        imageData: Data,
        filename: String,
        maxNewTokens: Int
    ) async throws -> String {
        var request = URLRequest(url: ServerConfig.inferURL)
        request.httpMethod = "POST"

        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue(
            "multipart/form-data; boundary=\(boundary)",
            forHTTPHeaderField: "Content-Type"
        )

        var body = Data()
        func appendPart(name: String, value: String) {
            body.append("--\(boundary)\r\n".data(using: .utf8)!)
            body.append(
                "Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n".data(using: .utf8)!)
            body.append(value.data(using: .utf8)!)
            body.append("\r\n".data(using: .utf8)!)
        }

        appendPart(name: "prompt", value: prompt)
        appendPart(name: "max_new_tokens", value: String(maxNewTokens))

        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append(
            "Content-Disposition: form-data; name=\"image\"; filename=\"\(filename)\"\r\n"
                .data(using: .utf8)!)
        body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
        body.append(imageData)
        body.append("\r\n".data(using: .utf8)!)
        body.append("--\(boundary)--\r\n".data(using: .utf8)!)

        request.httpBody = body

        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw URLError(.badServerResponse)
        }
        guard (200 ... 299).contains(http.statusCode) else {
            let msg = String(data: data, encoding: .utf8) ?? "HTTP \(http.statusCode)"
            throw NSError(
                domain: "RemoteVLMModel", code: http.statusCode,
                userInfo: [NSLocalizedDescriptionKey: msg])
        }

        let decoded = try JSONDecoder().decode(InferJSONResponse.self, from: data)
        return decoded.text
    }

    private struct InferJSONResponse: Decodable {
        let text: String
        let raw: String?
    }
}
