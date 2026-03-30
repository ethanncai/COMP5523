//
// Base URL for the Python FastAPI inference server (see smol/server/).
// Replace with your deployment host; no trailing slash.
//

import Foundation

enum ServerConfig {

    /// Example: `http://127.0.0.1:8000` (simulator to Mac), or `http://192.168.x.x:8000` for a LAN server.
    static let baseURLString = "http://127.0.0.1:8000"

    static var baseURL: URL {
        guard let url = URL(string: baseURLString) else {
            fatalError("Invalid ServerConfig.baseURLString: \(baseURLString)")
        }
        return url
    }

    static var healthURL: URL {
        baseURL.appendingPathComponent("health")
    }

    static var inferURL: URL {
        baseURL.appendingPathComponent("infer")
    }
}
