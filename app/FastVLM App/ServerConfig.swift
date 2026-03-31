//
// VLM API base URL — edit **here** only (this file).
//
// Path: ``FastVLM App/ServerConfig.swift`` (under repo ``app/``).
// Server implementation: repository root ``server/`` (FastAPI ``/infer``, ``/health``).
//
// Set ``baseURLString`` to your machine, e.g.:
//   - Simulator → Mac: ``http://127.0.0.1:8000``
//   - Device on LAN: ``http://192.168.x.x:8000`` (same port as ``uvicorn``)
// No trailing slash.
//

import Foundation

enum ServerConfig {

    /// API root (scheme + host + port). Change this to point at your ``server`` process.
    static let baseURLString = "http://192.168.99.190:8000"

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
