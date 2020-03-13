import XCTest
@testable import autoDiff_swift

final class autoDiff_swiftTests: XCTestCase {
    func testExample() {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct
        // results.
        XCTAssertEqual(autoDiff_swift().text, "Hello, World!")
    }

    static var allTests = [
        ("testExample", testExample),
    ]
}
