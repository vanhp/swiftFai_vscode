import XCTest
@testable import fully_connected

final class fully_connectedTests: XCTestCase {
    func testExample() {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct
        // results.
        XCTAssertEqual(fully_connected().text, "Hello, World!")
    }

    static var allTests = [
        ("testExample", testExample),
    ]
}
