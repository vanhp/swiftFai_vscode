import XCTest
@testable import matmul

final class matmulTests: XCTestCase {
    func testExample() {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct
        // results.
        XCTAssertEqual(matmul().text, "Hello, World!")
    }

    static var allTests = [
        ("testExample", testExample),
    ]
}
