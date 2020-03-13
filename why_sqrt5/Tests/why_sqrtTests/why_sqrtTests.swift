import XCTest
@testable import why_sqrt

final class why_sqrtTests: XCTestCase {
    func testExample() {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct
        // results.
        XCTAssertEqual(why_sqrt().text, "Hello, World!")
    }

    static var allTests = [
        ("testExample", testExample),
    ]
}
