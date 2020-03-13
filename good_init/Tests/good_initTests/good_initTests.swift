import XCTest
@testable import good_init

final class good_initTests: XCTestCase {
    func testExample() {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct
        // results.
        XCTAssertEqual(good_init().text, "Hello, World!")
    }

    static var allTests = [
        ("testExample", testExample),
    ]
}
