import XCTest
@testable import loadData

final class loadDataTests: XCTestCase {
    func testExample() {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct
        // results.
        XCTAssertEqual(loadData().text, "Hello, World!")
    }

    static var allTests = [
        ("testExample", testExample),
    ]
}
