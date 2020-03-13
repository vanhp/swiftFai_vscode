import XCTest
@testable import annealing

final class annealingTests: XCTestCase {
    func testExample() {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct
        // results.
        XCTAssertEqual(annealing().text, "Hello, World!")
    }

    static var allTests = [
        ("testExample", testExample),
    ]
}
