import XCTest
@testable import minibatch_training

final class minibatch_trainingTests: XCTestCase {
    func testExample() {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct
        // results.
        XCTAssertEqual(minibatch_training().text, "Hello, World!")
    }

    static var allTests = [
        ("testExample", testExample),
    ]
}
