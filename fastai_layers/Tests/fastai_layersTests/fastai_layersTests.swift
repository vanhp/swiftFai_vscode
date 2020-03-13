import XCTest
@testable import fastai_layers

final class fastai_layersTests: XCTestCase {
    func testExample() {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct
        // results.
        XCTAssertEqual(fastai_layers().text, "Hello, World!")
    }

    static var allTests = [
        ("testExample", testExample),
    ]
}
