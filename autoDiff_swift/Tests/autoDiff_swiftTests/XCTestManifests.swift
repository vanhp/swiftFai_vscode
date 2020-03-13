import XCTest

#if !canImport(ObjectiveC)
public func allTests() -> [XCTestCaseEntry] {
    return [
        testCase(autoDiff_swiftTests.allTests),
    ]
}
#endif
