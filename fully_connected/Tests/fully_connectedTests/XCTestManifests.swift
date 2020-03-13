import XCTest

#if !canImport(ObjectiveC)
public func allTests() -> [XCTestCaseEntry] {
    return [
        testCase(fully_connectedTests.allTests),
    ]
}
#endif
