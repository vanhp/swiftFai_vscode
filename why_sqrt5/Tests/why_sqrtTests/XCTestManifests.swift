import XCTest

#if !canImport(ObjectiveC)
public func allTests() -> [XCTestCaseEntry] {
    return [
        testCase(why_sqrtTests.allTests),
    ]
}
#endif
