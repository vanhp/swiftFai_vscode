import XCTest

#if !canImport(ObjectiveC)
public func allTests() -> [XCTestCaseEntry] {
    return [
        testCase(swiftfai_TestTests.allTests),
    ]
}
#endif
