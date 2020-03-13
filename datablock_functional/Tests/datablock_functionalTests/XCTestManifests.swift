import XCTest

#if !canImport(ObjectiveC)
public func allTests() -> [XCTestCaseEntry] {
    return [
        testCase(datablock_functionalTests.allTests),
    ]
}
#endif
