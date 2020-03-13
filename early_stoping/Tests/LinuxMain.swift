import XCTest

import early_stopingTests

var tests = [XCTestCaseEntry]()
tests += early_stopingTests.allTests()
XCTMain(tests)
