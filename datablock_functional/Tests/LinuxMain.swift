import XCTest

import datablock_functionalTests

var tests = [XCTestCaseEntry]()
tests += datablock_functionalTests.allTests()
XCTMain(tests)
