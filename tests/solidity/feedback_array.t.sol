// SPDX-License-Identifier: MIT
pragma solidity ^0.8.9;

import "forge-std/Test.sol";
import "../../contracts/OpenFLModel.sol";

// Unused ModelMock - Not sure if this is intentional

///// @notice A mock version of OpenFLModel for gas testing the fallback only
//contract OpenFLModelMock is OpenFLModel {
//    constructor(
//        bytes32 modelHash,
//        uint minCollateral,
//        uint maxCollateral,
//        uint reward,
//        uint8 minRounds,
//        uint8 punishFactor,
//        uint8 punishFactorContrib,
//        uint8 freeriderPenalty
//    ) OpenFLModel(modelHash, minCollateral, maxCollateral, reward, minRounds, punishFactor, punishFactorContrib, freeriderPenalty) {}
//
//    /// @notice Override feedback to skip all logic/modifiers
//    function feedback(address, int) public override {
//        // do nothing
//    }
//}

contract FallbackGasTest is Test {
    OpenFLModel model;

    address[] users;
    int256[] scores;

    uint256 constant N = 6; // change this to test different batch sizes


    function setUp() public {
        // Deploy a dummy model — constructor args do not matter for gas measurement
        model = new OpenFLModel(
            bytes32("testhash"),
            uint(1e18),
            uint(1.8e18),
            uint(1e18),
            3,
            3,
            3,
            50
        );

        model.setTesting(true);

        users = new address[](N);
        scores = new int256[](N);

        for (uint256 i = 0; i < N; i++) {
            users[i] = address(uint160(i + 1));
            scores[i] = int256(i);
        }
    }

    function testFallbackGas() public {
        bytes memory data = buildPacked(users, scores);

        // call fallback
        (bool ok,) = address(model).call(data);
        require(ok, "fallback call failed");
    }

    function testSubmitFeedbackBytesGas() public {
        bytes memory data = buildPacked(users, scores);

        // call the new function directly
        model.submitFeedbackBytes(data);
    }

    function testFallbackPackedFeedbackUpdatesTargetReputation() public {
        address voter = makeAddr("fallbackVoter");
        address target = makeAddr("fallbackTarget");
        _registerFeedbackPair(voter, target);

        address[] memory targets = new address[](1);
        targets[0] = target;
        int256[] memory votes = new int256[](1);
        votes[0] = 1;

        bytes memory data = buildPacked(targets, votes);
        vm.prank(voter);
        (bool ok,) = address(model).call(data);

        assertTrue(ok);
        (, , , int targetRoundReputation, , , , , , ) = model.getUser(target);
        assertEq(targetRoundReputation, int256(1 ether));
    }

    function testSubmitFeedbackBytesPackedFeedbackUpdatesTargetReputation() public {
        address voter = makeAddr("bytesVoter");
        address target = makeAddr("bytesTarget");
        _registerFeedbackPair(voter, target);

        address[] memory targets = new address[](1);
        targets[0] = target;
        int256[] memory votes = new int256[](1);
        votes[0] = -1;

        vm.prank(voter);
        model.submitFeedbackBytes(buildPacked(targets, votes));

        (, , , int targetRoundReputation, , , , , , ) = model.getUser(target);
        assertEq(targetRoundReputation, -int256(1 ether));
    }

    function _registerFeedbackPair(address voter, address target) internal {
        model.setTesting(false);

        vm.deal(voter, 1 ether);
        vm.deal(target, 1 ether);
        vm.prank(voter);
        model.register{value: 1 ether}();
        vm.prank(target);
        model.register{value: 1 ether}();

        vm.warp(block.timestamp + 86401);
        vm.roll(block.number + 1);
    }

    // -------------------------------------------------------------
    // Helper: Encode calldata exactly as your fallback expects
    // -------------------------------------------------------------
    function buildPacked(address[] memory a, int256[] memory v)
        internal pure
        returns (bytes memory out)
    {
        require(a.length == v.length, "length mismatch");

        uint256 n = a.length;
        uint256 total = n * (20 + 32);

        out = new bytes(total);
        uint256 offset = 0;

        // Write 20-byte addresses
        for (uint256 i = 0; i < n; i++) {
            uint256 addr = uint256(uint160(a[i]));

            assembly {
                // store left-padded address → only last 20 bytes matter
                mstore(add(add(out, 0x20), offset), shl(96, addr))
            }
            offset += 20;
        }

        // Write 32-byte ints
        for (uint256 i = 0; i < n; i++) {
            int256 val = v[i];

            assembly {
                mstore(add(add(out, 0x20), offset), val)
            }
            offset += 32;
        }
    }
}
