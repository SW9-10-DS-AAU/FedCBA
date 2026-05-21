// SPDX-License-Identifier: MIT
pragma solidity ^0.8.9;

import "forge-std/Test.sol";
import "../../contracts/OpenFLModel.sol";

// End-to-end exit flow against a plain OpenFLModel (no state-injection harness).
// Drives: register → round 1 (feedback + contribution scores + settle) → mark/exit →
// post-exit round with the remaining users → final exitModel for everyone.

contract LeaveExitIntegrationTest is Test {
    OpenFLModel model;

    address a = makeAddr("a");
    address b = makeAddr("b");
    address c = makeAddr("c");

    bytes32 constant MODEL_HASH = bytes32(0);
    uint constant COLLATERAL = 1 ether;
    uint constant REWARD = 1 ether;
    uint8 constant MIN_ROUNDS = 8;
    uint8 constant PUNISH_FACTOR = 3;
    uint8 constant PUNISH_CONTRIB = 3;
    uint8 constant FREERIDER_PENALTY = 50;

    event UserExited(address indexed user, uint grs);

    function setUp() public {
        // Fund the reward accounting so final exit assertions can check exact payouts.
        vm.deal(address(this), REWARD);
        model = new OpenFLModel{value: REWARD}(
            MODEL_HASH,
            COLLATERAL,
            COLLATERAL,
            REWARD,
            MIN_ROUNDS,
            PUNISH_FACTOR,
            PUNISH_CONTRIB,
            FREERIDER_PENALTY
        );

        vm.deal(a, 10 ether);
        vm.deal(b, 10 ether);
        vm.deal(c, 10 ether);
    }

    // ---------------------------------------------------------------------
    // Helpers — keep test bodies readable. Fill in as you go.
    // ---------------------------------------------------------------------

    function _register(address who) internal {
        vm.prank(who);
        model.register{value: COLLATERAL}();
    }

    // Advance time past the 1-day feedback gate (OpenFLModel.sol:112)
    // so that feedback() doesn't require all hashed weights to be submitted.
    function _openFeedbackRound() internal {
        vm.warp(block.timestamp + 86401);
        vm.roll(block.number + 1);
    }

    // Each registered user submits feedback (score = 1) for every other registered user.
    // `users` should be the *currently-active* list (skip any address(0) holes).
    function _everyoneVotesPositiveOnEveryoneElse(address[] memory users) internal {
        for (uint i = 0; i < users.length; i++) {
            for (uint j = 0; j < users.length; j++) {
                if (i == j || users[i] == address(0) || users[j] == address(0)) continue;
                vm.prank(users[i]);
                model.feedback(users[j], 1);
            }
        }
    }

    // Each active user submits a positive contribution score and an evaluation score of 1e18.
    function _submitContribAndEvalForAll(address[] memory users, int256 contrib, uint256 eval_) internal {
        for (uint i = 0; i < users.length; i++) {
            if (users[i] == address(0)) continue;
            vm.prank(users[i]);
            model.submitContributionScoreAndVotingEvaluation(contrib, eval_);
        }
    }

    // Read user fields from the model's public users mapping.
    // OpenFLModel.User struct (0-indexed):
    //   0: weightedContribScore (int256)
    //   1: globalReputationScore (uint)
    //   2: roundReputation (int256)
    //   3: addr (address)
    //   4: nrOfRoundsParticipated (uint8)
    //   5: nrOfVotesFromUser (uint8)
    //   6: isPunished (bool)
    //   7: isRegistered (bool)
    //   8: whitelistedForRewards (bool)
    //   9: isDisqualified (bool)
    //   10: isPassivePunished (bool)

    function _isRegistered(address user) internal view returns (bool) {
        (,,,,,,, bool reg,,,) = model.users(user);
        return reg;
    }

    function _grs(address user) internal view returns (uint) {
        (, uint globalReputationScore,,,,,,,,,) = model.users(user);
        return globalReputationScore;
    }

    // ---------------------------------------------------------------------
    // End-to-end test
    // ---------------------------------------------------------------------

    // Full flow:
    //   1) a, b, c register
    //   2) Round 0 runs: feedback → contrib/eval → settle. Assert all still active.
    //   3) b marks wantsToLeave; processExits.
    //      Assert: UserExited(b, b's GRS) emitted, b's ETH balance up by that GRS,
    //              nrOfActiveParticipants == 2, participants[1] == address(0).
    //   4) Round N runs with just a and c. Confirm settle completes and GRS changes.
    //      (Be careful: feedback / votes / contribution counts must match the post-exit
    //       active count of 2, not 3. The helpers above already skip address(0).)
    //   5) a and c each call exitModel.
    //      Assert both paid out; contract is drained (or near it — rounding/punishment may leave dust).
    function testFullRoundWithExit_endToEnd() public {
        // 1) register
        _register(a);
        _register(b);
        _register(c);

        assert(model.nrOfActiveParticipants() == 3);

        // 2) round 0
        _openFeedbackRound();
        address[] memory all = new address[](3);
        all[0] = a; all[1] = b; all[2] = c;
        _everyoneVotesPositiveOnEveryoneElse(all);
        _submitContribAndEvalForAll(all, int256(1e18), uint256(1e18));
        model.settle();

        assert(_isRegistered(a));
        assert(_isRegistered(b));
        assert(_isRegistered(c));

        // 3) b leaves
        vm.prank(b);
        model.markWantsToLeave();

        uint bBalanceBefore = b.balance;
        uint bGrsBefore = _grs(b);

        vm.expectEmit(true, false, false, true, address(model));
        emit UserExited(b, bGrsBefore);
        model.processExits();

        assertEq(b.balance - bBalanceBefore, bGrsBefore);
        assertEq(model.nrOfActiveParticipants(), 2);
        assertEq(model.participants(1), address(0));
        assertFalse(_isRegistered(b));
        assertFalse(model.wantsToLeave(b));

        // 4) Another round with a and c only
        uint aGrsBeforeSecondRound = _grs(a);
        uint cGrsBeforeSecondRound = _grs(c);

        _openFeedbackRound();
        address[] memory remaining = new address[](3);
        remaining[0] = a; remaining[1] = address(0); remaining[2] = c;
        _everyoneVotesPositiveOnEveryoneElse(remaining);
        _submitContribAndEvalForAll(remaining, int256(1e18), uint256(1e18));
        model.settle();

        assertTrue(_isRegistered(a));
        assertTrue(_isRegistered(c));
        assertGt(_grs(a), aGrsBeforeSecondRound);
        assertGt(_grs(c), cGrsBeforeSecondRound);

        // 5) Final exits
        uint aBalanceBefore = a.balance;
        uint cBalanceBefore = c.balance;
        uint aGrsBeforeExit = _grs(a);
        uint cGrsBeforeExit = _grs(c);

        vm.prank(a);
        model.exitModel();
        vm.prank(c);
        model.exitModel();

        assertEq(a.balance - aBalanceBefore, aGrsBeforeExit);
        assertEq(c.balance - cBalanceBefore, cGrsBeforeExit);
        assertEq(model.nrOfActiveParticipants(), 0);
        assertFalse(_isRegistered(a));
        assertFalse(_isRegistered(c));
    }
}
