// SPDX-License-Identifier: MIT
pragma solidity ^0.8.9;

import "forge-std/Test.sol";
import "../../contracts/OpenFLModel.sol"; // Production contract imported for the test harness.

// Tests for the leave/exit flow:
//   - processExits() three branches: remaining == 0, remaining == 1, remaining >= 2
//   - _exitUser mechanics: balance cap, zero-GRS, double-exit, reentry, participants[] holes

// Inline harness that exposes internal state and hooks for testing the leave/exit flow.
contract LeaveExitHarness is OpenFLModel {
    constructor(
        bytes32 _modelHash,
        uint _min_collateral,
        uint _max_collateral,
        uint _reward,
        uint8 _min_rounds,
        uint8 _punishfactor,
        uint8 _punishfactorContrib,
        uint8 _freeriderPenalty
    )
        OpenFLModel(
            _modelHash,
            _min_collateral,
            _max_collateral,
            _reward,
            _min_rounds,
            _punishfactor,
            _punishfactorContrib,
            _freeriderPenalty
        )
    {
        testing = true;
    }

    // Direct state injection — bypass register() so each test has a controllable starting point.
    function _initUser(address user, uint grs) external {
        participants.push(user);
        User storage u = users[user];
        u.addr = user;
        u.isRegistered = true;
        u.globalReputationScore = grs;
        u.nrOfRoundsParticipated = 1;
        nrOfActiveParticipants += 1;
    }

    function _setRewardLeft(uint v) external {
        rewardLeft = v;
    }

    function _setWantsToLeave(address user, bool v) external {
        wantsToLeave[user] = v;
    }

    function _setUserGRS(address user, uint v) external {
        users[user].globalReputationScore = v;
    }

    function _getGRS(address user) external view returns (uint) {
        return users[user].globalReputationScore;
    }

    function _isRegistered(address user) external view returns (bool) {
        return users[user].isRegistered;
    }

    function _participantAt(uint i) external view returns (address) {
        return participants[i];
    }

    function _participantsLength() external view returns (uint) {
        return participants.length;
    }
}

// Reentrancy probe: receive() calls back into exitModel on the model under test.
// Used by testExitModel_reentryIsSafe.
contract ReentrantExiter {
    OpenFLModel public model;
    uint public reentryCount;

    constructor(OpenFLModel _model) {
        model = _model;
    }

    receive() external payable {
        reentryCount += 1;
        // Try to re-enter. Should hit the !isRegistered early-return in _exitUser
        // and no-op (no revert, no double payout).
        model.exitModel();
    }
}

// The test runner
contract LeaveExitTest is Test {
    LeaveExitHarness model;

    address user1;
    address user2;
    address user3;
    address[] users_;

    bytes32 constant MODEL_HASH = bytes32(0);
    uint constant COLLATERAL = 1 ether;
    uint constant REWARD = 1 ether;
    uint8 constant MIN_ROUNDS = 8;
    uint8 constant PUNISH_FACTOR = 3;
    uint8 constant PUNISH_CONTRIB = 3;
    uint8 constant FREERIDER_PENALTY = 50;

    event UserExited(address indexed user, uint grs);


    function setUp() public {
        model = new LeaveExitHarness(
            MODEL_HASH,
            COLLATERAL,
            COLLATERAL,
            REWARD,
            MIN_ROUNDS,
            PUNISH_FACTOR,
            PUNISH_CONTRIB,
            FREERIDER_PENALTY
        );
    }


    // Register three real users via register() — for processExits branch tests
    // that need actual collateral deposited in the contract.
    function _registerThreeUsers() internal {
        user1 = makeAddr("user1");
        user2 = makeAddr("user2");
        user3 = makeAddr("user3");

        vm.deal(user1, COLLATERAL);
        vm.deal(user2, COLLATERAL);
        vm.deal(user3, COLLATERAL);
        vm.prank(user1); model.register{value: COLLATERAL}();
        vm.prank(user2); model.register{value: COLLATERAL}();
        vm.prank(user3); model.register{value: COLLATERAL}();
    }

    function _registerOneUser() internal {
        user1 = makeAddr("user1");

        vm.deal(user1, COLLATERAL);
    }


    // ---------------------------------------------------------------------
    // Part 1: processExits branches  (OpenFLModel.sol:710-765)
    // ---------------------------------------------------------------------

    // Branch: remaining == 0  (OpenFLModel.sol:728)
    // Setup:
    //   - register 3 users with collateral
    //   - flag all three with _setWantsToLeave
    //   - _setRewardLeft(3 ether)
    //   - vm.deal(address(model), enough to cover GRS + reward)
    // Action:
    //   - model.processExits()
    // Assert:
    //   - UserExited reports each user's GRS after the reward share is added
    //   - model.rewardLeft() == 0
    //   - model.nrOfActiveParticipants() == 0
    //   - none of them isRegistered anymore
    //   - 3 UserExited events emitted
    function testProcessExits_totalRegisteredOne_distributesRewardLeftAndExitsEveryone() public {
        _registerOneUser()();

        model._setWantsToLeave(user1, false);

        model._setRewardLeft(3 ether);

        vm.deal(address (model), 3 ether + COLLATERAL * 3); // fund the model to cover all exits

        model.processExits();

        assert(model._getGRS(user1) == 0);

        assert(model.rewardLeft() == 0);
        assert(model.nrOfActiveParticipants() == 0);
        assert(!model._isRegistered(user1));
    }


    function testProcessExits_remainingZero_distributesRewardLeftAndExitsEveryone() public {
        _registerThreeUsers();

        model._setWantsToLeave(user1, true);
        model._setWantsToLeave(user2, true);
        model._setWantsToLeave(user3, true);

        model._setRewardLeft(3 ether);

        vm.deal(address (model), 3 ether + COLLATERAL * 3); // fund the model to cover all exits

        // Default solidity event has: [checkTopic1..3, checkData, emitter].
        // We use only checkTopic1 (user) and checkData (grs), and we don't care about the emitter or other topics.
        // true   check user
        //  false  ignore topic2 because there is none
        //  false  ignore topic3 because there is none
        //  true   check grs
        //  address(model) event must come from model

        // Note: _exitUser() emits the event before it zeroes the user:
        // Assert with 2 ether: COLLATERAL (1 ether) + reward share (1 ether) = 2 ether total payout for user1
        vm.expectEmit(true, false, false, true, address(model));
        emit UserExited(user1, 2 ether);

        vm.expectEmit(true, false, false, true, address(model));
        emit UserExited(user2, 2 ether);

        vm.expectEmit(true, false, false, true, address(model));
        emit UserExited(user3, 2 ether);

        model.processExits();

        assert(model._getGRS(user1) == 0);
        assert(model._getGRS(user2) == 0);
        assert(model._getGRS(user3) == 0);

        assert(model.rewardLeft() == 0);
        assert(model.nrOfActiveParticipants() == 0);
        assert(!model._isRegistered(user1));
        assert(!model._isRegistered(user2));
        assert(!model._isRegistered(user3));
    }


    // Branch: remaining == 1  (OpenFLModel.sol:741)
    // Setup:
    //   - register 3 users; flag a and b only
    //   - _setRewardLeft(3 ether); fund model
    // Action:
    //   - model.processExits()
    // Assert:
    //   - c (survivor) exits with +3 ether added to GRS
    //   - a and b exit at their original GRS (no share)
    //   - rewardLeft == 0
    //   - all three end up not registered, nrOfActiveParticipants == 0
    function testProcessExits_remainingOne_grantsAllRewardToSurvivorThenExitsAll() public {
        _registerThreeUsers();

        model._setWantsToLeave(user1, true);
        model._setWantsToLeave(user2, true);

        model._setRewardLeft(3 ether);

        vm.deal(address (model), 3 ether + COLLATERAL * 3); // fund the model to cover all exits

        // Note: _exitUser() emits the event before it zeroes the user:
        vm.expectEmit(true, false, false, true, address(model));
        emit UserExited(user1, 1 ether);

        vm.expectEmit(true, false, false, true, address(model));
        emit UserExited(user2, 1 ether);

        vm.expectEmit(true, false, false, true, address(model));
        emit UserExited(user3, 4 ether);

        model.processExits();

        assert(model._getGRS(user1) == 0);
        assert(model._getGRS(user2) == 0);
        assert(model._getGRS(user3) == 0);

        assert(model.rewardLeft() == 0);
        assert(model.nrOfActiveParticipants() == 0);
        assert(!model._isRegistered(user1));
        assert(!model._isRegistered(user2));
        assert(!model._isRegistered(user3));
    }


    // Branch: remaining >= 2  (OpenFLModel.sol:757)
    // Setup:
    //   - register 4 users; flag two of them
    //   - _setRewardLeft(2 ether)
    // Action:
    //   - model.processExits()
    // Assert:
    //   - a and b exited (paid their GRS)
    //   - c and d untouched: still registered, GRS unchanged
    //   - rewardLeft == 2 ether (unchanged)
    //   - nrOfActiveParticipants == 2
    //   - participants[0] and participants[1] are address(0); slots for c and d unchanged
    function testProcessExits_remainingMany_exitsOnlyFlaggedNoRedistribution() public {
        _registerThreeUsers();

        // Register a 4th user so remaining == 4 - 2 == 2 (the >= 2 branch).
        address user4 = makeAddr("user4");
        vm.deal(user4, COLLATERAL);
        vm.prank(user4); model.register{value: COLLATERAL}();

        model._setWantsToLeave(user1, true);
        model._setWantsToLeave(user2, true);

        model._setRewardLeft(2 ether);

        // Only user1 and user2 are paid out — no redistribution to survivors.
        vm.expectEmit(true, false, false, true, address(model));
        emit UserExited(user1, 1 ether);

        vm.expectEmit(true, false, false, true, address(model));
        emit UserExited(user2, 1 ether);

        model.processExits();

        // Leavers: cleaned up.
        assert(!model._isRegistered(user1));
        assert(!model._isRegistered(user2));
        assert(model._getGRS(user1) == 0);
        assert(model._getGRS(user2) == 0);

        // Survivors: untouched. Still registered, GRS unchanged, no reward share added.
        assert(model._isRegistered(user3));
        assert(model._isRegistered(user4));
        assert(model._getGRS(user3) == 1 ether);
        assert(model._getGRS(user4) == 1 ether);

        // rewardLeft is NOT consumed in this branch.
        assert(model.rewardLeft() == 2 ether);
        assert(model.nrOfActiveParticipants() == 2);

        // Holes at slots 0 and 1; survivors still in place.
        assert(model.participants(0) == address(0));
        assert(model.participants(1) == address(0));
        assert(model.participants(2) == user3);
        assert(model.participants(3) == user4);
    }
    // ---------------------------------------------------------------------
    // Part 2: _exitUser mechanics
    // ---------------------------------------------------------------------

    // Underfunded cap  (OpenFLModel.sol:672-675)
    // Setup:
    //   - init user a with a normal GRS, but DO NOT fund the contract to cover it
    //     (e.g. _initUser(a, 5 ether) but contract balance = 0 — or fund partially with vm.deal)
    //   - Concretely: vm.deal(address(model), 1 ether); _initUser(a, 5 ether)
    // Action:
    //   - vm.prank(a); model.exitModel()
    // Assert:
    //   - a's ETH balance increased by exactly 1 ether (the contract's balance), NOT 5 ether
    //   - Contract balance is now 0
    //   - UserExited event emits grs = 5 ether (UNCAPPED — documents the discrepancy at line 677)
    //     Use vm.expectEmit + emit UserExited(a, 5 ether) before the call.
    function testExitModel_balanceCappedWhenContractUnderfunded() public {
        user1 = makeAddr("user1");
        model._initUser(user1, 5 ether);   // GRS = 5 ether
        vm.deal(address(model), 1 ether);  // contract only has 1 ether to pay out

        uint balBefore = user1.balance;

        // Event emits the UNCAPPED GRS (documents the discrepancy at OpenFLModel.sol:677).
        vm.expectEmit(true, false, false, true, address(model));
        emit UserExited(user1, 5 ether);

        vm.prank(user1);
        model.exitModel();

        // Payout is capped at the contract's available balance.
        assert(user1.balance - balBefore == 1 ether);
        assert(address(model).balance == 0);

        assert(!model._isRegistered(user1));
        assert(model._getGRS(user1) == 0);
        assert(model.nrOfActiveParticipants() == 0);
    }


    // Zero-GRS  (OpenFLModel.sol:692 — `if (val > 0)` guard)
    // Setup:
    //   - _initUser(a, 0)
    // Action:
    //   - vm.prank(a); model.exitModel()
    // Assert:
    //   - no revert
    //   - a's ETH balance unchanged (no transfer)
    //   - users[a].isRegistered == false
    //   - nrOfActiveParticipants decremented (was 1, now 0)
    //   - UserExited(a, 0) emitted
    function testExitModel_zeroGRSUserExitsCleanly() public {
        user1 = makeAddr("user1");
        model._initUser(user1, 0); // Use init_User so we can pin GRS to 0. Avoid the full register path.

        uint balBefore = user1.balance;

        vm.expectEmit(true, false, false, true, address(model));
        emit UserExited(user1, 0);

        vm.prank(user1);
        model.exitModel();

        assert(user1.balance == balBefore);              // no transfer (val == 0)
        assert(!model._isRegistered(user1));
        assert(model.nrOfActiveParticipants() == 0);
    }


    // Double-exit no-op  (OpenFLModel.sol:670)
    // Setup:
    //   - _initUser(a, 1 ether); vm.deal(address(model), 1 ether)
    //   - vm.prank(a); model.exitModel()    // first exit succeeds
    // Action:
    //   - vm.prank(a); model.exitModel()    // second call — should be no-op
    // Assert:
    //   - Second call does not revert
    //   - a's ETH balance does NOT increase a second time (snapshot before second call)
    //   - nrOfActiveParticipants does not underflow / change again
    //   - No second UserExited event (use vm.recordLogs + vm.getRecordedLogs)
    function testExitModel_doubleExitIsNoOp() public {
        user1 = makeAddr("user1");
        model._initUser(user1, 1 ether);
        vm.deal(address(model), 1 ether);

        vm.prank(user1);
        model.exitModel(); // First exit is expected to succeed.

        uint balAfterFirstExit = user1.balance;
        assert(balAfterFirstExit == 1 ether);
        assert(!model._isRegistered(user1));
        assert(model.nrOfActiveParticipants() == 0);

        // Second exit: hits the !isRegistered early-return at OpenFLModel.sol:670.
        vm.recordLogs();
        vm.prank(user1);
        model.exitModel();

        assert(user1.balance == balAfterFirstExit);          // no second payout
        assert(model.nrOfActiveParticipants() == 0);         // no underflow
        assert(vm.getRecordedLogs().length == 0);            // no second event
    }

    // Reentrancy safety
    // Setup:
    //   - Deploy ReentrantExiter attacker with `new ReentrantExiter(model)`
    //   - _initUser(address(attacker), 1 ether); vm.deal(address(model), 1 ether)
    // Action:
    //   - vm.prank(address(attacker)); model.exitModel()
    //     The contract's call{value: 1 ether} triggers attacker's receive(), which calls exitModel() again.
    //     The reentrant call should hit !isRegistered and no-op.
    // Assert:
    //   - attacker.reentryCount() == 1   (receive() ran once)
    //   - attacker's ETH balance == 1 ether (single payout, no doubling)
    //   - Contract balance == 0
    //   - users[attacker].isRegistered == false
    //   - Exactly 1 UserExited event

    function testExitModel_reentryIsSafe() public {
        ReentrantExiter attacker = new ReentrantExiter(model);

        model._initUser(address(attacker), 1 ether);

        vm.deal(address(model), 1 ether);
        vm.recordLogs();
        vm.prank(address(attacker));

        model.exitModel();

        assert(attacker.reentryCount() == 1);              // receive() ran
        assert(address(attacker).balance == 1 ether);      // single payout
        assert(address(model).balance == 0);
        assert(!model._isRegistered(address(attacker)));
        assert(model.nrOfActiveParticipants() == 0);

        // Exactly one UserExited event — the reentrant call early-returned, no second emit.
        Vm.Log[] memory logs = vm.getRecordedLogs();
        uint exitedCount = 0;
        bytes32 sig = keccak256("UserExited(address,uint256)");
        for (uint i = 0; i < logs.length; i++) {
            if (logs[i].topics.length > 0 && logs[i].topics[0] == sig) exitedCount++;
        }
        assert(exitedCount == 1);
    }

    // Downstream impact of address(0) holes in participants[]
    // Setup:
    //   - _initUser(a, ...); _initUser(b, ...); _initUser(c, ...)
    //   - vm.deal(address(model), enough)
    //   - vm.prank(b); model.exitModel()
    //     → participants[1] is now address(0); participants.length still 3
    // Action / Assert:
    //   - users[address(0)].isRegistered must still be false (was never touched)
    //   - users[address(0)].globalReputationScore must still be 0
    //   - nrOfActiveParticipants == 2
    //   - Iterate over model.participants(i) for i in 0..2 and confirm: a at [0], address(0) at [1], c at [2]
    //
    //   Now drive something that loops over participants. Two options:
    //     (a) flag a (wantsToLeave) and call processExits — assert it cleanly exits a and skips the hole
    //     (b) call settle()-adjacent helpers that iterate participants (e.g. isFeedBackRoundDone)
    //         and assert no revert and no read of users[address(0)] changes its state.
    //   Pick (a) — simpler and directly tests the leave path with a hole present.
    function testParticipantsArrayHolesDontCorruptSubsequentRound() public {
        user1 = makeAddr("user1");
        user2 = makeAddr("user2");
        user3 = makeAddr("user3");

        model._initUser(user1, 1 ether);
        model._initUser(user2, 1 ether);
        model._initUser(user3, 1 ether);

        // Fund the model enough to cover all three exits.
        vm.deal(address(model), 3 ether);

        // user2 leaves first → punches a hole at participants[1].
        vm.prank(user2);
        model.exitModel();

        // The hole is address(0); array length unchanged.
        assert(model._participantsLength() == 3);
        assert(model.participants(0) == user1);
        assert(model.participants(1) == address(0));
        assert(model.participants(2) == user3);

        // address(0) must not have been "touched" by the loop in _exitUser:
        // no fake registration, no fake GRS assigned.
        assert(!model._isRegistered(address(0)));
        assert(model._getGRS(address(0)) == 0);

        assert(model.nrOfActiveParticipants() == 2);

        // Now drive a path that iterates participants: flag user1 + processExits.
        // With one leaver and one survivor (user3), this hits the remaining == 1
        // branch: user1 exits at GRS and user3 absorbs rewardLeft before exit.
        // The key thing: the loop must skip the address(0) hole without reverting
        // or reading users[address(0)] in a way that corrupts state.
        model._setWantsToLeave(user1, true);
        model.processExits();

        assert(!model._isRegistered(user1));
        assert(!model._isRegistered(user3));
        assert(model.nrOfActiveParticipants() == 0);

        // address(0) state still clean after the second iteration.
        assert(!model._isRegistered(address(0)));
        assert(model._getGRS(address(0)) == 0);
    }
}
