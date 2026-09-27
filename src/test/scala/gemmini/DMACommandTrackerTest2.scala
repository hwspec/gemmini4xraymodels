// Modern chiseltest unit test for DMACommandTracker, replacing the
// pre-existing (and no-longer-compilable, chisel3.iotesters/old-scalatest
// -based) DMACommandTrackerTest.scala -- see ITANH_INVESTIGATION.md for
// why that file is currently excluded from the build (Test/build.sbt in
// chipyard's build.sbt).
//
// DMACommandTracker sits inside LoadController and is the module whose
// `cmds(cmd_id).bytes_left` bookkeeping is what ultimately allows
// LoadController to assert `io.completed` for a load instruction (which
// in turn is what lets ReservationStation release a downstream
// compute/accumulate instruction's dependency on that load -- see
// ITANH_INVESTIGATION.md's ReservationStation section). This is also the
// exact module whose assertion (`bytes_left >= bytes_read`,
// DMACommandTracker.scala:89) was the very first crash found, several
// sessions ago, in this entire ITANH investigation (traced at the time to
// an unrelated `act` field width/masking bug upstream, not a bug in this
// module itself). This test verifies the module's own core contract in
// isolation: a command is marked complete exactly when its declared byte
// count has been fully accounted for by `request_returned` events, no
// earlier and no later, and that multiple concurrently in-flight commands
// (using different cmd_ids) do not cross-contaminate each other's byte
// counts.
package gemmini

import chisel3._
import chisel3.util._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class DMACommandTrackerTest2 extends AnyFlatSpec with ChiselScalatestTester {
  def doAlloc(c: DMACommandTracker[UInt], tag: Int, bytesToRead: Int, maxWait: Int = 1000): Int = {
    c.io.alloc.valid.poke(true.B)
    c.io.alloc.bits.tag.poke(tag.U)
    c.io.alloc.bits.bytes_to_read.poke(bytesToRead.U)

    var waited = 0
    while (!c.io.alloc.ready.peek().litToBoolean && waited < maxWait) { c.clock.step(1); waited += 1 }
    assert(waited < maxWait, s"alloc(tag=$tag, bytes=$bytesToRead) never became ready")
    val cmdId = c.io.alloc.bits.cmd_id.peek().litValue.toInt
    c.clock.step(1)
    c.io.alloc.valid.poke(false.B)
    cmdId
  }

  def sendReturn(c: DMACommandTracker[UInt], cmdId: Int, bytesRead: Int): Unit = {
    c.io.request_returned.valid.poke(true.B)
    c.io.request_returned.bits.cmd_id.poke(cmdId.U)
    c.io.request_returned.bits.bytes_read.poke(bytesRead.U)
    c.clock.step(1)
    c.io.request_returned.valid.poke(false.B)
  }

  behavior of "DMACommandTracker"

  it should "not report completion until all declared bytes have been returned, then report exactly once" in {
    test(new DMACommandTracker(nCmds = 4, maxBytes = 64, tag_t = UInt(8.W))) { c =>
      c.io.cmd_completed.ready.poke(true.B)

      val cmdId = doAlloc(c, tag = 5, bytesToRead = 3)

      sendReturn(c, cmdId, 1)
      assert(!c.io.cmd_completed.valid.peek().litToBoolean, "reported complete after only 1/3 bytes returned")

      sendReturn(c, cmdId, 1)
      assert(!c.io.cmd_completed.valid.peek().litToBoolean, "reported complete after only 2/3 bytes returned")

      sendReturn(c, cmdId, 1)
      assert(c.io.cmd_completed.valid.peek().litToBoolean, "did not report complete after all 3/3 bytes returned")
      assert(c.io.cmd_completed.bits.cmd_id.peek().litValue == cmdId)
      assert(c.io.cmd_completed.bits.tag.peek().litValue == 5)

      c.clock.step(1) // cmd_completed fires and retires this entry
      assert(!c.io.busy.peek().litToBoolean, "tracker still busy after its only command completed and was retired")
    }
  }

  it should "not let two concurrently in-flight commands' byte counts cross-contaminate" in {
    test(new DMACommandTracker(nCmds = 4, maxBytes = 64, tag_t = UInt(8.W))) { c =>
      c.io.cmd_completed.ready.poke(true.B)

      val cmdA = doAlloc(c, tag = 0xAA, bytesToRead = 2)
      val cmdB = doAlloc(c, tag = 0xBB, bytesToRead = 5)
      assert(cmdA != cmdB, "two concurrently allocated commands got the same cmd_id")

      // Return all of B's bytes first; A must remain incomplete.
      sendReturn(c, cmdB, 5)
      assert(!c.io.cmd_completed.valid.peek().litToBoolean ||
        c.io.cmd_completed.bits.cmd_id.peek().litValue != cmdA,
        "command A reported complete after only command B's bytes were returned")

      // B should now be the one reporting complete.
      assert(c.io.cmd_completed.valid.peek().litToBoolean, "command B did not report complete after all its bytes returned")
      assert(c.io.cmd_completed.bits.cmd_id.peek().litValue == cmdB)
      assert(c.io.cmd_completed.bits.tag.peek().litValue == 0xBB)
      c.clock.step(1) // retire B

      // A should still need its 2 bytes; sending 1 must not complete it.
      sendReturn(c, cmdA, 1)
      assert(!c.io.cmd_completed.valid.peek().litToBoolean, "command A reported complete after only 1/2 of its own bytes")

      sendReturn(c, cmdA, 1)
      assert(c.io.cmd_completed.valid.peek().litToBoolean, "command A did not report complete after all 2/2 of its own bytes")
      assert(c.io.cmd_completed.bits.cmd_id.peek().litValue == cmdA)
      assert(c.io.cmd_completed.bits.tag.peek().litValue == 0xAA)
    }
  }

  it should "correctly reuse a cmd_id after it has been retired, with a fresh byte count" in {
    test(new DMACommandTracker(nCmds = 2, maxBytes = 64, tag_t = UInt(8.W))) { c =>
      c.io.cmd_completed.ready.poke(true.B)

      val cmd1 = doAlloc(c, tag = 1, bytesToRead = 1)
      sendReturn(c, cmd1, 1)
      assert(c.io.cmd_completed.valid.peek().litToBoolean)
      c.clock.step(1) // retire

      // Re-allocate; with nCmds=2 and one still-unused slot this might not
      // reuse cmd1's exact id, so allocate a second one too to force reuse
      // pressure, then confirm correctness regardless of which id comes back.
      val cmd2 = doAlloc(c, tag = 2, bytesToRead = 10)
      val cmd3 = doAlloc(c, tag = 3, bytesToRead = 1)

      // Fully satisfy cmd3 (1 byte) -- must not affect cmd2 (needs 10).
      sendReturn(c, cmd3, 1)
      assert(c.io.cmd_completed.valid.peek().litToBoolean)
      assert(c.io.cmd_completed.bits.cmd_id.peek().litValue == cmd3)
      assert(c.io.cmd_completed.bits.tag.peek().litValue == 3)
      c.clock.step(1)

      sendReturn(c, cmd2, 9)
      assert(!c.io.cmd_completed.valid.peek().litToBoolean, "cmd2 reported complete after only 9/10 bytes")
      sendReturn(c, cmd2, 1)
      assert(c.io.cmd_completed.valid.peek().litToBoolean)
      assert(c.io.cmd_completed.bits.cmd_id.peek().litValue == cmd2)
      assert(c.io.cmd_completed.bits.tag.peek().litValue == 2)
    }
  }
}
