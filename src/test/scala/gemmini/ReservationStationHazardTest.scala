// Chiseltest unit test checking whether ReservationStation correctly
// blocks a PRELOAD (the start of a compute-and-accumulate operation)
// from being issued until a prior LOAD (a bias mvin) TARGETING THE SAME
// ACCUMULATOR ADDRESS has genuinely COMPLETED -- not merely been issued.
//
// Context (see ITANH_INVESTIGATION.md): two earlier chiseltest checks
// found no bug in AccumulatorMem's own RAW-hazard handling, nor in
// LoopConvLdBias/LoopConvExecute's address-computation formulas. Reading
// ExecuteController.scala directly found that its accumulator write path
// is NOT actually backpressure-aware:
//   io.acc.write(i).valid := start_array_outputting && w_bank === i.U && ...
//   assert(!(io.acc.write(i).valid && !io.acc.write(i).ready),
//     "Execute controller write to AccumulatorMem was skipped")
// i.e. `valid` is asserted purely from the systolic mesh's own output
// timing, with NO dependency on `.ready` at all -- the design simply
// ASSUMES `ready` will always be true and asserts otherwise, rather than
// holding/retrying. If AccumulatorMem's own hazard-guard (confirmed
// real and correct by the first chiseltest test) ever drove
// `bio.write.ready` low at exactly the moment ExecuteController tries to
// write, the write would be silently dropped (valid && !ready never
// fires in hardware) with no compensating retry -- ExecuteController has
// no way to know or recover.
//
// The only thing that could prevent AccumulatorMem's guard from ever
// triggering in the first place is ReservationStation's OWN dependency
// tracking: a PRELOAD (which starts a compute-and-accumulate targeting
// address X) must not be *issued* to ExecuteController until any prior
// LOAD targeting the same address X has *completed* (not merely been
// issued to LoadController) -- giving AccumulatorMem's internal
// pipelined-write delay line (acc_latency cycles) time to actually
// commit the bias before anything tries to accumulate on top of it.
// This test checks that contract directly, at the ReservationStation
// level, by allocating a LOAD3_CMD (bias) and a PRELOAD_CMD targeting
// the SAME accumulator row and checking exactly when the PRELOAD becomes
// issuable.
package gemmini

import chisel3._
import chisel3.util._
import chiseltest._
import chiseltest.simulator.VerilatorBackendAnnotation
import org.scalatest.flatspec.AnyFlatSpec
import org.chipsalliance.cde.config.{Config, Parameters}
import freechips.rocketchip.tile.{TileKey, RocketTileParams}
import Arithmetic.SIntArithmetic._
import GemminiISA._

// Small helper: encodes a raw accumulator row number into the bit-packed
// LocalAddr format real RoCC commands use (reusing LocalAddr.cast_to_acc_addr
// directly, rather than hand-replicating its bit layout in Scala).
class AccAddrEncoder(local_addr_t: LocalAddr) extends Module {
  val io = IO(new Bundle {
    val row = Input(UInt(32.W))
    val accumulate = Input(Bool())
    val readFull = Input(Bool())
    val encoded = Output(UInt(32.W))
  })
  io.encoded := LocalAddr.cast_to_acc_addr(local_addr_t, io.row, io.accumulate, io.readFull).asUInt
}

class ReservationStationHazardHarness(implicit p: Parameters) extends Module {
  val config = GemminiConfigs.defaultConfig
  val cmd_t = new GemminiCmd(config.reservation_station_entries)

  val rs = Module(new ReservationStation(config, cmd_t))
  val encoder = Module(new AccAddrEncoder(config.local_addr_t))

  rs.io.issue.st.ready := false.B // never issuing store instructions in this test
  rs.io.counter.external_reset := false.B

  val io = IO(new Bundle {
    val alloc = Flipped(Decoupled(cmd_t.cloneType))
    val completed = Flipped(Valid(UInt(GemminiConfigs.defaultConfig.ROB_ID_WIDTH.W)))

    val issueLdValid = Output(Bool())
    val issueLdReady = Input(Bool())
    val issueLdRobId = Output(UInt(GemminiConfigs.defaultConfig.ROB_ID_WIDTH.W))

    val issueExValid = Output(Bool())
    val issueExReady = Input(Bool())
    val issueExRobId = Output(UInt(GemminiConfigs.defaultConfig.ROB_ID_WIDTH.W))

    val encRow = Input(UInt(32.W))
    val encAcc = Input(Bool())
    val encFull = Input(Bool())
    val encOut = Output(UInt(32.W))
  })

  io.alloc <> rs.io.alloc
  rs.io.completed <> io.completed

  io.issueLdValid := rs.io.issue.ld.valid
  rs.io.issue.ld.ready := io.issueLdReady
  io.issueLdRobId := rs.io.issue.ld.rob_id

  io.issueExValid := rs.io.issue.ex.valid
  rs.io.issue.ex.ready := io.issueExReady
  io.issueExRobId := rs.io.issue.ex.rob_id

  encoder.io.row := io.encRow
  encoder.io.accumulate := io.encAcc
  encoder.io.readFull := io.encFull
  io.encOut := encoder.io.encoded
}

class ReservationStationHazardTest extends AnyFlatSpec with ChiselScalatestTester {
  implicit val p: Parameters = new Config((_, _, _) => {
    case TileKey => RocketTileParams()
  })

  val config = GemminiConfigs.defaultConfig
  val block_rows = config.tileRows * config.meshRows
  val rowsFieldWidth = log2Up(block_rows + 1)

  def doAlloc(c: ReservationStationHazardHarness, funct: BigInt, rs1: BigInt, rs2: BigInt, maxWait: Int = 1000): Unit = {
    c.io.alloc.bits.cmd.inst.funct.poke(funct.U)
    c.io.alloc.bits.cmd.rs1.poke(rs1.U)
    c.io.alloc.bits.cmd.rs2.poke(rs2.U)
    c.io.alloc.bits.rob_id.valid.poke(false.B)
    c.io.alloc.bits.from_matmul_fsm.poke(false.B)
    c.io.alloc.bits.from_conv_fsm.poke(false.B)
    c.io.alloc.valid.poke(true.B)

    var waited = 0
    while (!c.io.alloc.ready.peek().litToBoolean && waited < maxWait) { c.clock.step(1); waited += 1 }
    assert(waited < maxWait, s"alloc of funct=$funct never accepted")
    c.clock.step(1)
    c.io.alloc.valid.poke(false.B)
  }

  behavior of "ReservationStation"

  it should "not let a PRELOAD targeting the same accumulator address as a pending LOAD become issuable until the LOAD completes (not merely issues)" in {
    test(new ReservationStationHazardHarness).withAnnotations(Seq(VerilatorBackendAnnotation)) { c =>
      // Encode one accumulator row address, used as the destination of
      // both the bias LOAD and the PRELOAD's accumulate target.
      c.io.encRow.poke(5.U)
      c.io.encAcc.poke(false.B)
      c.io.encFull.poke(false.B)
      val biasAddr = c.io.encOut.peek().litValue

      c.io.encRow.poke(5.U)
      c.io.encAcc.poke(true.B)
      c.io.encFull.poke(false.B)
      val accAddr = c.io.encOut.peek().litValue

      // 1) CONFIG_EX: set a_stride=1, c_stride=1 (needed so the PRELOAD's
      //    own encoded row-count -> address-range math isn't degenerate).
      val cfgExRs1 = BigInt(1) << 16 // a_stride := rs1(31,16) = 1
      val cfgExRs2 = BigInt(1) << 48 // c_stride := rs2(63,48) = 1
      doAlloc(c, CONFIG_CMD.litValue, cfgExRs1, cfgExRs2)

      // 2) LOAD3_CMD: bias mvin into accumulator row 5, exactly one row.
      val ldRs2 = (BigInt(1) << 48) | (BigInt(1) << 32) | biasAddr // rows=1, cols=1, dst=biasAddr
      doAlloc(c, LOAD3_CMD.litValue, BigInt(0), ldRs2)

      // 3) PRELOAD_CMD: preloads into a harmless, unrelated scratchpad
      //    address (rs1), but its accumulate destination (rs2) is the
      //    SAME accumulator row 5 -- this is what should conflict with
      //    the pending bias load.
      val preRs1 = (BigInt(1) << 48) | BigInt(0) // preload_rows=1, addr=0 (unrelated scratchpad addr)
      val preRs2 = (BigInt(1) << 48) | accAddr // preload_rows=1 (* c_stride=1 => range of 1), dst=accAddr
      doAlloc(c, PRELOAD_CMD.litValue, preRs1, preRs2)

      // --- Check 1: immediately after allocation, the PRELOAD must NOT be issuable ---
      c.io.issueLdReady.poke(false.B)
      c.io.issueExReady.poke(false.B)
      c.clock.step(1)
      assert(!c.io.issueExValid.peek().litToBoolean,
        "PRELOAD became issuable immediately, before the conflicting bias LOAD was even issued -- RAW hazard not detected")

      // --- Check 2: issuing (but not yet completing) the LOAD must NOT unblock the PRELOAD ---
      c.io.issueLdReady.poke(true.B)
      var waited = 0
      while (!c.io.issueLdValid.peek().litToBoolean && waited < 1000) { c.clock.step(1); waited += 1 }
      assert(waited < 1000, "bias LOAD never became issuable")
      val ldRobId = c.io.issueLdRobId.peek().litValue
      c.clock.step(1) // the LOAD issues on this edge
      c.io.issueLdReady.poke(false.B)

      c.clock.step(3) // a few cycles of "the load is now in flight but hasn't completed"
      assert(!c.io.issueExValid.peek().litToBoolean,
        "PRELOAD became issuable merely because the conflicting bias LOAD was ISSUED, before it COMPLETED -- " +
          "this is exactly the gap that could let ExecuteController accumulate before AccumulatorMem has committed the bias")

      // --- Check 3: completing the LOAD must unblock the PRELOAD ---
      c.io.completed.valid.poke(true.B)
      c.io.completed.bits.poke(ldRobId.U)
      c.clock.step(1)
      c.io.completed.valid.poke(false.B)

      waited = 0
      while (!c.io.issueExValid.peek().litToBoolean && waited < 1000) { c.clock.step(1); waited += 1 }
      assert(waited < 1000,
        "PRELOAD never became issuable even after the conflicting bias LOAD was marked completed")

      println("Confirmed: PRELOAD stayed blocked until the conflicting bias LOAD's completion, not just its issuance.")
    }
  }
}
