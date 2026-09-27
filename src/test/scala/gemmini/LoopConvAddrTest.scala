// Chiseltest unit test checking whether LoopConvLdBias (which loads each
// output pixel's bias into the accumulator, once) and LoopConvExecute
// (which issues the actual PRELOAD+COMPUTE pairs that accumulate into the
// SAME accumulator locations, once per (krow,kcol,kch-chunk) tap) ever
// compute DIFFERENT addresses for what should be the same (batch, orow,
// ocol, och-chunk) output position.
//
// Context (see ITANH_INVESTIGATION.md): a chiseltest-based check of
// AccumulatorMem itself found no RAW hazard there (three PASSED cases).
// The remaining candidate explanations for why a bias-controlled ITANH
// test at PtychoNN's real layer shape (KERNEL_DIM=3, IN_CHANNELS=64)
// never shows the intended magnitude include a same-address MISMATCH
// between LoopConvLdBias's bias-load address and LoopConvExecute's
// accumulate address -- this test checks exactly that, directly, rather
// than trusting that the two textually-similar address formulas
// (LoopConv.scala:125 and :669-670) actually walk through matching
// (b,orow,ocol,och) sequences when driven independently.
//
// Scope/limitation: this drives LoopConvLdBias and LoopConvExecute
// directly with hand-constructed, IDENTICAL request bundles (same
// addr_start/c_addr_start base, same outer/inner bounds) -- exactly what
// SHOULD happen for a single (non-pipelined) conv call. It does NOT
// exercise the top-level LoopConv orchestration's own head/tail
// concurrent-loop bookkeeping (which independently tracks
// ld_bias_addr_start/ex_c_addr_start in separate registers) -- if the
// real bug is in THAT bookkeeping feeding the two sub-modules
// inconsistent bases, this test would not catch it. See
// ITANH_INVESTIGATION.md for the full reasoning.
//
// Dimensions are a scaled-down but structurally equivalent proxy for
// PtychoNN's real ITANH layer (IN_CHANNELS=64, OUT_CHANNELS=1,
// KERNEL_DIM=3, IN_DIM=OUT_DIM=64): out_channels=1 (< block_size, one
// och-chunk, matching the real layer), in_channels=32 (two 16-wide
// kch-chunks, exercising the same channel-tiling the real 64-channel
// layer needs, just with fewer chunks for speed), kernel=3x3 (nine taps,
// matching exactly), orows=2/ocols=20 (multiple output rows, and ocols
// wide enough to force two 16-wide column-chunks -- exercising the same
// "ocol advances by block_size per command" chunking the real 64-wide
// layer needs, again with fewer chunks for speed).
package gemmini

import chisel3._
import chisel3.util._
import chiseltest._
import chiseltest.experimental.expose
import org.scalatest.flatspec.AnyFlatSpec
import org.chipsalliance.cde.config.{Config, Parameters}
import freechips.rocketchip.tile.{TileKey, RocketTileParams}
import GemminiISA._

class LoopConvAddrHarness(implicit p: Parameters) extends Module {
  val block_size = 16
  val large_iterator_bitwidth = 16
  val small_iterator_bitwidth = 16
  val tiny_iterator_bitwidth = 16
  val coreMaxAddrBits = 32
  val max_addr = 1 << 16
  val max_acc_addr = 1 << 16
  val acc_w = 32
  val max_block_len_acc = 4
  val concurrent_loops = 2
  val latency = 2

  val local_addr_t = new LocalAddr(4, 256, 2, 256)
  val config_mvin_rs1_t = new ConfigMvinRs1(32, 16, 8)
  val mvin_rs2_t = new MvinRs2(16, 16, local_addr_t)
  val config_ex_rs1_t = new ConfigExRs1(32)
  val preload_rs_t = new PreloadRs(16, 16, local_addr_t)
  val compute_rs_t = new ComputeRs(16, 16, local_addr_t)

  val ld_bias = Module(new LoopConvLdBias(
    block_size, coreMaxAddrBits, large_iterator_bitwidth, small_iterator_bitwidth, tiny_iterator_bitwidth,
    max_acc_addr, acc_w, max_block_len_acc, concurrent_loops, latency, config_mvin_rs1_t, mvin_rs2_t))
  val ex = Module(new LoopConvExecute(
    block_size, large_iterator_bitwidth, small_iterator_bitwidth, tiny_iterator_bitwidth, max_addr, max_acc_addr,
    concurrent_loops, latency, config_ex_rs1_t, preload_rs_t, preload_rs_t, compute_rs_t, compute_rs_t))

  ld_bias.io.cmd.ready := true.B
  ld_bias.io.rob_overloaded := false.B
  ld_bias.io.wait_for_prev_loop := false.B

  ex.io.cmd.ready := true.B
  ex.io.rob_overloaded := false.B
  ex.io.lda_completed := true.B
  ex.io.ldb_completed := true.B
  ex.io.ldd_completed := true.B

  val io = IO(new Bundle {
    val ldBiasReq = Flipped(Decoupled(new LoopConvLdBiasReq(
      coreMaxAddrBits, large_iterator_bitwidth, small_iterator_bitwidth, tiny_iterator_bitwidth, max_acc_addr, concurrent_loops)))
    val exReq = Flipped(Decoupled(new LoopConvExecuteReq(
      large_iterator_bitwidth, small_iterator_bitwidth, tiny_iterator_bitwidth, max_addr, max_acc_addr, concurrent_loops)))
  })
  io.ldBiasReq <> ld_bias.io.req
  io.exReq <> ex.io.req

  // Internal signals, exposed for the testbench (expose() is needed here
  // because these are reached through a grandchild module (command_p) or
  // are otherwise-dangling submodule outputs that a normal Chisel
  // connection/peek would not reliably reach).
  val ldBiasIdle = expose(ld_bias.io.idle)
  val ldBiasCmdInValid = expose(ld_bias.command_p.io.in.valid)
  val ldBiasCmdInReady = expose(ld_bias.command_p.io.in.ready)
  val ldBiasCmdInFunct = expose(ld_bias.command_p.io.in.bits.cmd.inst.funct)
  val ldBiasSpadAddr = expose(ld_bias.spad_addr)
  val ldBiasB = expose(ld_bias.b)
  val ldBiasOrow = expose(ld_bias.orow)
  val ldBiasOcol = expose(ld_bias.ocol)
  val ldBiasOch = expose(ld_bias.och)

  val exIdle = expose(ex.io.idle)
  val exCmdInValid = expose(ex.command_p.io.in.valid)
  val exCmdInReady = expose(ex.command_p.io.in.ready)
  val exCmdInFunct = expose(ex.command_p.io.in.bits.cmd.inst.funct)
  val exCAddr = expose(ex.c_addr)
  val exB = expose(ex.b)
  val exOrow = expose(ex.orow)
  val exOcol = expose(ex.ocol)
  val exOch = expose(ex.och)
}

class LoopConvAddrTest extends AnyFlatSpec with ChiselScalatestTester {
  implicit val p: Parameters = new Config((site, here, up) => {
    case TileKey => RocketTileParams()
  })

  // Scaled-down proxy for PtychoNN's real ITANH-layer conv shape -- see
  // file header comment for the full justification of these numbers.
  val BATCHES = 1
  val OROWS = 2
  val OCOLS = 20
  val OCHS = 1     // out_channels (< block_size == one och-chunk, matching the real layer)
  val KROWS = 3
  val KCOLS = 3
  val KCHS = 32    // in_channels (two 16-wide kch-chunks)
  val PAD = 1
  val STRIDE = 1
  val BASE_ADDR = 100 // same base for both ld_bias and ex -- what SHOULD happen for one conv call

  behavior of "LoopConvLdBias and LoopConvExecute"

  it should "compute matching accumulator addresses for every (batch,orow,ocol,och) position, across every (krow,kcol,kch-chunk) tap" in {
    test(new LoopConvAddrHarness) { c =>
      // --- Fire the LoopConvLdBias request ---
      c.io.ldBiasReq.bits.outer_bounds.batch_size.poke(BATCHES.U)
      c.io.ldBiasReq.bits.outer_bounds.in_row_dim.poke(0.U)
      c.io.ldBiasReq.bits.outer_bounds.in_col_dim.poke(0.U)
      c.io.ldBiasReq.bits.outer_bounds.in_channels.poke(0.U)
      c.io.ldBiasReq.bits.outer_bounds.out_channels.poke(OCHS.U)
      c.io.ldBiasReq.bits.outer_bounds.out_col_dim.poke(OCOLS.U)
      c.io.ldBiasReq.bits.outer_bounds.out_row_dim.poke(OROWS.U)
      c.io.ldBiasReq.bits.outer_bounds.out_stride.poke(0.U)
      c.io.ldBiasReq.bits.outer_bounds.in_stride.poke(0.U)
      c.io.ldBiasReq.bits.outer_bounds.weight_stride.poke(0.U)
      c.io.ldBiasReq.bits.outer_bounds.pool_out_row_dim.poke(0.U)
      c.io.ldBiasReq.bits.outer_bounds.pool_out_col_dim.poke(0.U)
      c.io.ldBiasReq.bits.outer_bounds.stride.poke(STRIDE.U)
      c.io.ldBiasReq.bits.outer_bounds.padding.poke(PAD.U)
      c.io.ldBiasReq.bits.outer_bounds.kernel_dim.poke(KROWS.U)
      c.io.ldBiasReq.bits.outer_bounds.kernel_dilation.poke(1.U)
      c.io.ldBiasReq.bits.outer_bounds.pool_size.poke(0.U)
      c.io.ldBiasReq.bits.outer_bounds.pool_stride.poke(0.U)
      c.io.ldBiasReq.bits.outer_bounds.pool_padding.poke(0.U)

      c.io.ldBiasReq.bits.inner_bounds.batches.poke(BATCHES.U)
      c.io.ldBiasReq.bits.inner_bounds.porows.poke(0.U)
      c.io.ldBiasReq.bits.inner_bounds.pocols.poke(0.U)
      c.io.ldBiasReq.bits.inner_bounds.pochs.poke(OCHS.U)
      c.io.ldBiasReq.bits.inner_bounds.krows.poke(KROWS.U)
      c.io.ldBiasReq.bits.inner_bounds.kcols.poke(KCOLS.U)
      c.io.ldBiasReq.bits.inner_bounds.kchs.poke(KCHS.U)
      c.io.ldBiasReq.bits.inner_bounds.lpad.poke(PAD.U)
      c.io.ldBiasReq.bits.inner_bounds.rpad.poke(PAD.U)
      c.io.ldBiasReq.bits.inner_bounds.upad.poke(PAD.U)
      c.io.ldBiasReq.bits.inner_bounds.dpad.poke(PAD.U)
      c.io.ldBiasReq.bits.inner_bounds.plpad.poke(0.U)
      c.io.ldBiasReq.bits.inner_bounds.prad.poke(0.U)
      c.io.ldBiasReq.bits.inner_bounds.pupad.poke(0.U)
      c.io.ldBiasReq.bits.inner_bounds.pdpad.poke(0.U)
      c.io.ldBiasReq.bits.inner_bounds.orows.poke(OROWS.U)
      c.io.ldBiasReq.bits.inner_bounds.ocols.poke(OCOLS.U)

      c.io.ldBiasReq.bits.derived_params.ochs.poke(OCHS.U)
      c.io.ldBiasReq.bits.derived_params.irows.poke(0.U)
      c.io.ldBiasReq.bits.derived_params.icols.poke(0.U)
      c.io.ldBiasReq.bits.derived_params.irows_unpadded.poke(0.U)
      c.io.ldBiasReq.bits.derived_params.icols_unpadded.poke(0.U)
      c.io.ldBiasReq.bits.derived_params.ichs.poke(0.U)
      c.io.ldBiasReq.bits.derived_params.out_channels_per_bank.poke(1.U)
      c.io.ldBiasReq.bits.derived_params.in_channels_per_bank.poke(0.U)
      c.io.ldBiasReq.bits.derived_params.bias_spad_stride.poke((BATCHES * OROWS * OCOLS).U)
      c.io.ldBiasReq.bits.derived_params.input_spad_stride.poke(0.U)
      c.io.ldBiasReq.bits.derived_params.weight_spad_stride.poke(0.U)

      c.io.ldBiasReq.bits.addr_start.poke(BASE_ADDR.U)
      c.io.ldBiasReq.bits.dram_addr.poke(0x1000.U) // must be non-zero, or ld_bias skips entirely
      c.io.ldBiasReq.bits.no_bias.poke(false.B)
      c.io.ldBiasReq.bits.loop_id.poke(0.U)
      c.io.ldBiasReq.valid.poke(true.B)

      var waited = 0
      while (!c.io.ldBiasReq.ready.peek().litToBoolean && waited < 1000) { c.clock.step(1); waited += 1 }
      assert(waited < 1000, "ld_bias request never accepted")
      c.clock.step(1)
      c.io.ldBiasReq.valid.poke(false.B)

      // --- Fire the LoopConvExecute request (same BASE_ADDR / shape) ---
      val irows = OROWS * STRIDE + KROWS - 1
      val icols = OCOLS * STRIDE + KCOLS - 1

      c.io.exReq.bits.outer_bounds.batch_size.poke(BATCHES.U)
      c.io.exReq.bits.outer_bounds.in_row_dim.poke(0.U)
      c.io.exReq.bits.outer_bounds.in_col_dim.poke(0.U)
      c.io.exReq.bits.outer_bounds.in_channels.poke(0.U)
      c.io.exReq.bits.outer_bounds.out_channels.poke(OCHS.U)
      c.io.exReq.bits.outer_bounds.out_col_dim.poke(OCOLS.U)
      c.io.exReq.bits.outer_bounds.out_row_dim.poke(OROWS.U)
      c.io.exReq.bits.outer_bounds.out_stride.poke(0.U)
      c.io.exReq.bits.outer_bounds.in_stride.poke(0.U)
      c.io.exReq.bits.outer_bounds.weight_stride.poke(0.U)
      c.io.exReq.bits.outer_bounds.pool_out_row_dim.poke(0.U)
      c.io.exReq.bits.outer_bounds.pool_out_col_dim.poke(0.U)
      c.io.exReq.bits.outer_bounds.stride.poke(STRIDE.U)
      c.io.exReq.bits.outer_bounds.padding.poke(PAD.U)
      c.io.exReq.bits.outer_bounds.kernel_dim.poke(KROWS.U)
      c.io.exReq.bits.outer_bounds.kernel_dilation.poke(1.U)
      c.io.exReq.bits.outer_bounds.pool_size.poke(0.U)
      c.io.exReq.bits.outer_bounds.pool_stride.poke(0.U)
      c.io.exReq.bits.outer_bounds.pool_padding.poke(0.U)

      c.io.exReq.bits.inner_bounds.batches.poke(BATCHES.U)
      c.io.exReq.bits.inner_bounds.porows.poke(0.U)
      c.io.exReq.bits.inner_bounds.pocols.poke(0.U)
      c.io.exReq.bits.inner_bounds.pochs.poke(OCHS.U)
      c.io.exReq.bits.inner_bounds.krows.poke(KROWS.U)
      c.io.exReq.bits.inner_bounds.kcols.poke(KCOLS.U)
      c.io.exReq.bits.inner_bounds.kchs.poke(KCHS.U)
      c.io.exReq.bits.inner_bounds.lpad.poke(PAD.U)
      c.io.exReq.bits.inner_bounds.rpad.poke(PAD.U)
      c.io.exReq.bits.inner_bounds.upad.poke(PAD.U)
      c.io.exReq.bits.inner_bounds.dpad.poke(PAD.U)
      c.io.exReq.bits.inner_bounds.plpad.poke(0.U)
      c.io.exReq.bits.inner_bounds.prad.poke(0.U)
      c.io.exReq.bits.inner_bounds.pupad.poke(0.U)
      c.io.exReq.bits.inner_bounds.pdpad.poke(0.U)
      c.io.exReq.bits.inner_bounds.orows.poke(OROWS.U)
      c.io.exReq.bits.inner_bounds.ocols.poke(OCOLS.U)

      c.io.exReq.bits.derived_params.ochs.poke(OCHS.U)
      c.io.exReq.bits.derived_params.irows.poke(irows.U)
      c.io.exReq.bits.derived_params.icols.poke(icols.U)
      c.io.exReq.bits.derived_params.irows_unpadded.poke((irows - 2 * PAD).U)
      c.io.exReq.bits.derived_params.icols_unpadded.poke((icols - 2 * PAD).U)
      c.io.exReq.bits.derived_params.ichs.poke(KCHS.U)
      c.io.exReq.bits.derived_params.out_channels_per_bank.poke(1.U)
      c.io.exReq.bits.derived_params.in_channels_per_bank.poke(2.U)
      c.io.exReq.bits.derived_params.bias_spad_stride.poke((BATCHES * OROWS * OCOLS).U)
      c.io.exReq.bits.derived_params.input_spad_stride.poke((BATCHES * irows * icols).U)
      c.io.exReq.bits.derived_params.weight_spad_stride.poke((KROWS * KCOLS * KCHS).U)

      c.io.exReq.bits.a_addr_start.poke(0.U)
      c.io.exReq.bits.b_addr_end.poke(60000.U)
      c.io.exReq.bits.c_addr_start.poke(BASE_ADDR.U)
      c.io.exReq.bits.wrot180.poke(false.B)
      c.io.exReq.bits.downsample.poke(false.B)
      c.io.exReq.bits.max_pixels_per_row.poke(1.U)
      c.io.exReq.bits.input_dilated.poke(false.B)
      c.io.exReq.bits.trans_weight_0132.poke(false.B)
      c.io.exReq.bits.trans_input_3120.poke(false.B)
      c.io.exReq.bits.loop_id.poke(0.U)
      c.io.exReq.valid.poke(true.B)

      waited = 0
      while (!c.io.exReq.ready.peek().litToBoolean && waited < 1000) { c.clock.step(1); waited += 1 }
      assert(waited < 1000, "ex request never accepted")
      c.clock.step(1)
      c.io.exReq.valid.poke(false.B)

      // --- Drain both modules, recording every bias-load address and every accumulate address ---
      case class Pos(b: Int, orow: Int, ocol: Int, och: Int)
      val biasAddrs = scala.collection.mutable.Map[Pos, BigInt]()
      val accAddrsByPos = scala.collection.mutable.Map[Pos, scala.collection.mutable.Set[BigInt]]()

      var cycles = 0
      val MAX_CYCLES = 20000
      while (cycles < MAX_CYCLES && !(c.ldBiasIdle.peek().litToBoolean && c.exIdle.peek().litToBoolean)) {
        val ldBiasFire = c.ldBiasCmdInValid.peek().litToBoolean && c.ldBiasCmdInReady.peek().litToBoolean
        if (ldBiasFire && c.ldBiasCmdInFunct.peek().litValue == LOAD3_CMD.litValue) {
          val pos = Pos(c.ldBiasB.peek().litValue.toInt, c.ldBiasOrow.peek().litValue.toInt,
            c.ldBiasOcol.peek().litValue.toInt, c.ldBiasOch.peek().litValue.toInt)
          biasAddrs(pos) = c.ldBiasSpadAddr.peek().litValue
        }
        val exFire = c.exCmdInValid.peek().litToBoolean && c.exCmdInReady.peek().litToBoolean
        if (exFire && c.exCmdInFunct.peek().litValue == PRELOAD_CMD.litValue) {
          val pos = Pos(c.exB.peek().litValue.toInt, c.exOrow.peek().litValue.toInt,
            c.exOcol.peek().litValue.toInt, c.exOch.peek().litValue.toInt)
          accAddrsByPos.getOrElseUpdate(pos, scala.collection.mutable.Set()) += c.exCAddr.peek().litValue
        }
        c.clock.step(1)
        cycles += 1
      }
      assert(cycles < MAX_CYCLES, "simulation did not finish within MAX_CYCLES")

      println(s"Collected ${biasAddrs.size} distinct bias-load positions, ${accAddrsByPos.size} distinct accumulate positions")
      assert(biasAddrs.nonEmpty, "no bias-load commands were ever observed")
      assert(accAddrsByPos.nonEmpty, "no accumulate (PRELOAD) commands were ever observed")

      // Every position ld_bias loaded a bias for must also have been
      // accumulated into by ex, at the SAME address, for EVERY tap.
      assert(biasAddrs.keySet == accAddrsByPos.keySet,
        s"position sets differ! bias-only: ${biasAddrs.keySet -- accAddrsByPos.keySet}, " +
          s"accumulate-only: ${accAddrsByPos.keySet -- biasAddrs.keySet}")

      for ((pos, biasAddr) <- biasAddrs) {
        val accAddrs = accAddrsByPos(pos)
        assert(accAddrs == Set(biasAddr),
          s"MISMATCH at $pos: bias-load address=$biasAddr, but accumulate addresses seen=$accAddrs")
      }

      println(s"All ${biasAddrs.size} positions matched: bias-load address == every accumulate address, across all taps.")
    }
  }
}
