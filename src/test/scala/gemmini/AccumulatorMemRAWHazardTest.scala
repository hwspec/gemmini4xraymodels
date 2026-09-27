// Focused chiseltest unit test for a suspected RAW hazard in
// AccumulatorMem: does an accumulate-write immediately following a
// plain (bias) write to the SAME address correctly see the bias value,
// or does it read stale/pre-existing SRAM content because the bias
// write hasn't yet committed through the `pipelined_writes` delay line?
//
// This directly tests the hardware mechanism suspected (see
// ITANH_INVESTIGATION.md) to explain why a bias-controlled ITANH
// saturation test, and later a real-shape (KERNEL_DIM=3, IN_CHANNELS=64)
// PtychoNN-shaped test, both failed to show `q` reflecting the intended
// bias/accumulated magnitude. A closer reading of AccumulatorMem.scala
// found `io.write.ready`/`io.read.req.ready` guards (lines ~384-392)
// that appear to defend against exactly this hazard by backpressuring
// the requester -- this test checks whether that defense actually works
// when driven with a real, protocol-respecting Decoupled handshake.
package gemmini

import chisel3._
import chisel3.util._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import Arithmetic.SIntArithmetic._

// Minimal wrapper exposing a bare AccumulatorMem plus the external adder
// that Scratchpad.scala normally wires in via AccPipeShared(acc_latency-1, ...).
class AccumulatorMemHarness(val n: Int = 4, val acc_latency: Int = 2) extends Module {
  val accType = SInt(32.W)
  val t = Vec(1, Vec(1, accType))
  val scale_t = UInt(1.W)
  val scale_func = (x: SInt, y: UInt) => x

  val mem = Module(new AccumulatorMem(
    n, t, scale_func, scale_t,
    /*acc_singleported=*/false, /*acc_sub_banks=*/1,
    /*use_shared_ext_mem=*/false, /*use_tl_ext_ram=*/false,
    acc_latency, accType, /*is_dummy=*/false
  ))

  val io = IO(new Bundle {
    val write = Flipped(Decoupled(new AccumulatorWriteReq(n, t)))
    val read = Flipped(new AccumulatorReadIO(n, t, scale_t))
  })
  io.write <> mem.io.write
  io.read <> mem.io.read

  val added = Wire(t.cloneType)
  added(0)(0) := mem.io.adder.op1(0)(0) + mem.io.adder.op2(0)(0)
  mem.io.adder.sum := ShiftRegister(added, acc_latency - 1)
}

class AccumulatorMemRAWHazardTest extends AnyFlatSpec with ChiselScalatestTester {
  def doWrite(c: AccumulatorMemHarness, addr: Int, data: BigInt, acc: Boolean, maxWait: Int = 1000): Unit = {
    c.io.write.valid.poke(true.B)
    c.io.write.bits.addr.poke(addr.U)
    c.io.write.bits.data(0)(0).poke(data.S)
    c.io.write.bits.acc.poke(acc.B)
    c.io.write.bits.mask.foreach(_.poke(true.B))

    var waited = 0
    while (!c.io.write.ready.peek().litToBoolean && waited < maxWait) {
      c.clock.step(1)
      waited += 1
    }
    assert(waited < maxWait, s"write to addr=$addr acc=$acc never became ready")
    c.clock.step(1) // the write actually fires on this edge
    c.io.write.valid.poke(false.B)
  }

  def doRead(c: AccumulatorMemHarness, addr: Int, maxWait: Int = 1000): BigInt = {
    c.io.read.req.valid.poke(true.B)
    c.io.read.req.bits.addr.poke(addr.U)
    c.io.read.req.bits.full.poke(true.B)
    c.io.read.req.bits.fromDMA.poke(false.B)
    c.io.read.req.bits.act.poke(0.U)
    c.io.read.req.bits.scale.poke(0.U)
    c.io.read.req.bits.igelu_qb.poke(0.S)
    c.io.read.req.bits.igelu_qc_lo.poke(0.S)
    c.io.read.req.bits.igelu_qc_hi.poke(0.S)
    c.io.read.req.bits.iexp_qln2.poke(0.S)
    c.io.read.req.bits.iexp_qln2_inv.poke(0.S)
    c.io.read.resp.ready.poke(true.B)

    var waited = 0
    while (!c.io.read.req.ready.peek().litToBoolean && waited < maxWait) {
      c.clock.step(1)
      waited += 1
    }
    assert(waited < maxWait, s"read of addr=$addr never became ready")
    c.clock.step(1) // the read request fires on this edge
    c.io.read.req.valid.poke(false.B)

    waited = 0
    while (!c.io.read.resp.valid.peek().litToBoolean && waited < maxWait) {
      c.clock.step(1)
      waited += 1
    }
    assert(waited < maxWait, s"read response for addr=$addr never arrived")
    val result = c.io.read.resp.bits.data(0)(0).peek().litValue
    c.clock.step(1)
    c.io.read.resp.ready.poke(false.B)
    result
  }

  behavior of "AccumulatorMem"

  it should "correctly retain a bias value across an immediately-following single zero-weight accumulate" in {
    test(new AccumulatorMemHarness(n = 4, acc_latency = 2)) { c =>
      val BIAS1 = BigInt(12345)
      doWrite(c, addr = 0, data = BIAS1, acc = false)
      doWrite(c, addr = 0, data = 0, acc = true)
      val result1 = doRead(c, addr = 0)
      assert(result1 == BIAS1, s"expected $BIAS1, got $result1")
    }
  }

  it should "correctly retain a bias value across nine immediately-following zero-weight accumulates (PtychoNN's real KERNEL_DIM=3/IN_CHANNELS=64 multi-tap shape)" in {
    test(new AccumulatorMemHarness(n = 4, acc_latency = 2)) { c =>
      val BIAS2 = BigInt(54321)
      doWrite(c, addr = 1, data = BIAS2, acc = false)
      for (_ <- 0 until 9) {
        doWrite(c, addr = 1, data = 0, acc = true)
      }
      val result2 = doRead(c, addr = 1)
      assert(result2 == BIAS2, s"expected $BIAS2, got $result2")
    }
  }

  it should "correctly compute bias + a real non-zero accumulate" in {
    test(new AccumulatorMemHarness(n = 4, acc_latency = 2)) { c =>
      val BIAS3 = BigInt(1000)
      val DELTA3 = BigInt(777)
      doWrite(c, addr = 2, data = BIAS3, acc = false)
      doWrite(c, addr = 2, data = DELTA3, acc = true)
      val result3 = doRead(c, addr = 2)
      assert(result3 == BIAS3 + DELTA3, s"expected ${BIAS3 + DELTA3}, got $result3")
    }
  }
}
