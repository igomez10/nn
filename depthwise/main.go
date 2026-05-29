// convolutions.go — standard vs depthwise separable convolutions, no dependencies.
//
// Standard convolution:   every output channel sees every input channel.
// Depthwise separable:    two steps —
//   1. depthwise conv: each input channel filtered independently
//   2. pointwise conv: 1×1 conv mixes channels together
//
// Run: go run convolutions.go

package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// ─── Tensor ──────────────────────────────────────────────────────────────────

// Tensor4D is [batch, channels, height, width].
type Tensor4D struct {
	data                               []float64
	BatchSize, Channels, Height, Width int
}

func NewTensor(b, c, h, w int) *Tensor4D {
	return &Tensor4D{data: make([]float64, b*c*h*w), BatchSize: b, Channels: c, Height: h, Width: w}
}

func (t *Tensor4D) at(b, c, h, w int) *float64 {
	return &t.data[b*t.Channels*t.Height*t.Width+c*t.Height*t.Width+h*t.Width+w]
}

func (t *Tensor4D) Get(b, c, h, w int) float64    { return *t.at(b, c, h, w) }
func (t *Tensor4D) Set(b, c, h, w int, v float64) { *t.at(b, c, h, w) = v }
func (t *Tensor4D) Add(b, c, h, w int, v float64) { *t.at(b, c, h, w) += v }

func (t *Tensor4D) Shape() string {
	return fmt.Sprintf("[%d×%d×%d×%d]", t.BatchSize, t.Channels, t.Height, t.Width)
}

// ─── Weight initialization ────────────────────────────────────────────────────

func randomTensor(rng *rand.Rand, b, c, h, w int) *Tensor4D {
	t := NewTensor(b, c, h, w)
	scale := math.Sqrt(2.0 / float64(c*h*w)) // He init
	for i := range t.data {
		t.data[i] = rng.NormFloat64() * scale
	}
	return t
}

// ─── Standard convolution ─────────────────────────────────────────────────────
//
// Kernel shape: [outC, inC, kH, kW]
// Each output channel is a sum over all input channels.

func standardConv(input *Tensor4D, kernel *Tensor4D, stride, pad int) *Tensor4D {
	BatchSize, inputChannels, inputHeight, inputWidth := input.BatchSize, input.Channels, input.Height, input.Width
	outputChannels, kernelHeight, kernelWidth := kernel.BatchSize, kernel.Height, kernel.Width // kernel.B = outChannels

	outH := (inputHeight+2*pad-kernelHeight)/stride + 1
	outW := (inputWidth+2*pad-kernelWidth)/stride + 1

	out := NewTensor(BatchSize, outputChannels, outH, outW)

	for b := range BatchSize {
		for oc := range outputChannels { // output channel
			for oh := range outH {
				for ow := range outW {
					sum := 0.0
					for ic := range inputChannels { // sum over input channels
						for kh := range kernelHeight {
							for kw := range kernelWidth {
								ih := oh*stride - pad + kh
								iw := ow*stride - pad + kw
								if ih < 0 || ih >= inputHeight || iw < 0 || iw >= inputWidth {
									continue // zero-pad
								}
								sum += input.Get(b, ic, ih, iw) *
									kernel.Get(oc, ic, kh, kw)
							}
						}
					}
					out.Set(b, oc, oh, ow, sum)
				}
			}
		}
	}
	return out
}

func standardConvParams(inC, outC, kH, kW int) int {
	return outC * inC * kH * kW
}

// ─── Depthwise separable convolution ─────────────────────────────────────────
//
// Step 1 — depthwise:  kernel shape [inC, 1, kH, kW]
//   Each input channel has its own spatial filter. No cross-channel mixing.
//
// Step 2 — pointwise: kernel shape [outC, inC, 1, 1]
//   A 1×1 conv that projects the depthwise features into outC channels.

func depthwiseConv(input *Tensor4D, kernel *Tensor4D, stride, pad int) *Tensor4D {
	B, inC, inH, inW := input.BatchSize, input.Channels, input.Height, input.Width
	kH, kW := kernel.Height, kernel.Width

	outH := (inH+2*pad-kH)/stride + 1
	outW := (inW+2*pad-kW)/stride + 1

	// Output has the same number of channels as input (one filter per channel).
	out := NewTensor(B, inC, outH, outW)

	for b := range B {
		for c := range inC { // each channel processed independently
			for oh := range outH {
				for ow := range outW {
					sum := 0.0
					for kh := range kH {
						for kw := range kW {
							ih := oh*stride - pad + kh
							iw := ow*stride - pad + kw
							if ih < 0 || ih >= inH || iw < 0 || iw >= inW {
								continue
							}
							sum += input.Get(b, c, ih, iw) *
								kernel.Get(c, 0, kh, kw) // kernel.C==inC, kernel.B==1
						}
					}
					out.Set(b, c, oh, ow, sum)
				}
			}
		}
	}
	return out
}

// pointwiseConv is a 1×1 conv: mixes channels without spatial filtering.
func pointwiseConv(input *Tensor4D, kernel *Tensor4D) *Tensor4D {
	B, inC, H, W := input.BatchSize, input.Channels, input.Height, input.Width
	outC := kernel.BatchSize

	out := NewTensor(B, outC, H, W)

	for b := range B {
		for oc := range outC {
			for h := range H {
				for w := range W {
					sum := 0.0
					for ic := range inC {
						sum += input.Get(b, ic, h, w) * kernel.Get(oc, ic, 0, 0)
					}
					out.Set(b, oc, h, w, sum)
				}
			}
		}
	}
	return out
}

func depthwiseSeparableConv(
	input *Tensor4D,
	dwKernel *Tensor4D, // depthwise kernel: [inC, 1, kH, kW]
	pwKernel *Tensor4D, // pointwise kernel: [outC, inC, 1, 1]
	stride, pad int,
) *Tensor4D {
	depthwiseOut := depthwiseConv(input, dwKernel, stride, pad)
	return pointwiseConv(depthwiseOut, pwKernel)
}

func depthwiseSeparableParams(inC, outC, kH, kW int) int {
	depthwise := inC * kH * kW // one spatial filter per input channel
	pointwise := outC * inC * 1 * 1
	return depthwise + pointwise
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

func elapsed(label string, fn func()) time.Duration {
	start := time.Now()
	fn()
	d := time.Since(start)
	fmt.Printf("  %-30s %v\n", label+":", d)
	return d
}

func paramReduction(std, dws int) float64 {
	return (1 - float64(dws)/float64(std)) * 100
}

// ─── Main ─────────────────────────────────────────────────────────────────────

func main() {
	rng := rand.New(rand.NewSource(42))

	// ── Configuration ────────────────────────────────────────────────────────
	batch := 1
	inC := 32  // input channels
	outC := 64 // output channels
	inH := 56  // spatial height
	inW := 56  // spatial width
	kSize := 3 // 3×3 kernel
	stride := 1
	pad := 1 // "same" padding

	fmt.Println("════════════════════════════════════════════════════════════")
	fmt.Println(" Standard vs Depthwise Separable Convolution")
	fmt.Println("════════════════════════════════════════════════════════════")
	fmt.Printf("\n  Input:    %d×%d spatial, %d channels, batch %d\n", inH, inW, inC, batch)
	fmt.Printf("  Output:   %d channels\n", outC)
	fmt.Printf("  Kernel:   %d×%d, stride %d, pad %d\n\n", kSize, kSize, stride, pad)

	input := randomTensor(rng, batch, inC, inH, inW)

	// ── Standard convolution ─────────────────────────────────────────────────
	fmt.Println("─── Standard convolution ───────────────────────────────────")
	fmt.Println()
	fmt.Println("  Each of the", outC, "output filters slides over ALL", inC, "input")
	fmt.Println("  channels simultaneously. The filter shape is:")
	fmt.Printf("  [outC=%d, inC=%d, kH=%d, kW=%d]\n\n", outC, inC, kSize, kSize)

	stdKernel := randomTensor(rng, outC, inC, kSize, kSize)
	stdParams := standardConvParams(inC, outC, kSize, kSize)

	var stdOut *Tensor4D
	stdTime := elapsed("Standard conv", func() {
		stdOut = standardConv(input, stdKernel, stride, pad)
	})

	fmt.Printf("  Input:   %s\n", input.Shape())
	fmt.Printf("  Kernel:  %s\n", stdKernel.Shape())
	fmt.Printf("  Output:  %s\n", stdOut.Shape())
	fmt.Printf("  Params:  %d × %d × %d × %d = %s\n",
		outC, inC, kSize, kSize, thousands(stdParams))

	// ── Depthwise separable convolution ──────────────────────────────────────
	fmt.Println()
	fmt.Println("─── Depthwise separable convolution ────────────────────────")
	fmt.Println()
	fmt.Println("  Step 1 — depthwise:  one spatial filter per input channel.")
	fmt.Printf("  Depthwise kernel:  [inC=%d, 1, kH=%d, kW=%d]\n\n", inC, kSize, kSize)
	fmt.Println("  Step 2 — pointwise: 1×1 conv mixes channels together.")
	fmt.Printf("  Pointwise kernel:  [outC=%d, inC=%d, 1, 1]\n\n", outC, inC)

	dwKernel := randomTensor(rng, inC, 1, kSize, kSize) // depthwise: one per channel
	pwKernel := randomTensor(rng, outC, inC, 1, 1)      // pointwise: channel mixer

	dwParams := inC * kSize * kSize
	pwParams := outC * inC
	dwsParams := depthwiseSeparableParams(inC, outC, kSize, kSize)

	var dwsOut *Tensor4D
	dwsTime := elapsed("Depthwise separable conv", func() {
		dwsOut = depthwiseSeparableConv(input, dwKernel, pwKernel, stride, pad)
	})

	fmt.Printf("  Input:    %s\n", input.Shape())
	fmt.Printf("  DW kern:  %s\n", dwKernel.Shape())
	fmt.Printf("  PW kern:  %s\n", pwKernel.Shape())
	fmt.Printf("  Output:   %s\n", dwsOut.Shape())
	fmt.Printf("  DW params: %d × %d × %d = %s\n", inC, kSize, kSize, thousands(dwParams))
	fmt.Printf("  PW params: %d × %d × 1 × 1 = %s\n", outC, inC, thousands(pwParams))
	fmt.Printf("  Total params: %s\n", thousands(dwsParams))

	// ── Comparison ───────────────────────────────────────────────────────────
	fmt.Println()
	fmt.Println("════════════════════════════════════════════════════════════")
	fmt.Println(" Comparison")
	fmt.Println("════════════════════════════════════════════════════════════")
	fmt.Println()
	fmt.Printf("  %-24s  %10s  %10s\n", "Metric", "Standard", "DW-Sep")
	fmt.Printf("  %-24s  %10s  %10s\n",
		"──────────────────────", "──────────", "──────────")
	fmt.Printf("  %-24s  %10s  %10s\n",
		"Output shape", stdOut.Shape()[1:], dwsOut.Shape()[1:])
	fmt.Printf("  %-24s  %10s  %10s\n",
		"Parameters", thousands(stdParams), thousands(dwsParams))
	fmt.Printf("  %-24s  %10.1f  %10.1f\n",
		"Time (ms)", float64(stdTime.Microseconds())/1000,
		float64(dwsTime.Microseconds())/1000)

	reduction := paramReduction(stdParams, dwsParams)
	ratio := float64(dwsParams) / float64(stdParams)
	fmt.Println()
	fmt.Printf("  Parameter reduction:  %.1f%%\n", reduction)
	fmt.Printf("  DWS / standard ratio: %.3f  (≈ 1/%d + 1/%d)\n",
		ratio, outC, kSize*kSize)

	fmt.Println()
	fmt.Println("  The theoretical ratio for a k×k kernel, inC input channels,")
	fmt.Println("  and outC output channels is:")
	fmt.Printf("  (1/outC + 1/k²) = (1/%d + 1/%d) = %.3f\n",
		outC, kSize*kSize, 1.0/float64(outC)+1.0/float64(kSize*kSize))
	fmt.Println()
	fmt.Println("  This is why MobileNet, Xception, and EfficientNet use")
	fmt.Println("  depthwise separable convolutions: same output shape, far")
	fmt.Println("  fewer multiplications — critical on mobile devices.")
	fmt.Println()
}

func thousands(n int) string {
	s := fmt.Sprintf("%d", n)
	out := []byte{}
	for i, ch := range s {
		rem := len(s) - i
		if i > 0 && rem%3 == 0 {
			out = append(out, ',')
		}
		out = append(out, byte(ch))
	}
	return string(out)
}
